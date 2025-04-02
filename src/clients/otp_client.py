### implement intelligent client class
import os
import csv
import json
import torch
import pandas as pd
import numpy as np
import flwr as fl
import threading
from torch import tensor
from scipy.optimize import curve_fit
from collections import OrderedDict
from config_folder import client_config_file, agent_config_file
from utils.client_cuda_train import client_parallel_train_step
from data.otp_client_dataset import get_action_partitioned_data
from utils.client_train_test_utils import client_train_step, client_test_step, client_test_metrics

#agent classes for training and taking actions during server communication rounds
from src.agent.ddpg import DDPG
from src.clients.ntp_client import get_naive_action #only needed to get action when action type is None - for ablation experiments
from config_folder import agent_config_file
from config_folder import client_config_file
from utils.agent_train_test_utils import exp_loss
from utils.agent_action_reward import AgentAction, get_class_counts_from_dataloader, get_reward

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

#set up a flower client
class OTPClient(fl.client.NumPyClient):
    def __init__(
            self, client_id, model, train_dataloader, val_dataloader, \
            public_partition_dataloader, neutral_dataloader, dataset_name, feature_name, \
            label_name, loss_fn, optimizer_config, num_classes, optimize_metric, meta_action_type, \
            experiment_type, dataset_id, seed, device, verbose
        ):
        super().__init__()
        self.client_id = client_id
        self.model = model
        self.model_clone = model #create a copy of the server cloned parameters for local fine-tuning
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.public_partition_dataloader = public_partition_dataloader
        self.neutral_dataloader = neutral_dataloader
        self.dataset_name = dataset_name
        self.feature_name = feature_name
        self.label_name = label_name
        self.loss_fn = loss_fn
        self.optimizer_config = optimizer_config
        self.num_classes = num_classes
        self.seed = seed
        self.device = device
        self.verbose = verbose
        self.fine_tune = True
        # self.train_agent = False
        self.estimated_loss = None
        self.optimize_metric = optimize_metric
        self.experiment_type = experiment_type
        self.dataset_id = dataset_id
        self.meta_action_type = meta_action_type
        self.total_server_rounds = client_config_file.TOTAL_SERVER_ROUNDS[dataset_id]
        self.agent_class = DDPG(
            obs_size = self.num_classes,
            act_size = self.num_classes,
            experiment_type = self.experiment_type,
            optimize_metric = self.optimize_metric,
            actor_path = agent_config_file.DDPG_ACTOR_SAVE_PATH, 
            critic_path = agent_config_file.DDPG_CRITIC_SAVE_PATH,
            dataset_name = dataset_name,
            total_server_rounds = self.total_server_rounds,
            seed = self.seed
        )
        self.loss_estimation_waiting_period = client_config_file.LOSS_ESTIMATION_WAITING_PERIOD
        self.action_buffer = list()
        try:
            temp_df = pd.read_csv(
                agent_config_file.AGENT_TRAINING_DATA_FILE
            )
            temp_df = temp_df[(temp_df['experiment_type']==experiment_type)&(temp_df['dataset_name']==dataset_name)&(temp_df['seed']==seed)]
            for row in temp_df[['server_round','action', 'reward', f'{self.optimize_metric}_on_local_dataset_after_aggregation']].iterrows():
                self.action_buffer.append(
                    {
                        'server_round': row[1]['server_round'],
                        'reward': [eval(row[1]['reward']).item()], 
                        'action': json.loads(row[1]['action']),
                        'state': json.loads(row[1][f'{self.optimize_metric}_on_local_dataset_after_aggregation'])
                    }
                )

            del temp_df 
        except:
            self.action_buffer = list()

    def set_optimizer(self, round, task="local_train"):
        learning_rate = self.optimizer_config["learning_rate"]*self.optimizer_config["learning_rate_decay"] \
            if round%self.optimizer_config["learning_rate_decay_period"]==0 else self.optimizer_config["learning_rate"]
        weight_decay = self.optimizer_config["weight_decay"]
        if task=="local_train":
            print(f"Setting optimizer for {task} | Learning Rate: {learning_rate} | Weight Decay: {weight_decay}")
            self.optimizer = torch.optim.SGD(
                params = self.model.parameters(),
                lr = learning_rate,
                weight_decay = weight_decay
            )

    def set_parameters(self, parameters):
        """
        Set the local parameters with parameters received from the server
        Create 2 copies of the smart client model
        Model 1. one which is trained using the action taken by the agent and shares its parameters with the server
        Model 2. two which is a copy of model 1, but it is also finetuned on the remaining data every time the aggregated parameters are received from the server. The parameters of
        this finetuned model are never shared back with the server or else this would share more information with the server than necessary (maybe).
        BUT
        if you are only fine tuning the model then that really should not be sharing too much information.
        """
        # set model parameters
        param_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in param_dict})
        self.model.load_state_dict(state_dict, strict=True)
        
        # turn on gradients for all parameters for the local update
        total_parameters = len([i for i in self.model.parameters()])
        for _, param in zip(range(total_parameters), self.model.parameters()):
            param.requires_grad=True

    def get_parameters(self, config):
        """
        Get paramaters from the local client after local update
        """
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
        
    def save_model(self):
        """
        Save model at the end of federated learning to fine tune locally
        """
        filename = client_config_file.get_client_checkpoint_path(f"server_copy_seed_{self.seed}")
        torch.save(
            self.model.state_dict(), 
            filename
        )


    def fit(self, parameters, config):
        """
        Method used by Flower during federated learning. This method performs the local/client train step on local data.
        parameters: default argument expected by Flower
        config: default argument expected by Flower. Contain dictionary of configurations for training.
        """
        #config utilized by flower during federation
        self.server_round = config["server_round"]
        self.epochs = config["local_epochs"]
        self.strategy = config["strategy"]
        print(f"Server round: {self.server_round}")

        self.set_parameters(parameters) # set local parameters with parameters received from the server

        #final fine tune at the end of the total federated scheme
        if self.server_round==client_config_file.TOTAL_SERVER_ROUNDS[self.dataset_id]:
            self.save_model() # fine tune the model copy on the local dataset

        # set optimizer
        self.set_optimizer(
            round=self.server_round,
            task="local_train"
        )
        self.f1_scores_on_validation_dataset_after_aggregation, \
        self.precision_on_validation_dataset_after_aggregation, \
        self.recall_on_validation_dataset_after_aggregation, \
        self.loss_on_validation_dataset_after_aggregation, \
        self.accuracy_on_validation_dataset_after_aggregation = client_test_metrics(
            model = self.model,
            data_loader = self.val_dataloader, 
            feature_name = self.feature_name,
            label_name = self.label_name,
            loss_fn = self.loss_fn,
            device = self.device
        )

        # current state
        self.f1_scores_on_local_dataset_after_aggregation, \
        self.precision_on_local_dataset_after_aggregation, \
        self.recall_on_local_dataset_after_aggregation, \
        self.loss_on_local_dataset_after_aggregation, \
        self.accuracy_on_local_dataset_after_aggregation = client_test_metrics(
            model = self.model,
            data_loader = self.train_dataloader, 
            feature_name = self.feature_name,
            label_name = self.label_name,
            loss_fn = self.loss_fn,
            device = self.device
        )

        if self.meta_action_type!="None": #for non ablation experiments
            #class counts from the train dataset
            class_counts = get_class_counts_from_dataloader(
                self.train_dataloader,
                feature_name=self.feature_name,
                label_name=self.label_name
            )

            # take action
            agent_action = AgentAction(
                n=self.server_round,
                action_buffer=self.action_buffer,
                agent_model=self.agent_class.get_actor(),
                current_state=self.f1_scores_on_local_dataset_after_aggregation, 
                class_counts=class_counts, 
                lbound=0.1, 
                ubound=0.8,
                optimize_metric=self.optimize_metric, #only required if taking the weighted metric action
                lookback_period_for_weighted_metric=3, #only required if taking the weighted metric action
                meta_action_type=self.meta_action_type,
            )

            self.action, self.action_type = agent_action.action()

            self.action_buffer.append(self.action) #add current action to the action buffer
            self.action_partitioned_dataloader, self.remaining_partitioned_dataloader = get_action_partitioned_data(
                dataloader = self.train_dataloader,
                action = self.action,
                feature_name = self.feature_name,
                label_name = self.label_name,
                batch_size = 64,
                shuffle = True
            )

            #local model training on the action partitioned dataset
            if torch.cuda.is_available(): 
                client_train_step( #client_parallel_train_step
                    model = self.model, 
                    data_loader = self.action_partitioned_dataloader,
                    feature_name = self.feature_name,
                    label_name = self.label_name,
                    loss_fn = self.loss_fn,
                    optimizer = self.optimizer,
                    epochs = client_config_file.LOCAL_TRAINING_EPOCHS,
                    device = self.device,
                    verbose = self.verbose
                )
            else:
                client_train_step(
                    model = self.model, 
                    data_loader = self.action_partitioned_dataloader,
                    feature_name = self.feature_name,
                    label_name = self.label_name,
                    loss_fn = self.loss_fn,
                    optimizer = self.optimizer,
                    epochs = client_config_file.LOCAL_TRAINING_EPOCHS,
                    device = self.device,
                    verbose = self.verbose
                )
        
        if self.meta_action_type=="None": #for ablation experiment - we use all data
            self.action = get_naive_action(
                train_dataloader=self.train_dataloader,
                val_dataloader=self.val_dataloader,
                label_name=self.label_name
            )
            self.action_type = "naive"
            self.action_buffer.append(self.action)
            if torch.cuda.is_available():
                client_train_step( #client_parallel_train_step
                    model = self.model, 
                    data_loader = self.train_dataloader,
                    feature_name = self.feature_name,
                    label_name = self.label_name,
                    loss_fn = self.loss_fn,
                    optimizer = self.optimizer,
                    epochs = client_config_file.LOCAL_TRAINING_EPOCHS,
                    device = self.device,
                    verbose = self.verbose
                )
            else:
                client_train_step(
                    model = self.model, 
                    data_loader = self.train_dataloader,
                    feature_name = self.feature_name,
                    label_name = self.label_name,
                    loss_fn = self.loss_fn,
                    optimizer = self.optimizer,
                    epochs = client_config_file.LOCAL_TRAINING_EPOCHS,
                    device = self.device,
                    verbose = self.verbose
                )

        #compute loss after local training on local dataset and validation dataset before next round's aggregation
        self.f1_scores_on_local_dataset_after_local_training, \
        self.precision_on_local_dataset_after_local_training, \
        self.recall_on_local_dataset_after_local_training, \
        self.loss_on_local_dataset_after_local_training, \
        self.accuracy_on_local_dataset_after_local_training = client_test_metrics(
            model = self.model,
            data_loader = self.action_partitioned_dataloader if self.meta_action_type!="None" else self.train_dataloader,
            feature_name = self.feature_name,
            label_name = self.label_name,
            loss_fn = self.loss_fn,
            device = self.device
        )
        self.f1_scores_on_validation_dataset_after_local_training, \
        self.precision_on_validation_dataset_after_local_training, \
        self.recall_on_validation_dataset_after_local_training, \
        self.loss_on_validation_dataset_after_local_training, \
        self.accuracy_on_validation_dataset_after_local_training = client_test_metrics(
            model = self.model,
            data_loader = self.val_dataloader,
            feature_name = self.feature_name,
            label_name = self.label_name,
            loss_fn = self.loss_fn,
            device = self.device
        )

        # do loss estimation and get reward
        if self.loss_estimation_waiting_period<=self.server_round and self.meta_action_type!="None": #this line will evaluate to true when loss_estimation_waiting_period==0
            try:
                print("Updating Loss Estimation") 
                training_losses = pd.read_csv(client_config_file.OTP_METRICS_FILE)['loss_on_local_dataset_after_local_training'].to_numpy()
                communication_rounds_completed = np.arange(len(training_losses))
                popt, pcov = curve_fit(
                    exp_loss,
                    communication_rounds_completed,
                    training_losses,
                    bounds=(-3,3)
                )

                #estimate loss for the next round
                self.estimated_loss = exp_loss(
                    self.server_round+1,
                    *popt
                ) 

                self.reward = get_reward(
                    torch.tensor(self.action),
                    torch.tensor(self.estimated_loss), #estimated loss
                    torch.tensor(self.loss_on_local_dataset_after_aggregation),
                    epsilon=0.1
                ).cpu()  
            
            except RuntimeError as e:
                print(f"Runtime error in loss estimation: {e}")
                self.reward = get_reward(
                    torch.tensor(self.action),
                    torch.tensor(self.loss_on_local_dataset_after_local_training),
                    torch.tensor(self.loss_on_local_dataset_after_aggregation),
                    epsilon=0.1
                ).cpu()
        else:
            self.reward = get_reward(
                torch.tensor(self.action),
                torch.tensor(self.loss_on_local_dataset_after_local_training),
                torch.tensor(self.loss_on_local_dataset_after_aggregation),
                epsilon=0.1
            ).cpu()

        self.save_agent_data()
        self.save_otp_metrics()

        if self.server_round%agent_config_file.EPISODE_LENGTH==0 and self.meta_action_type!="None": #train agent at the end of each pre-defined episode
            self.agent_class.train_ddpg(self.server_round)

        client_dict = {
            'local_loss':self.loss_on_validation_dataset_after_local_training
        }

        if self.meta_action_type=="None":
            return self.get_parameters(config=None), len(self.train_dataloader), client_dict
        if self.meta_action_type!="None":
            return self.get_parameters(config=None), len(self.action_partitioned_dataloader), client_dict
    
    def save_otp_metrics(self):
        """
        Save metrics before and after action partitiioned data local model training
        before_aggregation: always refers to post local training after previous round's aggregation
        """
        if os.path.isfile(client_config_file.OTP_METRICS_FILE):
            print(f"Writing data to {client_config_file.OTP_METRICS_FILE}")
            with open(client_config_file.OTP_METRICS_FILE, 'a') as file:
                writer = csv.writer(file)
                writer.writerow(
                    [
                        self.server_round,
                        self.estimated_loss,
                        self.strategy,
                        self.experiment_type,
                        self.dataset_name,
                        json.dumps(self.f1_scores_on_validation_dataset_after_aggregation.to_dict()),
                        json.dumps(self.precision_on_validation_dataset_after_aggregation.to_dict()),
                        json.dumps(self.recall_on_validation_dataset_after_aggregation.to_dict()),
                        self.loss_on_validation_dataset_after_aggregation,
                        self.accuracy_on_validation_dataset_after_aggregation,
                        json.dumps(self.f1_scores_on_local_dataset_after_aggregation.to_dict()),
                        json.dumps(self.precision_on_local_dataset_after_aggregation.to_dict()),
                        json.dumps(self.recall_on_local_dataset_after_aggregation.to_dict()),
                        self.loss_on_local_dataset_after_aggregation,
                        self.accuracy_on_local_dataset_after_aggregation,
                        json.dumps(self.f1_scores_on_local_dataset_after_local_training.to_dict()),
                        json.dumps(self.precision_on_local_dataset_after_local_training.to_dict()),
                        json.dumps(self.recall_on_local_dataset_after_local_training.to_dict()),
                        self.loss_on_local_dataset_after_local_training,
                        self.accuracy_on_local_dataset_after_local_training,
                        json.dumps(self.f1_scores_on_validation_dataset_after_local_training.to_dict()),
                        json.dumps(self.precision_on_validation_dataset_after_local_training.to_dict()),
                        json.dumps(self.recall_on_validation_dataset_after_local_training.to_dict()),
                        self.loss_on_validation_dataset_after_local_training,
                        self.accuracy_on_validation_dataset_after_local_training,
                        self.seed
                    ]
                )
        else:
            with open(client_config_file.OTP_METRICS_FILE, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(
                    [
                        'server_round',
                        'estimated_loss',
                        'strategy',
                        'experiment_type',
                        'dataset_name',
                        'f1_scores_on_validation_dataset_after_aggregation',
                        'precision_on_validation_dataset_after_aggregation',
                        'recall_on_validation_dataset_after_aggregation',
                        'loss_on_validation_dataset_after_aggregation',
                        'accuracy_on_validation_dataset_after_aggregation',
                        'f1_scores_on_local_dataset_after_aggregation',
                        'precision_on_local_dataset_after_aggregation',
                        'recall_on_local_dataset_after_aggregation',
                        'loss_on_local_dataset_after_aggregation',
                        'accuracy_on_local_dataset_after_aggregation',
                        'f1_scores_on_local_dataset_after_local_training',
                        'precision_on_local_dataset_after_local_training',
                        'recall_on_local_dataset_after_local_training',
                        'loss_on_local_dataset_after_local_training',
                        'accuracy_on_local_dataset_after_local_training',
                        'f1_scores_on_validation_dataset_after_local_training',
                        'precision_on_validation_dataset_after_local_training',
                        'recall_on_validation_dataset_after_local_training',
                        'loss_on_validation_dataset_after_local_training',
                        'accuracy_on_validation_dataset_after_local_training',
                        'seed'
                    ]
                )
                writer.writerow(
                    [
                        self.server_round,
                        self.estimated_loss,
                        self.strategy,
                        self.experiment_type,
                        self.dataset_name,
                        json.dumps(self.f1_scores_on_validation_dataset_after_aggregation.to_dict()),
                        json.dumps(self.precision_on_validation_dataset_after_aggregation.to_dict()),
                        json.dumps(self.recall_on_validation_dataset_after_aggregation.to_dict()),
                        self.loss_on_validation_dataset_after_aggregation,
                        self.accuracy_on_validation_dataset_after_aggregation,
                        json.dumps(self.f1_scores_on_local_dataset_after_aggregation.to_dict()),
                        json.dumps(self.precision_on_local_dataset_after_aggregation.to_dict()),
                        json.dumps(self.recall_on_local_dataset_after_aggregation.to_dict()),
                        self.loss_on_local_dataset_after_aggregation,
                        self.accuracy_on_local_dataset_after_aggregation,
                        json.dumps(self.f1_scores_on_local_dataset_after_local_training.to_dict()),
                        json.dumps(self.precision_on_local_dataset_after_local_training.to_dict()),
                        json.dumps(self.recall_on_local_dataset_after_local_training.to_dict()),
                        self.loss_on_local_dataset_after_local_training,
                        self.accuracy_on_local_dataset_after_local_training,
                        json.dumps(self.f1_scores_on_validation_dataset_after_local_training.to_dict()),
                        json.dumps(self.precision_on_validation_dataset_after_local_training.to_dict()),
                        json.dumps(self.recall_on_validation_dataset_after_local_training.to_dict()),
                        self.loss_on_validation_dataset_after_local_training,
                        self.accuracy_on_validation_dataset_after_local_training,
                        self.seed
                    ]
                )
            print(f"New metrics file created: {client_config_file.OTP_METRICS_FILE}")
    
    def save_agent_data(self):
        """
        Save training data for agent
        """
        print(os.getcwd())
        if os.path.isfile(agent_config_file.AGENT_TRAINING_DATA_FILE):
            print("Writing data to agent_data.csv")
            with open(agent_config_file.AGENT_TRAINING_DATA_FILE, 'a') as file:
                writer = csv.writer(file)
                writer.writerow(
                    [
                        self.server_round,
                        self.action_type,
                        self.experiment_type,
                        self.dataset_name,
                        json.dumps(self.action.tolist()),
                        self.reward,
                        json.dumps(self.f1_scores_on_local_dataset_after_aggregation.to_dict()),
                        json.dumps(self.precision_on_local_dataset_after_aggregation.to_dict()),
                        json.dumps(self.recall_on_local_dataset_after_aggregation.to_dict()),
                        self.seed
                    ]
                )
        else:
            with open(agent_config_file.AGENT_TRAINING_DATA_FILE, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(
                    [
                        'server_round',
                        'action_type',
                        'experiment_type',
                        'dataset_name',
                        'action',
                        'reward',
                        'f1_scores_on_local_dataset_after_aggregation',
                        'precision_on_local_dataset_after_aggregation',
                        'recall_on_local_dataset_after_aggregation',
                        'seed'
                    ]
                )
                writer.writerow(
                    [
                        self.server_round,
                        self.action_type,
                        self.experiment_type,
                        self.dataset_name,
                        json.dumps(self.action.tolist()),
                        self.reward,
                        json.dumps(self.f1_scores_on_local_dataset_after_aggregation.to_dict()),
                        json.dumps(self.precision_on_local_dataset_after_aggregation.to_dict()),
                        json.dumps(self.recall_on_local_dataset_after_aggregation.to_dict()),
                        self.seed
                    ]
                )
            print(f"New agent training data file created: {agent_config_file.AGENT_TRAINING_DATA_FILE}")


#set up a flower client
class OTPFineTuneClient:
    def __init__(
            self, client_id, model, train_dataloader, val_dataloader, \
            public_partition_dataloader, neutral_dataloader, dataset_name, feature_name, \
            label_name, loss_fn, optimizer_config, num_classes, \
            experiment_type, dataset_id, strategy, server_round, seed, device, verbose
        ):
        self.client_id = client_id
        self.model = model
        self.model_clone = model #create a copy of the server cloned parameters for local fine-tuning
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.public_partition_dataloader = public_partition_dataloader
        self.neutral_dataloader = neutral_dataloader
        self.dataset_name = dataset_name
        self.feature_name = feature_name
        self.label_name = label_name
        self.loss_fn = loss_fn
        self.optimizer_config = optimizer_config
        self.num_classes = num_classes
        self.seed = seed
        self.device = device
        self.verbose = verbose
        self.experiment_type = experiment_type
        self.dataset_id = dataset_id
        self.strategy = strategy
        self.server_round = server_round
        

    def set_parameters_to_fine_tune(self, parameters):
        """
        Take the aggregated parameters for the model and load them into the model copy.
        The model copy paramters are fine tuned on the public partition after data aggregation and never shared with the server.
        Hypothesis:
        - At the end of the all training rounds the model copy will be at least as good as the naive models.
        """
        param_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in param_dict})
        self.model.load_state_dict(state_dict, strict=True)

        # turn off gradients for all parameters except for the last one to fine tune the model on the remaining data
        total_parameters = len([i for i in self.model.parameters()])
        # for _, param in zip(range(total_parameters - 1), self.model_clone.parameters()):
        #     param.requires_grad = False
        for _, param in zip(range(total_parameters), self.model.parameters()):
            param.requires_grad = True

    def set_optimizer(self, round, task="local_train"):
        learning_rate = self.optimizer_config["learning_rate"]*self.optimizer_config["learning_rate_decay"] \
            if round%self.optimizer_config["learning_rate_decay_period"]==0 else self.optimizer_config["learning_rate"]
        weight_decay = self.optimizer_config["weight_decay"]
        if task=="fine_tune":
            print(f"Setting optimizer for {task} | Learning Rate: {learning_rate} | Weight Decay: {weight_decay}")
            self.optimizer = torch.optim.SGD(
                params = self.model.parameters(),
                lr = learning_rate,
                weight_decay = weight_decay
            )

    def fine_tune_model(self, parameters):
        """
        Fine tune the model parameters received from the server after aggregation in the previous round.
        There will be a need to maintain a separate copy of the local client which is not finetuned on the remaining data
        or in the next round the parameters finetuned on the remaining data will be trained on the local training data and shared back with the server
        """
        print(f"Finetuning OTP Client..")
        self.set_parameters_to_fine_tune(parameters)
        for epoch in range(client_config_file.FINE_TUNE_EPOCHS[self.dataset_id]):
            self.fine_tune_epoch = epoch
            self.set_optimizer(
                round=epoch,
                task="fine_tune"
            )
            if torch.cuda.is_available() and torch.cuda.device_count()>1: 
            #     client_parallel_train_step(
            #         model = self.model, 
            #         data_loader = self.train_dataloader,
            #         feature_name = self.feature_name,
            #         label_name = self.label_name,
            #         loss_fn = self.loss_fn,
            #         optimizer = self.optimizer,
            #         epochs = client_config_file.LOCAL_TRAINING_EPOCHS,
            #         device = self.device,
            #         verbose = self.verbose
            #     )
            # else:
                client_train_step(
                    model = self.model, 
                    data_loader = self.train_dataloader,
                    feature_name = self.feature_name,
                    label_name = self.label_name,
                    loss_fn = self.loss_fn,
                    optimizer = self.optimizer,
                    epochs = client_config_file.LOCAL_TRAINING_EPOCHS,
                    device = self.device,
                    verbose = self.verbose
                )

            #measure performance on the local dataset after the finetuning step
            self.f1_scores_on_local_dataset_after_aggregation_and_finetuning, \
            self.precision_on_local_dataset_after_aggregation_and_finetuning, \
            self.recall_on_local_dataset_after_aggregation_and_finetuning, \
            self.loss_on_local_dataset_after_aggregation_and_finetuning, \
            self.accuracy_on_local_dataset_after_aggregation_and_finetuning = client_test_metrics(
                model = self.model,
                data_loader = self.train_dataloader,
                feature_name = self.feature_name,
                label_name = self.label_name,
                loss_fn = self.loss_fn,
                device = self.device
            )

            #measure performance on the validation dataset after the finetuning step
            self.f1_scores_on_validation_dataset_after_aggregation_and_finetuning, \
            self.precision_on_validation_dataset_after_aggregation_and_finetuning, \
            self.recall_on_validation_dataset_after_aggregation_and_finetuning, \
            self.loss_on_validation_dataset_after_aggregation_and_finetuning, \
            self.accuracy_on_validation_dataset_after_aggregation_and_finetuning = client_test_metrics(
                model = self.model,
                data_loader = self.val_dataloader,
                feature_name = self.feature_name,
                label_name = self.label_name,
                loss_fn = self.loss_fn,
                device = self.device
            )

            ##save metrics on local and validation data
            self.save_otp_copy_metrics()

    
    def save_otp_copy_metrics(self):
        """Save metrics from fine tuning"""
        if os.path.isfile(client_config_file.OTP_CLONE_METRICS_FILE):
            print(f"Writing data to {client_config_file.OTP_CLONE_METRICS_FILE}")
            with open(client_config_file.OTP_CLONE_METRICS_FILE, 'a') as file:
                writer = csv.writer(file)
                writer.writerow(
                    [
                        self.server_round,
                        self.strategy,
                        self.experiment_type,
                        self.fine_tune_epoch,
                        self.dataset_name,
                        json.dumps(self.f1_scores_on_local_dataset_after_aggregation_and_finetuning.to_dict()),
                        json.dumps(self.precision_on_local_dataset_after_aggregation_and_finetuning.to_dict()),
                        json.dumps(self.recall_on_local_dataset_after_aggregation_and_finetuning.to_dict()),
                        self.loss_on_local_dataset_after_aggregation_and_finetuning,
                        self.accuracy_on_local_dataset_after_aggregation_and_finetuning,
                        json.dumps(self.f1_scores_on_validation_dataset_after_aggregation_and_finetuning.to_dict()),
                        json.dumps(self.precision_on_validation_dataset_after_aggregation_and_finetuning.to_dict()),
                        json.dumps(self.recall_on_validation_dataset_after_aggregation_and_finetuning.to_dict()),
                        self.loss_on_validation_dataset_after_aggregation_and_finetuning,
                        self.accuracy_on_validation_dataset_after_aggregation_and_finetuning,
                        self.seed
                    ]
                )
        else:
            with open(client_config_file.OTP_CLONE_METRICS_FILE, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(
                    [
                    'server_round',
                    'strategy',
                    'experiment_type',
                    'fine_tune_epoch',
                    'dataset_name',
                    'f1_scores_on_local_dataset_after_aggregation_and_finetuning',
                    'precision_on_local_dataset_after_aggregation_and_finetuning',
                    'recall_on_local_dataset_after_aggregation_and_finetuning',
                    'loss_on_local_dataset_after_aggregation_and_finetuning',
                    'accuracy_on_local_dataset_after_aggregation_and_finetuning',
                    'f1_scores_on_validation_dataset_after_aggregation_and_finetuning',
                    'precision_on_validation_dataset_after_aggregation_and_finetuning',
                    'recall_on_validation_dataset_after_aggregation_and_finetuning',
                    'loss_on_validation_dataset_after_aggregation_and_finetuning',
                    'accuracy_on_validation_dataset_after_aggregation_and_finetuning',
                    'seed'
                    ]
                )
                writer.writerow(
                    [
                        self.server_round,
                        self.strategy,
                        self.experiment_type,
                        self.fine_tune_epoch,
                        self.dataset_name,
                        json.dumps(self.f1_scores_on_local_dataset_after_aggregation_and_finetuning.to_dict()),
                        json.dumps(self.precision_on_local_dataset_after_aggregation_and_finetuning.to_dict()),
                        json.dumps(self.recall_on_local_dataset_after_aggregation_and_finetuning.to_dict()),
                        self.loss_on_local_dataset_after_aggregation_and_finetuning,
                        self.accuracy_on_local_dataset_after_aggregation_and_finetuning,
                        json.dumps(self.f1_scores_on_validation_dataset_after_aggregation_and_finetuning.to_dict()),
                        json.dumps(self.precision_on_validation_dataset_after_aggregation_and_finetuning.to_dict()),
                        json.dumps(self.recall_on_validation_dataset_after_aggregation_and_finetuning.to_dict()),
                        self.loss_on_validation_dataset_after_aggregation_and_finetuning,
                        self.accuracy_on_validation_dataset_after_aggregation_and_finetuning,
                        self.seed
                    ]
                )
            print(f"New metrics file created: {client_config_file.OTP_CLONE_METRICS_FILE}")