import os
import csv
import json
import torch
import numpy as np
import flwr as fl
import pandas as pd
from collections import OrderedDict
from config_folder import client_config_file
from utils.agent_action_reward import get_reward
from utils.client_cuda_train import client_parallel_train_step
from utils.client_train_test_utils import client_train_step, client_test_step, client_test_metrics

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

#set up a flower client
class NTPClient(fl.client.NumPyClient):
    def __init__(
        self, client_id, model, train_dataloader, val_dataloader, \
        public_partition_dataloader, neutral_dataloader, dataset_name, feature_name, \
        label_name, loss_fn, optimizer_config, experiment_type, dataset_id, seed, device, verbose
    ):
        super().__init__()
        self.client_id = client_id
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.public_partition_dataloader = public_partition_dataloader
        self.neutral_dataloader = neutral_dataloader
        self.dataset_name = dataset_name
        self.feature_name = feature_name
        self.label_name = label_name
        self.loss_fn = loss_fn
        self.optimizer_config = optimizer_config
        self.experiment_type = experiment_type
        self.dataset_id = dataset_id
        self.seed = seed
        self.device = device
        self.verbose = verbose

    def set_parameters(self, parameters):
        """
        Set the local parameters with parameters received from the server
        """
        # set model parameters
        param_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in param_dict})
        self.model.load_state_dict(state_dict, strict=True)
        
        # turn on gradients for all parameters for the local update
        total_parameters = len([i for i in self.model.parameters()])
        for _, param in zip(range(total_parameters), self.model.parameters()):
            param.requires_grad=True
    
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
    
    def get_parameters(self, config):
        """
        Get paramaters from the local clients after local update
        """
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

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
        
        self.set_parameters(parameters)

        #set optimizer
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

        print(f"Dataloader Size: {len(self.train_dataloader)}")
        #local model training on the action partitioned dataset
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
            data_loader = self.train_dataloader,
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

        self.action = get_naive_action(
            train_dataloader=self.train_dataloader,
            val_dataloader=self.val_dataloader,
            label_name=self.label_name
        )

        self.reward = get_reward(
            torch.tensor(self.action),
            torch.tensor(self.loss_on_local_dataset_after_local_training),
            torch.tensor(self.loss_on_local_dataset_after_aggregation),
            epsilon=0.1
        ).cpu()

        #measure performance on the validation dataset after every aggregation step from the server
        self.save_metrics()
        
        #after local train step get the parameters from the locals
        client_dict = {
            'local_loss':self.loss_on_validation_dataset_after_local_training
        }

        return self.get_parameters(config=None), len(self.train_dataloader),  client_dict

    def save_metrics(self):
        """
        Save metrics before and after action partitiioned data local model training
        before_aggregation: always refers to post local training after previous round's aggregation
        """
        if os.path.isfile(client_config_file.NTP_METRICS_FILE):
            print(f"Writing data to {client_config_file.NTP_METRICS_FILE}")
            with open(client_config_file.NTP_METRICS_FILE, 'a') as file:
                writer = csv.writer(file)
                writer.writerow(
                    [
                        self.server_round,
                        self.client_id,
                        self.strategy,
                        self.action,
                        self.reward,
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
            with open(client_config_file.NTP_METRICS_FILE, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(
                    [
                        'server_round',
                        'client_id',
                        'strategy',
                        'action',
                        'reward',
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
                        self.client_id,
                        self.strategy,
                        self.action,
                        self.reward,
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
            print(f"New metrics file created: {client_config_file.NTP_METRICS_FILE}")

def get_naive_action(train_dataloader, val_dataloader, label_name):
    """
    Given train and val dataloader get naive action for the NTP client
    """
    y_train = [i[label_name] for i in train_dataloader.dataset.to_list()]
    y_val = [i[label_name] for i in val_dataloader.dataset.to_list()]
    y = y_train+y_val
    y_counts = pd.DataFrame(y, index=y).groupby(level=0).count()
    y_train_counts = pd.DataFrame(y_train, index=y_train).groupby(level=0).count()
    action = y_train_counts/y_counts
    action = action.dropna().values.flatten()
    return action