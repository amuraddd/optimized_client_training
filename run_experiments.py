import os
import gc
import torch
import contextlib
import pandas as pd
from torch import nn
from server import run_server
import torch.multiprocessing as mp
from dataclasses import dataclass
from utils.utils import set_torch_device, get_parameters
from data.data_prepper import load_datasets
from src.clients.otp_client import OTPFineTuneClient
from src.models.server_models import initialize_model
from config_folder import server_config_file, client_config_file

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) #deprecation warning for cid replaced by flwr.common.Context
warnings.filterwarnings("ignore", category=UserWarning)


@contextlib.contextmanager
def clear_memory():
    try:
        yield
    finally:
        gc.collect

#fine tuning model
@dataclass
class TempContext:
    node_id: int
    node_config: dict
    state: dict
    run_config: dict

    def __init__(  # pylint: disable=too-many-arguments
        self,
        node_id: int,
        node_config: dict
    ) -> None:
        self.node_id = node_id
        self.node_config = node_config

def execute_strategy(strategies=server_config_file.STRATEGIES):
    """
    Execute a single strategy with all action types
    """
    all_jobs_status = list()
    for seed in [48]: #40, 42, 48
        for dataset_id in server_config_file.DATASET_IDS:
            print(f"Running Experiments for: {client_config_file.DATASETS[dataset_id]['name']}")
            for strategy_name in strategies:
                for meta_action_type in server_config_file.ACTION_TYPES:
                    job_status = run_server(
                        num_cpus=10,
                        num_gpus=1 if torch.cuda.is_available() else 0,
                        dataset_id=dataset_id,
                        optimize_metric='f1_scores',
                        strategy_name=strategy_name,
                        meta_action_type=meta_action_type,
                        seed=seed
                    )
                    dataset_name = client_config_file.DATASETS[dataset_id]['name']
                    num_classes = client_config_file.DATASETS[dataset_id]['num_classes']
                    feature_name = client_config_file.DATASETS[dataset_id]['feature_name']
                    label_name = client_config_file.DATASETS[dataset_id]['label_name']
                    server_rounds = client_config_file.DATASETS[dataset_id]['training_periods']
                    optimizer_config = client_config_file.DATASETS[dataset_id]['optimizer_config']
                    input_size = client_config_file.DATASETS[dataset_id]['input_shape']

                    
                    temp_context = TempContext(
                        node_id=0,
                        node_config={
                            "partition-id":0,
                            "num-partitions":client_config_file.NUM_CLIENTS
                        }
                    )
                    
                    # load local train dataloader partitions and the neutral dataset
                    train_dataloader, val_dataloader, neutral_dataloader, fds_partitioners = load_datasets(
                        partition_id = temp_context.node_id,
                        num_partitions = temp_context.node_config["num-partitions"],
                        dataset_name = dataset_name,
                        feature_name = feature_name,
                        label_name = label_name,
                        input_size = input_size,
                        batch_size = client_config_file.LOCAL_TRAINING_BATCH_SIZE,
                        seed=seed
                    )

                    del fds_partitioners
                    
                    DEVICE = set_torch_device(
                        strategy=strategy_name,
                        manual_seed=seed
                    )
                    print(f"Device used for Local Finetuning: {DEVICE}")
                    filename = client_config_file.get_server_checkpoint_path(f"{dataset_name}_{strategy_name}_{meta_action_type}")#get_client_checkpoint_path("server_copy")

                    print(f"Server Copy File Name: {filename}")

                    #initialize the model and then load the saved model's state dict in it
                    model, input_size = initialize_model(
                        model_name="resnet",
                        num_classes=num_classes,
                        feature_extract=False,
                        use_pretrained=None
                    )
                    client_loss_fn = nn.CrossEntropyLoss()

                    if os.path.isfile(filename):
                        print(f"Loading server copy")
                        model.load_state_dict(
                            torch.load(
                                filename,
                                map_location = torch.device(DEVICE)
                            )
                        )
                    else:
                        print(f"No server copy found")

                    #add a fully connected layer with the right number of classes as in the train dataset
                    # num_classes = len(pd.Series(
                    #         [i[label_name] for i in train_dataloader.dataset.to_list()]+\
                    #             [i[label_name] for i in val_dataloader.dataset.to_list()]
                    #     ).unique())
                    num_classes = pd.Series(
                            [i[label_name] for i in train_dataloader.dataset.to_list()]+\
                                [i[label_name] for i in val_dataloader.dataset.to_list()]
                        ).unique().max()+1 #class labels have to be the largest - if you train on less classes then the model will throw an error
                    
                    print(f"Fine tuning classes {num_classes}")
                    num_ftrs = model.fc.in_features
                    model.fc = nn.Linear(num_ftrs, num_classes)
                    # print(model)
                    otp_client = OTPFineTuneClient(
                        client_id=temp_context.node_id,
                        model=model,
                        train_dataloader = train_dataloader, 
                        val_dataloader = val_dataloader,
                        public_partition_dataloader = neutral_dataloader,
                        neutral_dataloader = neutral_dataloader,
                        dataset_name = dataset_name, 
                        feature_name = feature_name, 
                        label_name = label_name, 
                        loss_fn = client_loss_fn, 
                        optimizer_config = optimizer_config, 
                        num_classes = num_classes,
                        seed = seed,
                        device = DEVICE, 
                        verbose = True,
                        experiment_type = f"{strategy_name}_{meta_action_type}",
                        dataset_id = dataset_id,
                        server_round = server_rounds,
                        strategy=strategy_name
                    )
                    
                    parameters = get_parameters(model)
                    otp_client.fine_tune_model(
                        parameters=parameters
                    )

                    all_jobs_status.append(job_status)

    return all_jobs_status

# 'FedMedian', 'FedProx', 'FedCDA']

with clear_memory():
    if __name__=='__main__':
        all_jobs_status = execute_strategy()
        for status in all_jobs_status:
            print(status)