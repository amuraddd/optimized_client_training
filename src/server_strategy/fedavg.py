import torch
import flwr as fl
import numpy as np
from flwr.common import (
    FitRes,
    Parameters,
    Scalar
)
from collections import OrderedDict
from flwr.server.client_proxy import ClientProxy
from typing import List, Tuple, Dict, Union, Optional
from config_folder import client_config_file
from config_folder.client_config_file import get_server_checkpoint_path
from utils.utils import get_parameters

def fedavg_strategy(model, client_config, dataset_name, strategy_name, meta_action_type, seed=42):
    """
    Implement FedAvg Strategy
    """
    initial_parameters = fl.common.ndarrays_to_parameters(get_parameters(model)) #set intial parameters

    class SaveFedAvgModelStrategy(fl.server.strategy.FedAvg):
        def aggregate_fit(
            self,
            server_round: int,
            results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
            failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
            ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
            """Aggregate model weights using weighted average and store checkpoint"""
            # Call aggregate_fit from base class (FedAvg) to aggregate parameters and metrics
            aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)

            if aggregated_parameters is not None:
                print(f"Saving round {server_round} aggregated_parameters...")

                # Convert `Parameters` to `List[np.ndarray]`
                aggregated_ndarrays: List[np.ndarray] = fl.common.parameters_to_ndarrays(aggregated_parameters)

                # Convert `List[np.ndarray]` to PyTorch`state_dict`
                params_dict = zip(model.state_dict().keys(), aggregated_ndarrays)
                state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
                model.load_state_dict(state_dict, strict=True)

                # Save the model
                SERVER_CHECKPOINT_PATH = get_server_checkpoint_path(f"{dataset_name}_{strategy_name}_{meta_action_type}_seed_{seed}")
                torch.save(
                    model.state_dict(), 
                    SERVER_CHECKPOINT_PATH
                )

            return aggregated_parameters, aggregated_metrics

    strategy = SaveFedAvgModelStrategy(
        fraction_fit = 0.8,
        fraction_evaluate = 1.0,
        min_fit_clients = int(0.8*client_config_file.NUM_CLIENTS),
        min_evaluate_clients = client_config_file.NUM_CLIENTS,
        min_available_clients = client_config_file.NUM_CLIENTS,
        on_fit_config_fn = client_config,
        on_evaluate_config_fn= client_config,
        initial_parameters=initial_parameters
    )
    return strategy