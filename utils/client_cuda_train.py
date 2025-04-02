import os
import socket
import torch
import numpy as np
import torch.nn.functional as F

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from config_folder import client_config_file
from utils.client_train_test_utils import accuracy_fn

def get_unused_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        # Bind the socket to an ephemeral port
        s.bind(('', 0))
        # Get the port number assigned by the OS
        unused_port = s.getsockname()[1]
        while unused_port==22:
            unused_port = s.getsockname()[1]

    print(f"Port used for parallel training: {unused_port}")
    return unused_port

class CudaTrainParallel:
    def __init__(self, gpu_id, model, train_dataloader, feature_name, label_name, loss_fn, optimizer, epochs, device, verbose=True):
        self.gpu_id = gpu_id
        self.model = model.to(gpu_id)
        self.model = DDP(model, device_ids=[gpu_id])
        self.train_dataloader = train_dataloader
        self.feature_name = feature_name
        self.label_name = label_name
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device
        self.verbose= verbose
        self.train_loss = 0
        self.train_acc = 0
    
    def run_batch(self, X, y, batch_id):
        self.optimizer.zero_grad()
        y_logits = self.model(X)
        loss = self.loss_fn(
            y_logits, 
            y
        )
        if len(y_logits)>1: #if batch has more than 1 sample no need to add another dimension
            y_pred = F.softmax(
                y_logits.squeeze(), 
                dim=0).argmax(dim=1) #use softmax for multi class classification and argmax gives the index of the predicted class
        elif len(y_logits)==1: #if batch size is 1 then add 1 more dimension
            y_pred = F.softmax(
                y_logits.squeeze(), 
                dim=0).unsqueeze(dim=0).argmax(dim=1)
        acc = accuracy_fn(
            y_true=y,
            y_pred=y_pred
        )
        self.train_acc += acc #accumulate accuracy per batch
        self.train_loss += loss
        loss.backward()
        self.optimizer.step()
        if self.verbose:
            if batch_id%50==0:
                print(f"Local Training Batch: {batch_id} | Train Loss: {loss.item():.3f} | Train Acc: {acc:.3f}")
    
    def run_epoch(self, epoch):
        self.train_dataloader.sampler.set_epoch(epoch)
        for i, batch in enumerate(self.train_dataloader):
            X, y = batch[self.feature_name].to(self.gpu_id), batch[self.label_name].to(self.gpu_id)
            self.run_batch(X, y, batch_id=i)

    def train(self, epochs):
        for epoch in range(epochs):
            self.run_epoch(epoch)

# Set up the process group for distributed training
def setup(rank, world_size):
    """
    Args:
    - rank: unique identifier of each process
    - world_size: total number of processes
    """
    os.environ['MASTER_ADDR'] = 'localhost' #address of the machine running the rank process
    os.environ['MASTER_PORT'] = str(get_unused_port()) #any free port
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

# Clean up the process group
def cleanup():
    dist.destroy_process_group()

def get_distributed_dataloader(train_dataloader):
    return DataLoader(
        dataset = train_dataloader.dataset,
        batch_size=client_config_file.LOCAL_TRAINING_BATCH_SIZE,
        pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler(train_dataloader.dataset)
    )

def train_main(
    rank: int,
    world_size: int,
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    feature_name: str,
    label_name: str,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epochs: int=1,
    device: torch.device="cpu",
    verbose: bool=True,
    ):
    """
    Perform model training using torch DataLoader.
    """
    setup(rank, world_size)
    data_loader = get_distributed_dataloader(data_loader)
    trainer = CudaTrainParallel(
        gpu_id=rank,
        model=model,
        train_dataloader=data_loader,
        feature_name=feature_name,
        label_name=label_name,
        loss_fn=loss_fn,
        optimizer=optimizer,
        epochs=epochs,
        device=device,
        verbose=verbose
    )
    trainer.train(
        epochs=epochs
    )
    cleanup()

def client_parallel_train_step(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    feature_name: str,
    label_name: str,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epochs: int=1,
    device: torch.device="cpu",
    verbose: bool=True, 
    ):
    if torch.cuda.is_available():
        world_size = torch.cuda.device_count() if torch.cuda.is_available() else None
        mp.spawn(train_main, args=(world_size, model, data_loader, feature_name, label_name, loss_fn, optimizer, epochs, device, verbose), nprocs=world_size, join=True)