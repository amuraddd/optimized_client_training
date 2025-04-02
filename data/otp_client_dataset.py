import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from config_folder import client_config_file

class SmartCleintDataset(Dataset):
    def __init__(self, samples, labels, feature_name, label_name):
        self.samples = samples
        self.labels = labels
        self.feature_name = feature_name
        self.label_name = label_name
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        x = self.samples[idx]
        y = self.labels[idx]
        sample = {self.feature_name: x, self.label_name: y}
        return sample

def get_action_partitioned_data(dataloader, action, feature_name='img', label_name='fine_label', batch_size=64, shuffle=True):
    """
    Input:
    - dataloader: train dataloader for the current training round
    - action: action taken based on the current classwise accuracy
    - feature_name: name of the features in the dataset (ie. name of the X variable)
    - label_name: name of the labels in the dataset (ie. name of the Y variable)
    - batch_size: batch size used to create the dataloaders
    - shuffle: boolean for if or not to shuffle the returned dataloaders
    Output:
    - action_partitioned_dataloader: based on the action partitioned dataset
    - remaining_partitioned_dataloader: complement of the action_partitioned_dataloader 
    """
    torch.manual_seed(42) #set the torch seed so the dataloader produces the same location of indices each time
    dataset = [i for i in dataloader]
    x_data_all_batches = list()
    y_data_all_batches = list()
    for i, batch in enumerate(dataset):
        temp_batch_size = batch[feature_name].shape[0] #batch size
        for img in range(temp_batch_size):
            x_data_all_batches.append(dataset[i][feature_name][img].cpu().numpy())
            y_data_all_batches.append(dataset[i][label_name][img].cpu().numpy())

    x_data_all_batches = np.array(x_data_all_batches)
    y_data_all_batches = np.array(y_data_all_batches)

    classes = np.unique(
        y_data_all_batches,
        return_counts=True
        )[0]
    class_counts = np.unique(
        y_data_all_batches,
        return_counts=True
        )[1]
    action_partitioned_counts = np.floor(class_counts*action)

    ## set the seed so that the data selected for local training in each round is the same as the round before - this ensures we don't overshare information with the server
    action_partitioned_indices = list()
    for c, ct in zip(classes, action_partitioned_counts):
        np.random.seed(42)
        temp_indices = np.where(y_data_all_batches==c)[0][:int(ct)]      
        action_partitioned_indices.append(
            temp_indices
        )
        print(f"class: {c} | {temp_indices}")
    action_partitioned_indices = np.hstack(action_partitioned_indices)

    # np.random.shuffle(action_partitioned_indices) #in place shuffle

    # action partitioned 
    y_action_partitioned = np.take(y_data_all_batches, action_partitioned_indices)
    x_action_partitioned = np.take(x_data_all_batches, action_partitioned_indices, axis=0) #pull along the 0 axis to get image arrays 
    print(x_action_partitioned.shape)

    #set up remaining data which will be used to fine tune the smart client after receiving aggregated parameters from the server
    remaining_indices = np.array([i for i in range(len(y_data_all_batches)) if i not in action_partitioned_indices])

    np.random.shuffle(remaining_indices)
    y_remaining = np.take(y_data_all_batches, remaining_indices)
    x_remaining = np.take(x_data_all_batches, remaining_indices, axis=0)

    action_partitioned_dataset = SmartCleintDataset(
        x_action_partitioned, 
        y_action_partitioned, 
        feature_name=feature_name, 
        label_name=label_name
    )
    remaining_partitioned_dataset = SmartCleintDataset(
        x_remaining, 
        y_remaining,
        feature_name=feature_name,
        label_name=label_name
    )

    action_partitioned_dataloader = DataLoader(
        action_partitioned_dataset, 
        batch_size=client_config_file.LOCAL_TRAINING_BATCH_SIZE, 
        shuffle=shuffle
    )
    remaining_partitioned_dataloader = DataLoader(
        remaining_partitioned_dataset, 
        batch_size=client_config_file.LOCAL_TRAINING_BATCH_SIZE, 
        shuffle=shuffle
    )
    # return action_partitioned_indices, remaining_indices
    return action_partitioned_dataloader, remaining_partitioned_dataloader