from torch.utils.data import DataLoader
from torchvision import transforms
from flwr_datasets import FederatedDataset
from flwr_datasets.utils import divide_dataset
from flwr_datasets.partitioner import DirichletPartitioner
from torch.utils.data import DataLoader, DistributedSampler

from config_folder import client_config_file
from utils.utils import set_torch_device
# DEVICE = set_torch_device()

class DataTransforms:
    def __init__(self, dataset_name, feature_name, input_size):
        self.dataset_name = dataset_name
        self.feature_name = feature_name
        self.input_size = input_size

    def set_transform(self):
        if self.dataset_name=="fashion_mnist":
            self.data_transforms = transforms.Compose([
                transforms.Resize(self.input_size),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor()
            ])
        elif self.dataset_name=="cifar100":
            self.data_transforms = transforms.Compose([
                transforms.Resize(self.input_size),
                transforms.ToTensor()
            ])
        elif self.dataset_name=="cifar10":
            self.data_transforms = transforms.Compose([
                transforms.Resize(self.input_size),
                transforms.ToTensor()
            ])

    def apply_transforms(self, batch):
        self.set_transform()
        batch[self.feature_name] = [self.data_transforms(img) for img in batch[self.feature_name]]
        return batch

##create partitioned datasets
def load_datasets(
    partition_id, num_partitions,
    dataset_name="fashion_mnist", feature_name="image", label_name="label",
    input_size=224, batch_size=16, local_test_size=0.2, seed=42
    ):
    """
    Partition the dataset into the specified number of client and the test/neutral datasets
    dataset_name: name of the dataset
    num_clients: number of clients

    Returns:
    train_dataloaders: list of pytorch dataloaders for local clients
    neutral_dataset: flower dataset for testing
    """
    print(f"Dataset Name: {dataset_name}")
    data_transformer = DataTransforms(
        dataset_name=dataset_name,
        feature_name=feature_name,
        input_size=input_size
    )

    partitioner = DirichletPartitioner(
        num_partitions=num_partitions, 
        partition_by=label_name,
        alpha=0.3, 
        min_partition_size=10,
        self_balancing=True,
        seed=seed
    )
    fds = FederatedDataset(
        dataset=dataset_name,
        partitioners={
            "train": partitioner
        },
        seed=seed
    )

    try:
        neutral_dataset = fds.load_split("test")
    except:
        neutral_dataset = None

    # local_train_dataloaders = []
    # local_val_dataloaders = []
    # for p in range(num_partitions):
    partition = fds.load_partition(partition_id, "train")
    partition = partition.with_transform(data_transformer.apply_transforms)
    # train_dataset, val_dataset = divide_dataset(partition, [1.0-local_test_size, local_test_size])
    partition_local_train_test = partition.train_test_split(test_size=local_test_size, seed=42)
    
    train_dataloader = DataLoader(
        partition_local_train_test["train"],
        batch_size=batch_size, 
        shuffle=True
    )
    val_dataloader = DataLoader(
        partition_local_train_test["test"],
        batch_size=batch_size, 
        shuffle=False
    )

    # local_train_dataloaders.append(train_dataloader)
    # local_val_dataloaders.append(val_dataloader)
    
    if neutral_dataset!=None:
        neutral_dataset = neutral_dataset.with_transform(data_transformer.apply_transforms)
        neutral_dataloader = DataLoader(
            neutral_dataset,
            batch_size=batch_size,
            shuffle=False
        )
    
    #this will return one extra test dataloader since there is one partition for each client and one additional public partition
    #this extra dataloader can be ignored or combined with the public partition to create a bigger public partition
    return train_dataloader, val_dataloader, neutral_dataloader, fds.partitioners