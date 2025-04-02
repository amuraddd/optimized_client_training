DEVICE="cuda"
# DEVICE_CUDA='cuda'
DEVICE_MPS='mps'
DEVICE_CPU='cpu'
NUM_CLIENTS = 8
NUM_PARTITIONS = NUM_CLIENTS #one extra partition for public partition dataset
TOTAL_SERVER_ROUNDS = [100, 100, 100]
LOCAL_TRAINING_EPOCHS = 1
LOCAL_LEARNING_RATE = 1e-5 #1e-3, 1e-5
LOSS_ESTIMATION_WAITING_PERIOD = 5
LOCAL_TRAINING_BATCH_SIZE = 16
FINE_TUNE_EPOCHS = TOTAL_SERVER_ROUNDS
FINE_TUNE_LEARNING_RATE = 1e-6 #1e-5, 1e-6

DATASETS = [
    {
        'name': 'cifar100',
        'num_classes': 100,
        'feature_name': 'img',
        'label_name': 'fine_label',
        'input_shape': 224,
        'training_periods': TOTAL_SERVER_ROUNDS[0],
        'optimizer_config':
            {
                'learning_rate': 0.001,
                'learning_rate_decay': 0.1,
                'learning_rate_decay_period': 30,
                'weight_decay': 1e-4,
            },
    },
    {
        'name': 'cifar10',
        'num_classes': 10,
        'feature_name': 'img',
        'label_name': 'label',
        'input_shape': 224,
        'training_periods': TOTAL_SERVER_ROUNDS[1],
        'optimizer_config':
            {
                'learning_rate': 0.001,
                'learning_rate_decay': 0.1,
                'learning_rate_decay_period': 30,
                'weight_decay': 1e-4,
            },
    },
    {
        'name': 'fashion_mnist',
        'num_classes': 10,
        'feature_name': 'image',
        'label_name': 'label',
        'input_shape': 224,
        'training_periods': TOTAL_SERVER_ROUNDS[2],
        'optimizer_config':
            {
                'learning_rate': 0.0001,
                'learning_rate_decay': 0.1,
                'learning_rate_decay_period': 30,
                'weight_decay': 1e-4,
            },
    }
]

#client data files
OTP_CLONE_METRICS_FILE = "data/saved_metrics/ddpg/ddpg_otp_clone_metrics.csv"
OTP_METRICS_FILE = "data/saved_metrics/ddpg/ddpg_otp_metrics.csv"
NTP_METRICS_FILE = "data/saved_metrics/ddpg/ddpg_ntp_metrics.csv"

#server checkpoint path
def get_server_checkpoint_path(filename):
    return f"src/saved_models/server_models/server/{filename}.pth"

def get_client_checkpoint_path(filename):
    return f"src/saved_models/optimal_client/{filename}.pth"
