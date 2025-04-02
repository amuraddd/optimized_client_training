#agent training variables
DEVICE='cuda'
DEVICE_MPS='mps'
DEVICE_CPU='cpu'

# agent training params
BATCH_SIZE = 4
GAMMA = 0.99
# ENTROPY_BETA = 1e-4
REWARD_STEPS = BATCH_SIZE #this has a big impact on performance as well as learning rate. This should be the same as episode length.
EPISODE_LENGTH = BATCH_SIZE #size of the episodes to train the agent on - each episode is composed of 1 or many steps

#model parameters
HID_SIZE = 128

#model learning rate
ACTOR_LEARNING_RATE=0.02 #0.1
CRITIC_LEARNING_RATE=0.05#5e-5#1e-4#5e-5

#action/data thresholds
LBOUND=0.1
UBOUND=0.8
LOOKBACK_PERIOD_FOR_WEIGHTED_METRIC_ACTION=2

#agent model checkpoint path
BEHAVIOR_MODEL_SAVE_PATH = "src/saved_models/agent_models/behavior/a2c_model.pt"
TARGET_MODEL_SAVE_PATH = "src/saved_models/agent_models/target/a2c_model.pt"
DDPG_ACTOR_SAVE_PATH = "src/saved_models/agent_models/ddpg/ddpg_actor"
DDPG_CRITIC_SAVE_PATH = "src/saved_models/agent_models/ddpg/ddpg_critic"

#persistent variables
PERSISTENT_VARIABLES = "config_folder/ddpg_persistent_variables.json"

#agent training data file
AGENT_TRAINING_DATA_FILE = "data/agent_data/ddpg/ddpg_agent_training_data.csv"

#agent training metrics
AGENT_TRAINING_METRICS_FILE = "data/saved_metrics/ddpg/ddpg_agent_training_metrics.csv"
