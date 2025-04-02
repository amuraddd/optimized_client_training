import math
import torch
import json
import ptan
import pandas as pd
import numpy as np
from collections import namedtuple
from config_folder import agent_config_file, client_config_file
from utils.utils import set_torch_device
# device = set_torch_device()

def calc_logprob(mu_v, var_v, actions_v):
    """
    Logarithm of the taken actions given the policy.
    """
    p1 = -((mu_v - actions_v)**2)/(2*var_v.clamp(min=1e-3))
    p2 = -torch.log(torch.sqrt(2 * math.pi * var_v))
    return p1 + p2

def exp_loss(x, a, b, c):
    """
    Function to estimate loss
    """
    return a*np.exp(-b*x)+c

def create_agent_training_data(
        server_round,
        agent_training_data_file='data/agent_data/agent_training_data.csv',
        episode_length=agent_config_file.EPISODE_LENGTH,
        metric='precision'
    ):
    """
    Create episodes of training data from the data collected during the federated learning communication rounds.
    This function will take the total data and create sets of it based on the EPISODE_LENGTH.
    It is based on the idea that a set of transitions/communication rounds that is of size equal to the EPISODE_LENGTH is one episode.
    The generated training data is a collection of complete episodes. These episodes are then broken into training batches.
    """
    from torch import tensor
    df = pd.read_csv(agent_training_data_file)[server_round-agent_config_file.EPISODE_LENGTH:server_round]
    # df = pd.concat([df]*100, ignore_index=True) #just for testing
    print(f"Total state, action, reward tuples: {len(df)}")
    Episode = namedtuple(
        'Episode', field_names=['episode_reward', 'steps']
    )
    EpisodeStep = namedtuple(
        'EpisodeStep', field_names=['state', 'action', 'reward']
    )

    episode_steps = []
    episode_reward = 0
    agent_training_data = []
    for row in df.iterrows():
        state = torch.tensor(
            np.array(
                list(
                    json.loads(
                        row[1][f'{metric}_on_local_dataset_after_aggregation']
                        )[metric].values()
                    )
            )
        )
        action = torch.tensor(np.array(json.loads(row[1]['action'])))
        reward = eval(row[1]['reward']) #this will only work if you have imported tensor from torch
        step = EpisodeStep(
            state=state,
            action=action,
            reward=reward
        )
        
        # print(len(episode_steps)) # <= episode_length
        if len(episode_steps) != episode_length: #the length of the episode determines training frequency - if data for atleast one episode is not collected then training can not happen.
            episode_steps.append(step)
            episode_reward += reward.item()
        if len(episode_steps) == episode_length:
            episode = Episode(
                episode_reward = episode_reward,
                steps = episode_steps
            )
            agent_training_data.append(episode)
            
            episode_reward = 0
            episode_steps = []
    # print(agent_training_data)
    return agent_training_data

def get_training_batches(
        exp_source,
        batch_size=64
    ):
    """
    Create a list of batches containing all transitions for training.
    """
    batch = []
    all_batches = []
    for idx, exp in enumerate(exp_source):
        batch.append(exp)
        if len(batch)<batch_size:
            continue
        all_batches.append(batch)
        batch = []
    all_batches.append(batch) #the final batch which will have less items them BATCH_SIZE
    return all_batches

def get_exp_source_first_last(batch, step_count=2):
    """
    build experience to train on
    """
    ExperienceSourceFirstLast = namedtuple(
        'ExperienceSourceFirstLast', field_names=['state', 'action', 'reward', 'last_state']
    )
    exp_source = []
    for idx, exp in enumerate(batch):
        for step_idx, step in enumerate(exp.steps):
            if step_idx+step_count<len(exp.steps):
                current_exp = ExperienceSourceFirstLast(
                    state=step.state,
                    action=step.action, 
                    reward=step.reward, 
                    last_state=exp.steps[step_idx+step_count].state)
            elif step_idx+step_count>=len(exp.steps):
                current_exp = ExperienceSourceFirstLast(
                    state=step.state,
                    action=step.action,
                    reward=step.reward, 
                    last_state=None)
            exp_source.append(current_exp)
    return exp_source

def float32_preprocessor(list_of_tensors, types='state'):
    """
    process tensors
    """
    if types=='state':
        tensor = torch.tensor(
            np.array([i.numpy() for i in list_of_tensors]),
            dtype=torch.float32
        )
        return tensor
    if types=='action':
        tensor = torch.tensor(
            np.array([i.numpy() for i in list_of_tensors]),
            dtype=torch.float32
        )
        return tensor
    
def unpack_batch_a2c(batch, net, last_val_gamma, device="cpu"):
    """
    Convert batch into training tensors
    :param batch:
    :param net:
    :return: states variable, actions tensor, reference values variable
    """
    states = []
    actions = []
    rewards = []
    not_done_idx = []
    last_states = []
    for idx, exp in enumerate(batch):
        states.append(exp.state)
        actions.append(exp.action)
        rewards.append(exp.reward)
        if exp.last_state is not None:
            not_done_idx.append(idx)
            last_states.append(exp.last_state)
    states_v = float32_preprocessor(states, types='state').to(device)
    actions_v = float32_preprocessor(actions, types='action').to(device)

    # handle rewards
    rewards_np = np.array(rewards, dtype=np.float32)
    if not_done_idx:
        last_states_v = float32_preprocessor(last_states).to(device)
        last_vals_v = net(last_states_v)[2]
        last_vals_np = last_vals_v.data.cpu().numpy()[:, 0]
        rewards_np[not_done_idx] += last_val_gamma * last_vals_np

    ref_vals_v = torch.FloatTensor(rewards_np).to(device)
    return states_v, actions_v, ref_vals_v


def unpack_batch_ddqn(batch, device="cpu"):
    states, actions, rewards, dones, last_states = [], [], [], [], []
    for exp in batch:
        states.append(exp.state)
        actions.append(exp.action)
        rewards.append(exp.reward)
        dones.append(exp.last_state is None)
        if exp.last_state is None:
            last_states.append(exp.state)
        else:
            last_states.append(exp.last_state)
    states_v = float32_preprocessor(states).to(device)
    actions_v = float32_preprocessor(actions).to(device)
    rewards_v = float32_preprocessor(rewards).to(device)
    last_states_v = float32_preprocessor(last_states).to(device)
    dones_t = torch.BoolTensor(dones).to(device)
    return states_v, actions_v, rewards_v, dones_t, last_states_v

class ExperienceReplayBuffer:
    def __init__(self, experience_source, buffer_size):
        self.experience_source_iter = None if experience_source is None else iter(experience_source)
        self.buffer = []
        self.capacity = buffer_size
        self.pos = 0

    def __len__(self):
        return len(self.buffer)

    def __iter__(self):
        return iter(self.buffer)

    def sample(self, batch_size):
        """
        Get one random batch from experience replay
        TODO: implement sampling order policy
        :param batch_size:
        :return:
        """
        if len(self.buffer) <= batch_size:
            return self.buffer
        # Warning: replace=False makes random.choice O(n)
        keys = np.random.choice(len(self.buffer), batch_size, replace=True)
        return [self.buffer[key] for key in keys]

    def _add(self, sample):
        if len(self.buffer) < self.capacity:
            self.buffer.append(sample)
        else:
            self.buffer[self.pos] = sample
        self.pos = (self.pos + 1) % self.capacity

    def populate(self, samples):
        """
        Populates samples into the buffer
        :param samples: how many samples to populate
        """
        for _ in range(samples):
            entry = next(self.experience_source_iter)
            self._add(entry)