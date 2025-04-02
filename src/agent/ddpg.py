import os
import csv
import json
import ptan
import math
import json
import torch
import random
import numpy as np
import pandas as pd
from torch import nn
import pandas as pd
import torch.nn.functional as F
from collections import namedtuple
from utils.agent_train_test_utils import (
    get_exp_source_first_last, unpack_batch_a2c, calc_logprob,
    get_training_batches, create_agent_training_data, float32_preprocessor,
    unpack_batch_ddqn, ExperienceReplayBuffer
)
from config_folder import agent_config_file, client_config_file

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
# torch.manual_seed(42)

# seed = 42
# torch.manual_seed(seed)
# np.random.seed(seed)
# random.seed(seed)

class DDPGActor(nn.Module):
    def __init__(self, obs_size, act_size):
        super(DDPGActor, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(obs_size, 400),
            nn.Softplus(),
            nn.Linear(400, 300),
            nn.Softplus(),
            nn.Linear(300, act_size),
            nn.Softplus()
        )
    
    def forward(self, x):
        return self.net(x)
    
class DDPGCritic(nn.Module):
    def __init__(self, obs_size, act_size):
        super(DDPGCritic, self).__init__()

        self.obs_net = nn.Sequential(
            nn.Linear(obs_size, 400),
            nn.Softplus(),
        )

        self.out_net = nn.Sequential(
            nn.Linear(400 + act_size, 300),
            nn.Softplus(),
            nn.Linear(300, 1)
        )
    
    def forward(self, x , a):
        obs = self.obs_net(x)
        return self.out_net(torch.cat([obs, a], dim=1))
    


class AgentDDPG(ptan.agent.BaseAgent):
    def __init__(self, net, device=agent_config_file.DEVICE if torch.cuda.is_available() else "cpu",
                ou_enabled=True, ou_mu=0.0, ou_theta=0.15, ou_sigma=0.2, ou_epsilon=1.0):
        self.net = net
        self.device = device
        self.ou_enabled = ou_enabled
        self.ou_mu = ou_mu
        self.ou_theta = ou_theta
        self.ou_sigma = ou_sigma
        self.ou_epsilon = ou_epsilon
    
    def initial_state(self):
        return None
    
    def __call__(self, states, agent_states):
        states_v = float32_preprocessor(states)
        states_v = states_v.to(self.device)
        mu_v = self.net(states_v)
        actions = mu_v.data.cpu().numpy()
        if self.ou_enabled and self.ou_epsilon > 0:
            new_a_states = []
            for a_state, action in zip(agent_states, actions):
                if a_state is None:
                    a_state = np.zeros(
                        shape=action.shape, dtype=np.float32
                    )
                    a_state += self.ou_theta * (self.ou_mu - a_state)
                    a_state += self.ou_sigma * np.random.normal(
                        size=action.shape
                    )
                    action+= self.ou_epsilon * a_state
                    new_a_states.append(a_state)
                else:
                    new_a_states = agent_states
                actions = np.clip(
                    actions,
                    agent_config_file.LBOUND,
                    agent_config_file.UBOUND 
                )
        return actions, new_a_states
    

class DDPG:
    def __init__(
        self,
        obs_size, 
        act_size, 
        experiment_type,
        optimize_metric,
        actor_path=agent_config_file.DDPG_ACTOR_SAVE_PATH,
        critic_path=agent_config_file.DDPG_CRITIC_SAVE_PATH,
        dataset_name="cifar10",
        total_server_rounds=100,
        seed = 42,
        device=agent_config_file.DEVICE  if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize agent
        """
        self.device = device
        self.experiment_type = experiment_type
        self.optimize_metric = optimize_metric
        self.seed = seed
        
        #pt is the format the model will be saved in. Actor/Critic path comes from client config and experiment type comes from the current type of experiment.
        self.actor_save_path = f"{actor_path}_{dataset_name}_seed{self.seed}_{experiment_type}.pt" 
        self.critic_save_path = f"{critic_path}_{dataset_name}_seed{self.seed}_{experiment_type}.pt"

        self.actor = DDPGActor(
            obs_size,
            act_size
        ).to(self.device)
        
        self.actor_optimizer = torch.optim.SGD(
            params = self.actor.parameters(),
            # weight_decay=1e-1,                        
            lr = agent_config_file.ACTOR_LEARNING_RATE
        )

        self.critic = DDPGCritic(
            obs_size,
            act_size
        ).to(self.device)
        
        self.critic_optimizer = torch.optim.SGD(
            params = self.critic.parameters(),
            # weight_decay=1e-1,       
            # momentum=0.1,                  
            lr = agent_config_file.CRITIC_LEARNING_RATE
        )

        
        # self.actor_lr_scheduler = torch.optim.lr_scheduler.StepLR(
        #     self.actor_optimizer, 
        #     step_size=1,#agent_config_file.EPISODE_LENGTH//4, 
        #     gamma=0.1, 
        #     verbose=True
        # ) 
        self.actor_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.actor_optimizer, 
            T_max=total_server_rounds//agent_config_file.EPISODE_LENGTH, 
            verbose=True
        ) 
        # self.critic_lr_scheduler = torch.optim.lr_scheduler.StepLR(
        #     self.critic_optimizer, 
        #     step_size=1,#agent_config_file.EPISODE_LENGTH//4, 
        #     gamma=0.1,#0.02, 
        #     verbose=True
        # ) 
        self.critic_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.critic_optimizer, 
            T_max=total_server_rounds//agent_config_file.EPISODE_LENGTH, 
            verbose=True
        ) 

        print(f"Checking for saved actor model checkpoint")
        if os.path.isfile(self.actor_save_path):
            print(f"Loading saved actor checkpoint {self.actor_save_path}")
            checkpoint = torch.load(
                self.actor_save_path,
                map_location=agent_config_file.DEVICE if torch.cuda.is_available() else "cpu"
            )
            self.actor.load_state_dict(checkpoint['model_state_dict'])
            self.actor_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.actor_lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        else:
            print(f"No actor checkpoint found")
        
        print(f"Checking for saved critic checkpoint")
        if os.path.isfile(self.critic_save_path):
            print(f"Loading saved critic checkpoint {self.critic_save_path}")
            checkpoint = torch.load(
                self.critic_save_path,
                map_location=agent_config_file.DEVICE if torch.cuda.is_available() else "cpu"
            )
            self.critic.load_state_dict(checkpoint['model_state_dict'])
            self.critic_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.critic_lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        else:
            print(f"No critic checkpoint found")

        self.agent = AgentDDPG(
            self.actor,
            device = self.device
        )

    def get_actor(self):
        return self.actor
    
    def set_exp_source(self, server_round):
        """
        Set experience source to train the agent on. 
        """
        self.training_data = create_agent_training_data(
            server_round,
            agent_training_data_file=agent_config_file.AGENT_TRAINING_DATA_FILE,
            metric=self.optimize_metric
        )
        self.exp_source = get_exp_source_first_last(
            self.training_data, 
            step_count=agent_config_file.REWARD_STEPS
        )

    def train_ddpg(
            self, 
            server_round, 
            replay_buffer_size=agent_config_file.EPISODE_LENGTH
        ):
        self.server_round = server_round
        self.set_exp_source(self.server_round) #set the source of experience/trajectories the agent will train on
        
        #target networks
        target_actor_net = ptan.agent.TargetNet(self.actor)
        target_critic_net = ptan.agent.TargetNet(self.critic)

        #set model to torch training mode
        self.actor.train() 
        self.critic.train()
        buffer = ExperienceReplayBuffer(
            self.exp_source, buffer_size=replay_buffer_size
        )
        batch_id = 0
        # return buffer
        try:
            while True: 
                # try:
                buffer.populate(1)
                batch = buffer.sample(agent_config_file.BATCH_SIZE)
                batch_id += 1
                states_v, actions_v, rewards_v,\
                dones_mask, last_states_v = unpack_batch_ddqn(
                    batch, device=agent_config_file.DEVICE if torch.cuda.is_available() else "cpu"
                )
                # train critic
                self.critic_optimizer.zero_grad()
                q_v = self.critic(states_v, actions_v)
                last_act_v = target_actor_net.target_model(
                    last_states_v)
                q_last_v = target_critic_net.target_model(
                    last_states_v, last_act_v)
                q_last_v[dones_mask] = 0.0
                q_ref_v = rewards_v.unsqueeze(dim=-1) + \
                            q_last_v * agent_config_file.GAMMA
                critic_loss_v = F.mse_loss(q_v, q_ref_v.detach())
                critic_loss_v.backward()
                self.critic_optimizer.step()
                self.critic_lr_scheduler.step()

                # train actor
                self.actor_optimizer.zero_grad()
                cur_actions_v = self.actor(states_v)
                actor_loss_v = -self.critic(states_v, cur_actions_v)
                actor_loss_v = actor_loss_v.mean()
                actor_loss_v.backward()
                self.actor_optimizer.step()
                self.actor_lr_scheduler.step()

                target_actor_net.alpha_sync(alpha=1 - 1e-3)
                target_critic_net.alpha_sync(alpha=1 - 1e-3)

                print(f"Batch Size: {len(batch)}| Batch {batch_id} | Critic Loss {critic_loss_v.item()} | Actor Loss {actor_loss_v.item()}")
                if os.path.isfile(agent_config_file.AGENT_TRAINING_METRICS_FILE):
                    print(f"Writing data to {agent_config_file.AGENT_TRAINING_METRICS_FILE}")
                    with open(agent_config_file.AGENT_TRAINING_METRICS_FILE, 'a') as file:
                        writer = csv.writer(file)
                        writer.writerow(
                            [
                                batch_id,
                                self.server_round,
                                critic_loss_v.item(),
                                actor_loss_v.item(),
                                json.dumps(self.experiment_type),
                                self.seed
                            ]
                        )
                else:
                    with open(agent_config_file.AGENT_TRAINING_METRICS_FILE, 'w', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow(
                            [
                                'batch_id',
                                'server_round',
                                'critic_loss',
                                'actor_loss',
                                'experiment_type',
                                'seed'
                            ]
                        )
                        writer.writerow(
                            [
                                batch_id,
                                self.server_round,
                                critic_loss_v.item(),
                                actor_loss_v.item(),
                                json.dumps(self.experiment_type),
                                self.seed
                            ]
                        )
                    print(f"New agent metrics file created: {agent_config_file.AGENT_TRAINING_METRICS_FILE}")
                
                #save model checkpoint
                torch.save({
                    'model_state_dict': self.actor.state_dict(),
                    'optimizer_state_dict': self.actor_optimizer.state_dict(),
                    'scheduler_state_dict': self.actor_lr_scheduler.state_dict(),
                    'loss': actor_loss_v.item()
                }, self.actor_save_path)

                #save model checkpoint
                torch.save({
                    'model_state_dict': self.critic.state_dict(),
                    'optimizer_state_dict': self.critic_optimizer.state_dict(),
                    'scheduler_state_dict': self.critic_lr_scheduler.state_dict(),
                    'loss': critic_loss_v.item()
                }, self.critic_save_path)
        except Exception as e:
            print(e)
