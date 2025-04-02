import torch
import numpy as np
import pandas as pd
from config_folder import agent_config_file

def get_reward(
    action, 
    estimated_local_loss, 
    loss_after_aggregation, 
    epsilon=0.01, 
    terminal_state="no"
): #w_c_before, w_c_after, 
    """
    action: action taken in updating the tricky local
    Loss after local training should be lower on the local dataset compared to the loss after aggregation on the local dataset.

    terminal_state: flag for whether if the current state is the terminal state
    """
    return ((loss_after_aggregation-estimated_local_loss)/estimated_local_loss)*(1/(torch.mean(action) - epsilon))
    # return ((loss_after_aggregation-estimated_local_loss)/estimated_local_loss)*(1/torch.sigmoid((torch.mean(action - epsilon))))
    # return ((estimated_local_loss-loss_after_aggregation)/loss_after_aggregation)*(1/torch.sigmoid((torch.mean(action - epsilon))))


def get_class_counts_from_dataloader(
        dataloader, 
        feature_name, 
        label_name
    ):
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

    class_counts = pd.DataFrame(
        np.unique(
            y_data_all_batches,
            return_counts=True
        )
    ).T
    class_counts.columns = ['class', 'counts']
    
    return class_counts

class AgentAction:
    def __init__(self, 
                n, action_buffer, agent_model,
                current_state, class_counts, 
                lbound=0.1, ubound=0.8,
                optimize_metric='f1_scores', lookback_period_for_weighted_metric=agent_config_file.LOOKBACK_PERIOD_FOR_WEIGHTED_METRIC_ACTION,
                meta_action_type='take_epsilon_greedy_normalized_action',
                device=agent_config_file.DEVICE  if torch.cuda.is_available() else "cpu"):
        self.n = n
        self.action_buffer = action_buffer
        self.agent_model = agent_model
        self.current_state = current_state
        self.class_counts = class_counts
        self.lbound = lbound
        self.ubound = ubound
        self.optimize_metric = optimize_metric
        self.lookback_period_for_weighted_metric = lookback_period_for_weighted_metric
        self.meta_action_type = meta_action_type
        self.device = device

    def get_normalized_action(self, action):
        self.class_counts['normalized'] = np.round((action/np.sum(action))*self.class_counts['counts'].sum(),0)
        adjusted_counts = list()
        for i, j in zip(self.class_counts['counts'], self.class_counts['normalized']):
            if j<=i:
                adjusted_counts.append(int(j))
            else:
                adjusted_counts.append(int(i))
        self.class_counts['adjusted'] = adjusted_counts
        self.class_counts['normalized_action'] = self.class_counts['adjusted']/self.class_counts['counts']
        return self.class_counts['normalized_action'].to_numpy()

    def take_epsilon_greedy_weighted_metric_action(self):
        """
        Return take_epsilon_greedy_weighted_metric_action
        """
        random_prob = np.random.uniform(0, 1)
        lookback = self.lookback_period_for_weighted_metric
        if len(self.action_buffer)>lookback:
            s0 = pd.DataFrame(self.action_buffer[self.n-lookback]['state'])
            s1 = self.current_state

            states = s0.merge(s1, left_on='class', right_on='class', suffixes=['_s0', '_s1'])

            states['diff'] = states[f'{self.optimize_metric}_s1'] - states[f'{self.optimize_metric}_s0']
            change = np.abs(states[['diff']][states[['diff']]<=0].fillna(0))
            sum = change['diff'].sum()
            weights = change['diff']/sum
            weights = weights+1 #percentage increase
            print(f"Current Round Action Weights: {weights}")
        else:
            weights = np.ones(len(self.current_state)) #all ones
            print(f"Current Round Action Weights: {weights}")

        self.current_state.sort_values(by=['class'], inplace=True)
        print(self.current_state.head(10))
        current_state_tensor = torch.tensor(
            self.current_state[self.optimize_metric].to_numpy(), 
            dtype=torch.float32
        ).to(self.device)

        with torch.no_grad():
            if (1/(self.n)**(1/2)) < random_prob: #add 1 to n to avoid division by zero
                _agent_action = self.agent_model(current_state_tensor)#[0]#0th index has the mu values for each class - use only for a2c
                agent_action = _agent_action.squeeze(dim=0).data.cpu().numpy()
                agent_action = np.clip(agent_action*weights, self.lbound, self.ubound)
                print("agent action")
                return agent_action, "agent_action" 
            else:
                try:
                    greedy_action = sorted(self.action_buffer, key=lambda by: (by['reward']), reverse=True)[0]['action']
                    greedy_action = np.clip(np.array(greedy_action)*weights, self.lbound, self.ubound)
                    print("greedy action")
                    return greedy_action, "greedy_action"
                except:
                    return np.clip(np.random.uniform(0, 1, len(current_state_tensor))*weights, self.lbound, self.ubound), "random_action"


    def take_epsilon_greedy_normalized_action(self):
        """
        Return take_epsilon_greedy_normalized_action
        """
        random_prob = np.random.uniform(0, 1)
        self.current_state.sort_values(by=['class'], inplace=True)
        print(self.current_state.head(10))
        current_state_tensor = torch.tensor(
            self.current_state[self.optimize_metric].to_numpy(), 
            dtype=torch.float32
        ).to(self.device)

        with torch.no_grad():
            if (1/(self.n)**(1/2)) < random_prob: #add 1 to n to avoid division by zero
                _agent_action = self.agent_model(current_state_tensor)#[0]#0th index has the mu values for each class - use only for a2c
                agent_action = _agent_action.squeeze(dim=0).data.cpu().numpy()
                agent_action = np.clip(
                    self.get_normalized_action(agent_action), 
                    self.lbound, 
                    self.ubound
                )
                print("agent action")
                return agent_action, "agent_action" 
            else:
                try:
                    greedy_action = sorted(self.action_buffer, key=lambda by: (by['reward']), reverse=True)[0]['action']
                    greedy_action = np.clip(
                        np.array(self.get_normalized_action(greedy_action)), 
                        self.lbound, 
                        self.ubound
                    )
                    print("greedy action")
                    return greedy_action, "greedy_action"
                except:
                    return np.clip(
                        self.get_normalized_action(
                            np.random.uniform(0, 1, len(current_state_tensor))
                        ),
                        self.lbound, 
                        self.ubound
                    ), "random_action"

    def take_random_action(self):
        """
        Return random action with random uniform probability.
        """
        return np.clip(np.random.uniform(0, 1, len(self.current_state)), self.lbound, self.ubound), "random_action"
    
    def action(self):
        if self.meta_action_type=="take_epsilon_greedy_weighted_metric_action":
            try:
                return self.take_epsilon_greedy_weighted_metric_action()
            except:
                return self.take_random_action()
        if self.meta_action_type=="take_epsilon_greedy_normalized_action":
            try:
                return self.take_epsilon_greedy_normalized_action()
            except:
                return self.take_random_action()
        if self.meta_action_type=="take_random_action":
            return self.take_random_action()