#!/usr/bin/env python
# coding: utf-8

# In[156]:


import math
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque


# ### Data Preparation

# In[157]:


# Define the 10 assets (tickers) for the portfolio
# tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "BA", "NFLX", "NVDA", "META", "SBUX"]
tickers = ["AAPL", "MSFT", "GOOGL", "SBUX", "TSLA"]
# Date range for historical data
start_date = "2019-01-01"
end_date   = "2023-12-31"

# Try to load price data from a local CSV, otherwise download using yfinance
data_file = "prices.csv"
try:
    prices_df = pd.read_csv(data_file, index_col=0, parse_dates=True)
    print("Loaded data from", data_file)
except FileNotFoundError:
    print("Downloading price data for tickers:", tickers)
    import yfinance as yf
    df = yf.download(tickers, start=start_date, end=end_date, interval="1d")
    # Extract the 'Close' prices from the MultiIndex DataFrame
    prices_df = df.xs('Close', axis=1, level='Price')
    prices_df.dropna(inplace=True)
    prices_df.to_csv(data_file)
    print("Data downloaded and saved to", data_file)

# Split data into training (first 4 years) and testing (last year)
train_df = prices_df[prices_df.index < "2023-01-01"]
test_df  = prices_df[prices_df.index >= "2023-01-01"]
train_prices = train_df.values  # shape: [train_days, 5]
test_prices  = test_df.values   # shape: [test_days, 5]
num_assets = train_prices.shape[1]
print(f"Training days: {train_prices.shape[0]}, Testing days: {test_prices.shape[0]}")


# ### State Encoding/Decoding and Actions

# In[158]:


def encode_state(increments):
    """
    Encode a state (tuple of 5 integer increments) as a 5-letter string.
    Each integer (0-20) represents allocation in 5% units.
    'A' corresponds to 0 (0%), 'B' to 1 (5%), ..., 'U' to 20 (100%).
    """
    return "".join(chr(ord('A') + inc) for inc in increments)

def decode_state(state_str):
    """
    Decode a 5-letter state string back into a tuple of 5 increments (0-20 each).
    """
    return tuple(ord(ch) - ord('A') for ch in state_str)

# Define all possible actions (including no-action) as tuples
all_actions = [(i, j) for i in range(num_assets) for j in range(num_assets) if i != j]
all_actions.append((None, None))  # (None,None) denotes no rebalance action
action_count = len(all_actions)  # should be 21 (for 5 assets)
# Create a mapping from action tuple to index in all_actions list
action_to_index = {act: idx for idx, act in enumerate(all_actions)}

def get_valid_actions(state):
    """
    Given a state (tuple of 5 increments or state string), return a list of valid action tuples.
    A transfer action (i,j) is valid if the current state has at least 5% in asset i (increment >= 1)
    and at most 95% in asset j (increment <= 19). The no-action (None,None) is always valid.
    """
    # If state is given as a string, decode it to a tuple of increments
    increments = decode_state(state) if isinstance(state, str) else tuple(state)
    valid_actions = []
    for act in all_actions:
        if act == (None, None):
            valid_actions.append(act)
        else:
            i, j = act
            if increments[i] >= 1 and increments[j] <= 19:
                valid_actions.append(act)
    return valid_actions

def apply_action(state, action):
    """
    Apply a rebalancing action to a state.
    - state: current state as a tuple of 5 increments (sums to 20).
    - action: a tuple (i, j) meaning transfer 5% (one increment) from asset i to asset j.
              (None, None) means no rebalancing.
    Returns the new state (tuple of 5 increments) after applying the action.
    """
    increments = list(state)
    if action is None or action == (None, None):
        # No rebalance
        return tuple(increments)
    i, j = action
    # Transfer one increment from i to j (if possible)
    if increments[i] >= 1 and increments[j] <= 19:
        increments[i] -= 1
        increments[j] += 1
    return tuple(increments)

def compute_reward(weights_frac, price_today, price_next):
    """
    Compute the log return of the portfolio for one time step.
    - weights_frac: list of 5 asset weight fractions after rebalancing on day t (sum=1).
    - price_today: prices of the 5 assets at day t.
    - price_next: prices of the 5 assets at day t+1.
    Returns: log(portfolio return) from day t to t+1.
    """
    # Portfolio value growth factor = sum_k w_k * (price_next_k / price_today_k)
    growth_factor = 0.0
    for k in range(num_assets):
        growth_factor += weights_frac[k] * (price_next[k] / price_today[k])
    # Reward is the log of the growth factor
    return math.log(growth_factor)


# ### Replay Buffer for Experience Replay

# In[159]:


class ReplayBuffer:
    def __init__(self, capacity):
        """
        Replay buffer to store past transitions for experience replay.
        Stores tuples of (state, action_index, reward, next_state, done).
        """
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action_idx, reward, next_state, done):
        """Save a transition to the buffer."""
        self.buffer.append((state, action_idx, reward, next_state, done))

    def sample(self, batch_size):
        """
        Sample a random batch of transitions from the buffer.
        Returns: tuples (states, action_idxs, rewards, next_states, dones) for the batch.
        """
        batch = random.sample(self.buffer, batch_size)
        # Extract each component into separate lists
        states, action_idxs, rewards, next_states, dones = zip(*batch)
        return list(states), list(action_idxs), list(rewards), list(next_states), list(dones)

    def __len__(self):
        """Current size of the buffer."""
        return len(self.buffer)


# ### Prioritized Replay Buffer

# In[ ]:


# class PrioritizedReplayBuffer:
#     def __init__(self, capacity, alpha=0.6):
#         """
#         Prioritized replay buffer with given capacity.
#         alpha: how much prioritization to use (0 = no prioritization, 1 = full prioritization).
#         """
#         self.capacity = capacity
#         self.alpha = alpha
#         self.buffer = []      # to store transitions (s, a, r, s', done)
#         self.priorities = np.zeros(capacity, dtype=np.float32)  # priority values for each entry
#         self.next_idx = 0
#         self.size = 0

#     def add(self, transition, priority=1.0):
#         """Add a new experience with given priority (use high priority for new transitions to ensure sampling)."""
#         max_prio = self.priorities.max() if self.size > 0 else 1.0
#         # Use max priority for new transitions to guarantee it will be sampled at least once
#         if priority is None:
#             priority = max_prio
#         # If buffer not yet full, append; otherwise overwrite oldest
#         if self.size < self.capacity:
#             self.buffer.append(transition)
#         else:
#             self.buffer[self.next_idx] = transition
#         # Set priority (raised to alpha)
#         self.priorities[self.next_idx] = (priority ** self.alpha)
#         # Move index forward and wrap around
#         self.next_idx = (self.next_idx + 1) % self.capacity
#         # Track size up to capacity
#         self.size = min(self.size + 1, self.capacity)

#     def sample(self, batch_size, beta=0.4):
#         """
#         Sample a batch of experiences with probabilities proportional to priorities.
#         beta: compensation factor for importance sampling (0 = no correction, 1 = full correction).
#         Returns (batch, indices, weights) for the sampled experiences.
#         """
#         assert self.size > 0, "Replay buffer is empty!"
#         # Compute probabilities for each entry
#         prios = self.priorities[:self.size]
#         probs = prios / prios.sum()
#         # Sample indices according to probability distribution
#         indices = np.random.choice(self.size, batch_size, p=probs, replace=False)
#         # Collect transitions for these indices
#         batch = [self.buffer[idx] for idx in indices]
#         # Compute importance-sampling weights to correct bias
#         total = self.size
#         weights = (total * probs[indices]) ** (-beta)   # smaller probability -> larger weight
#         weights = weights / weights.max()              # normalize weights (max weight = 1)
#         weights = np.array(weights, dtype=np.float32)
#         return batch, indices, weights

#     def update_priorities(self, indices, errors, epsilon=1e-3):
#         """
#         Update priorities of sampled transitions based on new TD-errors.
#         Adding a small epsilon to avoid zero priority.
#         """
#         for idx, error in zip(indices, errors):
#             new_priority = abs(error) + epsilon
#             self.priorities[idx] = (new_priority ** self.alpha)

#     def __len__(self):
#         """Current size of the buffer."""
#         return len(self.buffer)
  # if len(prioritized_replay_buffer) >= batch_size:
        #     # Sample a batch of transitions
        #     batch, idxs, weights = prioritized_replay_buffer.sample(batch_size)
        #     states_batch, actions_batch, rewards_batch, next_states_batch, dones_batch = zip(*batch)

        #     # Convert to tensors
        #     # State and next state inputs as batch_size x 5 tensors (normalize allocations to [0,1])
        #     state_tensor = torch.FloatTensor([ [inc/20.0 for inc in s] for s in states_batch ])
        #     next_state_tensor = torch.FloatTensor([ [inc/20.0 for inc in s] for s in next_states_batch ])
        #     action_tensor = torch.LongTensor(actions_batch)
        #     reward_tensor = torch.FloatTensor(rewards_batch)
        #     done_tensor   = torch.BoolTensor(dones_batch)

        #     # Compute current Q values for each state-action in the batch
        #     # policy_net(state_tensor) has shape [batch, action_dim]; gather along actions
        #     q_values = policy_net(state_tensor)  # [batch, action_count]
        #     state_action_values = q_values.gather(1, action_tensor.unsqueeze(1)).squeeze(1)
        #     # Compute target Q values using target network
        #     with torch.no_grad():
        #         next_q_values = target_net(next_state_tensor)  # [batch, action_count]
        #     target_values = []
        #     td_errors = []
        #     for i in range(batch_size):
        #         if done_tensor[i]:
        #             # If episode ended at this transition, target is just the immediate reward
        #             target = reward_tensor[i] 
        #         else:
        #             # Not done: use Bellman update with next state's max Q
        #             # Mask invalid actions for next state i
        #             next_state = next_states_batch[i]
        #             valid_next_actions = get_valid_actions(next_state)
        #             # Find indices of valid actions for next state
        #             valid_idxs = [action_to_index[a] for a in valid_next_actions]
        #             # Max Q-value among valid next actions
        #             max_next_Q = torch.max(next_q_values[i, valid_idxs])
        #             target = rewards_batch[i] + gamma * max_next_Q.item()

        #         target_values.append(target)
        #         td_error = (target - state_action_values[i].item())
        #         td_errors.append(td_error)
        #     target_tensor = torch.FloatTensor(target_values)
        #     # Optimize the model: MSE loss between state_action_values and target_values
        #     loss = F.mse_loss(state_action_values, target_tensor)
        #     optimizer.zero_grad()
        #     loss.backward()
        #     optimizer.step()
        #     # Now update the replay buffer priorities with the computed TD errors:
        #     prioritized_replay_buffer.update_priorities(idxs, td_errors)
        # if len(prioritized_replay_buffer) >= batch_size:
        #     # Sample a batch of transitions from the prioritized replay buffer
        #     batch, idxs, weights = prioritized_replay_buffer.sample(batch_size)
        #     states_batch, actions_batch, rewards_batch, next_states_batch, dones_batch = zip(*batch)

        #     # Convert to tensors:
        #     state_tensor = torch.FloatTensor([[inc / 20.0 for inc in s] for s in states_batch])
        #     next_state_tensor = torch.FloatTensor([[inc / 20.0 for inc in s] for s in next_states_batch])
        #     action_tensor = torch.LongTensor(actions_batch)
        #     reward_tensor = torch.FloatTensor(rewards_batch)
        #     done_tensor = torch.BoolTensor(dones_batch)

        #     # Compute current Q values for each state-action pair in the batch using the online network
        #     q_values = policy_net(state_tensor)  # shape: [batch, action_dim]
        #     state_action_values = q_values.gather(1, action_tensor.unsqueeze(1)).squeeze(1)

        #     # Compute target Q values using Double DQN logic:
        #     with torch.no_grad():
        #         if use_double_dqn:
        #             # Step 1: For each next_state, select the best action using the online network
        #             online_next_q = policy_net(next_state_tensor)  # shape: [batch, action_dim]
        #             best_actions = online_next_q.argmax(dim=1, keepdim=True)  # best action indices from online net
        #             # Step 2: Evaluate these actions using the target network
        #             target_next_q = target_net(next_state_tensor)
        #             selected_q = target_next_q.gather(1, best_actions).squeeze(1)
        #         else:
        #             # Standard DQN target: use the max Q-value from the target network directly
        #             target_next_q = target_net(next_state_tensor)
        #             selected_q = target_next_q.max(dim=1)[0]

        #     # If done flag is true for a transition, we do not add future reward.
        #     selected_q = selected_q * (1 - done_tensor.float())

        #     # Compute target values for the batch: 
        #     target_values = rewards_tensor = reward_tensor + gamma * selected_q

        #     # Now compute TD errors and per-sample losses:
        #     # Optionally, if using PER, we can compute element-wise loss:
        #     losses = torch.nn.functional.smooth_l1_loss(state_action_values, target_values, reduction='none')
        #     loss = (losses * torch.FloatTensor(weights)).mean() if 'weights' in locals() else losses.mean()

        #     optimizer.zero_grad()
        #     loss.backward()
        #     optimizer.step()

        #     # Compute individual TD errors for updating priorities:
        #     td_errors = []
        #     for i in range(batch_size):
        #         td_error = target_values[i].item() - state_action_values[i].item()
        #         td_errors.append(td_error)
        #     prioritized_replay_buffer.update_priorities(idxs, td_errors)


# ### Neural Network for Q-value Function

# In[161]:


class DQNNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        """
        Neural network that approximates Q(s,a) for all actions a given state s.
        state_dim: dimensionality of state input (e.g. 5)
        action_dim: number of possible actions (e.g. 21)
        """
        super(DQNNetwork, self).__init__()
        # Simple 3-layer MLP: two hidden layers and one output layer
        hidden1 = 128
        hidden2 = 128
        self.fc1 = nn.Linear(state_dim, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, action_dim)

    def forward(self, x):
        # x is a tensor of shape [batch_size, state_dim]
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q_values = self.fc3(x)  # outputs Q-values for each action
        return q_values


# ### DQN Agent Training Setup

# In[ ]:


# Hyperparameters
gamma = 0.97            # discount factor for future rewards
learning_rate = 5e-4  # learning rate for optimizer
epsilon_start = 1.0     # initial exploration rate
epsilon_min   = 0.2     # minimum exploration rate
epsilon_decay = 0.995    # multiplicative decay factor per episode
episodes = 100          # number of training episodes
batch_size = 64         # mini-batch size for replay updates
target_update_freq = 10 # how often (in episodes) to update the target network
replay_capacity = 10000 # capacity of the replay buffer
use_double_dqn = True  # use Double DQN for training
state_dim = 5
action_dim = action_count  # 21

# Initialize replay memory, policy network, target network, optimizer
replay_buffer = ReplayBuffer(replay_capacity)
# prioritized_replay_buffer = PrioritizedReplayBuffer(replay_capacity)

policy_net = DQNNetwork(state_dim, action_dim)
target_net = DQNNetwork(state_dim, action_dim)
# Copy initial weights to target network
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()  # target network in evaluation mode (not strictly necessary)
optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)

# Helper function: select action using epsilon-greedy policy
def select_action(state, epsilon):
    """
    Choose an action index for the given state using an epsilon-greedy strategy.
    - state: current state as a tuple of increments.
    - epsilon: current exploration rate.
    Returns: (action_idx, action_tuple)
    """
    valid_actions = get_valid_actions(state)
    if random.random() < epsilon:
        # Exploration: random valid action
        action = random.choice(valid_actions)
    else:
        # Exploitation: choose best action according to Q-network
        # Prepare state as a tensor (1x5) with normalized allocations
        state_input = torch.FloatTensor([inc/20.0 for inc in state]).unsqueeze(0)
        with torch.no_grad():
            q_values = policy_net(state_input)  # shape [1, action_dim]
        q_values = q_values.numpy().squeeze()  # shape [action_dim]
        # Mask out invalid actions by setting their Q-value very low
        # (So they won't be chosen as max)
        invalid_actions = set(all_actions) - set(valid_actions)
        for act in invalid_actions:
            idx = action_to_index[act]
            q_values[idx] = -1e9  # large negative to disable
        best_idx = int(np.argmax(q_values))
        action = all_actions[best_idx]
    # Return both the index and the tuple representation
    return action_to_index[action], action

# Training loop
initial_state = (4, 4, 4, 4, 4)  # start each episode with equal weights ("EEEEE")
epsilon = epsilon_start
train_days = train_prices.shape[0]
for ep in range(1, episodes+1):
    state = initial_state
    # Iterate over each day in training data (except last, as we look ahead one day for reward)
    for t in range(train_days - 1):
        # Choose action (epsilon-greedy)
        action_idx, action = select_action(state, epsilon)
        # Apply action to get new state
        new_state = apply_action(state, action)
        # Compute reward from day t to t+1
        weights_new = [inc/20.0 for inc in new_state]  # convert increments to fractions
        reward = compute_reward(weights_new, train_prices[t], train_prices[t+1])
        # Check if we've reached the end of an episode (done flag)
        done = (t == train_days - 2)  # True if next_state will be the last state of episode
        # Store the transition in replay memory
        replay_buffer.push(state, action_idx, reward, new_state, done)
        # prioritized_replay_buffer.add((state, action_idx, reward, new_state, done), priority=1.0)

        # Update state
        state = new_state
        # # Perform a learning step if we have enough samples
        if len(replay_buffer) >= batch_size:
            # Sample a batch of transitions
            states_batch, actions_batch, rewards_batch, next_states_batch, dones_batch = replay_buffer.sample(batch_size)

            # Convert to tensors
            # State and next state inputs as batch_size x 5 tensors (normalize allocations to [0,1])
            state_tensor = torch.FloatTensor([ [inc/20.0 for inc in s] for s in states_batch ])
            next_state_tensor = torch.FloatTensor([ [inc/20.0 for inc in s] for s in next_states_batch ])
            action_tensor = torch.LongTensor(actions_batch)
            reward_tensor = torch.FloatTensor(rewards_batch)
            done_tensor   = torch.BoolTensor(dones_batch)

            # Compute current Q values for each state-action in the batch
            # policy_net(state_tensor) has shape [batch, action_dim]; gather along actions
            q_values = policy_net(state_tensor)  # [batch, action_count]
            state_action_values = q_values.gather(1, action_tensor.unsqueeze(1)).squeeze(1)
            # Compute target Q values using target network
            with torch.no_grad():
                # next_q_values = target_net(next_state_tensor)  # [batch, action_count]
                if use_double_dqn:
                    # Step 1: For each next_state, select the best action using the online network
                    online_next_q = policy_net(next_state_tensor)  # shape: [batch, action_dim]
                    best_actions = online_next_q.argmax(dim=1, keepdim=True)  # best action indices from online net
                    # Step 2: Evaluate these actions using the target network
                    target_next_q = target_net(next_state_tensor)
                    selected_q = target_next_q.gather(1, best_actions).squeeze(1)
                else:
                    # Standard DQN target: use the max Q-value from the target network directly
                    target_next_q = target_net(next_state_tensor)
                    selected_q = target_next_q.max(dim=1)[0]

            selected_q = selected_q * (1 - done_tensor.float())

            # target_values = []
            # for i in range(batch_size):
            #     if done_tensor[i]:
            #         # If episode ended at this transition, target is just the immediate reward
            #         target_values.append(reward_tensor[i])
            #     else:
            #         # Not done: use Bellman update with next state's max Q
            #         # Mask invalid actions for next state i
            #         next_state = next_states_batch[i]
            #         valid_next_actions = get_valid_actions(next_state)
            #         # Find indices of valid actions for next state
            #         valid_idxs = [action_to_index[a] for a in valid_next_actions]
            #         # Max Q-value among valid next actions
            #         max_next_Q = torch.max(next_q_values[i, valid_idxs])
            #         target = rewards_batch[i] + gamma * max_next_Q.item()
            #         target_values.append(target)
            target_values = reward_tensor + gamma * selected_q

            # target_tensor = torch.FloatTensor(target_values)
            losses = F.smooth_l1_loss(state_action_values, target_values, reduction='none')
            # loss = (losses * torch.FloatTensor(weights)).mean() if 'weights' in locals() else losses.mean()
            loss = losses.mean()
            # Optimize the model: MSE loss between state_action_values and target_values
            # loss = F.mse_loss(state_action_values, target_values)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


    # Decay epsilon after each episode
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay
        epsilon = max(epsilon, epsilon_min)
    # Update target network periodically
    if ep % target_update_freq == 0:
        target_net.load_state_dict(policy_net.state_dict())
    if ep % 10 == 0 or ep == episodes:
        print(f"Episode {ep}/{episodes} completed, epsilon={epsilon:.3f}")
print("Training completed.")


# ### Policy Evaluation on Test Data

# In[171]:


def evaluate_policy(price_array, model, initial_state):
    """
    Simulate the portfolio value over time on given price data using the provided model (greedy policy).
    Returns a list of portfolio values for each day in the price data.
    """
    days = price_array.shape[0]
    state = initial_state
    portfolio_value = 1.0  # start with $1.0
    values = [portfolio_value]
    # Baseline (buy-and-hold equal weights) for comparison
    baseline_value = 1.0
    # Compute initial shares for baseline (with equal weights)
    baseline_weights = [0.2] * num_assets  # 20% each
    baseline_shares = [baseline_weights[i] * baseline_value / price_array[0][i] for i in range(num_assets)]
    for t in range(days - 1):
        # Agent action: choose greedy (highest Q) action for current state
        state_input = torch.FloatTensor([inc/20.0 for inc in state]).unsqueeze(0)
        with torch.no_grad():
            q_vals = model(state_input).numpy().squeeze()
        # Mask invalid actions for current state
        valid_acts = get_valid_actions(state)
        invalid_acts = set(all_actions) - set(valid_acts)
        for act in invalid_acts:
            q_vals[action_to_index[act]] = -1e9
        best_act_idx = int(np.argmax(q_vals))
        best_action = all_actions[best_act_idx]
        # Rebalance portfolio according to best action
        state = apply_action(state, best_action)
        # Compute portfolio growth factor from day t to t+1 for agent
        weights = [inc/20.0 for inc in state]
        growth_factor = 0.0
        for k in range(num_assets):
            growth_factor += weights[k] * (price_array[t+1][k] / price_array[t][k])
        portfolio_value *= growth_factor
        values.append(portfolio_value)
        # Update baseline value (its shares just appreciate with market, no rebalance)
        baseline_portfolio_val = 0.0
        for k in range(num_assets):
            baseline_portfolio_val += baseline_shares[k] * price_array[t+1][k]
        baseline_value = baseline_portfolio_val
    # After iterating, 'values' list contains portfolio value from start to end of period
    final_return = (portfolio_value - 1.0) / 1.0 * 100  # in %
    baseline_return = (baseline_value - 1.0) / 1.0 * 100
    print(f"Test period: Agent final portfolio value = {portfolio_value:.4f} (Return = {final_return:.2f}%)")
    print(f"Test period: Baseline final value = {baseline_value:.4f} (Return = {baseline_return:.2f}%)")
    return values, baseline_weights

# Evaluate the trained model on test data
agent_values, baseline_weights = evaluate_policy(test_prices, policy_net, initial_state)


# ### Dueling DQN

# In[164]:


class DuelingDQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DuelingDQN, self).__init__()
        # Shared feature extraction layers
        self.features = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )
        # Value stream: outputs one value per state
        self.value_stream = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        # Advantage stream: outputs advantage for each action
        self.adv_stream = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        x = self.features(x)
        value = self.value_stream(x)         # shape: [batch, 1]
        advantage = self.adv_stream(x)         # shape: [batch, action_dim]
        # Combine streams: Q(s,a) = V(s) + (A(s,a) - mean(A(s,·)))
        advantage_mean = advantage.mean(dim=1, keepdim=True)
        q_vals = value + (advantage - advantage_mean)
        return q_vals



# ### Dueling DQN Training Setup

# In[ ]:


# policy_net = DuelingDQN(state_dim, action_dim)
# target_net = DuelingDQN(state_dim, action_dim)
# target_net.load_state_dict(policy_net.state_dict())
# target_net.eval()
# optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
# replay_buffer = ReplayBuffer(replay_capacity)
# # Optionally, create a flag to indicate which version is being used, e.g.,
# use_dueling_dqn = True


# In[ ]:


# def select_action(state, epsilon):
#     """
#     Choose an action index for the given state using an epsilon-greedy strategy.
#     - state: current state as a tuple of increments.
#     - epsilon: current exploration rate.
#     Returns: (action_idx, action_tuple)
#     """
#     valid_actions = get_valid_actions(state)
#     if random.random() < epsilon:
#         # Exploration: random valid action
#         action = random.choice(valid_actions)
#     else:
#         # Exploitation: choose best action according to Q-network
#         # Prepare state as a tensor (1x5) with normalized allocations
#         state_input = torch.FloatTensor([inc/20.0 for inc in state]).unsqueeze(0)
#         with torch.no_grad():
#             q_values = policy_net(state_input)  # shape [1, action_dim]
#         q_values = q_values.numpy().squeeze()  # shape [action_dim]
#         # Mask out invalid actions by setting their Q-value very low
#         # (So they won't be chosen as max)
#         invalid_actions = set(all_actions) - set(valid_actions)
#         for act in invalid_actions:
#             idx = action_to_index[act]
#             q_values[idx] = -1e9  # large negative to disable
#         best_idx = int(np.argmax(q_values))
#         action = all_actions[best_idx]
#     # Return both the index and the tuple representation
#     return action_to_index[action], action

# # Training loop
# initial_state = (4, 4, 4, 4, 4)  # start each episode with equal weights ("EEEEE")
# epsilon = epsilon_start
# train_days = train_prices.shape[0]
# for ep in range(1, episodes+1):
#     state = initial_state
#     # Iterate over each day in training data (except last, as we look ahead one day for reward)
#     for t in range(train_days - 1):
#         # Choose action (epsilon-greedy)
#         action_idx, action = select_action(state, epsilon)
#         # Apply action to get new state
#         new_state = apply_action(state, action)
#         # Compute reward from day t to t+1
#         weights_new = [inc/20.0 for inc in new_state]  # convert increments to fractions
#         reward = compute_reward(weights_new, train_prices[t], train_prices[t+1])
#         # Check if we've reached the end of an episode (done flag)
#         done = (t == train_days - 2)  # True if next_state will be the last state of episode
#         # Store the transition in replay memory
#         replay_buffer.push(state, action_idx, reward, new_state, done)
#         # prioritized_replay_buffer.add((state, action_idx, reward, new_state, done), priority=1.0)

#         # Update state
#         state = new_state
#         # # Perform a learning step if we have enough samples
#         if len(replay_buffer) >= batch_size:
#             # Sample a batch of transitions
#             states_batch, actions_batch, rewards_batch, next_states_batch, dones_batch = replay_buffer.sample(batch_size)

#             # Convert to tensors
#             # State and next state inputs as batch_size x 5 tensors (normalize allocations to [0,1])
#             state_tensor = torch.FloatTensor([ [inc/20.0 for inc in s] for s in states_batch ])
#             next_state_tensor = torch.FloatTensor([ [inc/20.0 for inc in s] for s in next_states_batch ])
#             action_tensor = torch.LongTensor(actions_batch)
#             reward_tensor = torch.FloatTensor(rewards_batch)
#             done_tensor   = torch.BoolTensor(dones_batch)

#             # Compute current Q values for each state-action in the batch
#             # policy_net(state_tensor) has shape [batch, action_dim]; gather along actions
#             q_values = policy_net(state_tensor)  # [batch, action_count]
#             state_action_values = q_values.gather(1, action_tensor.unsqueeze(1)).squeeze(1)
#             # Compute target Q values using target network
#             with torch.no_grad():
#                 # next_q_values = target_net(next_state_tensor)  # [batch, action_count]
#                 if use_double_dqn:
#                     # Step 1: For each next_state, select the best action using the online network
#                     online_next_q = policy_net(next_state_tensor)  # shape: [batch, action_dim]
#                     best_actions = online_next_q.argmax(dim=1, keepdim=True)  # best action indices from online net
#                     # Step 2: Evaluate these actions using the target network
#                     target_next_q = target_net(next_state_tensor)
#                     selected_q = target_next_q.gather(1, best_actions).squeeze(1)
#                 else:
#                     # Standard DQN target: use the max Q-value from the target network directly
#                     target_next_q = target_net(next_state_tensor)
#                     selected_q = target_next_q.max(dim=1)[0]

#             selected_q = selected_q * (1 - done_tensor.float())

#             target_values = reward_tensor + gamma * selected_q


#             loss = F.mse_loss(state_action_values, target_values)
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#     if epsilon > epsilon_min:
#         epsilon *= epsilon_decay
#         epsilon = max(epsilon, epsilon_min)
#     # Update target network periodically
#     if ep % target_update_freq == 0:
#         target_net.load_state_dict(policy_net.state_dict())
#     if ep % 10 == 0 or ep == episodes:
#         print(f"Episode {ep}/{episodes} completed, epsilon={epsilon:.3f}")
# print("Training completed.")
# agent_values, baseline_weights = evaluate_policy(test_prices, policy_net, initial_state)


# In[ ]:





# In[ ]:




