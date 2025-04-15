#!/usr/bin/env python
# coding: utf-8

# In[34]:


# from google.colab import drive
# drive.mount('/content/drive')


# In[35]:


import math
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import yfinance as yf
from collections import deque


# ### Data Preparation

# In[36]:


# Define the 10 assets (tickers) for the portfolio
# tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "BA", "NFLX", "NVDA", "META", "SBUX"]
# tickers = ["AAPL", "MSFT", "GOOGL", "SBUX", "TSLA"]
tickers = ["GME", "AMC", "SPCE", "NVAX", "NOK"]

# Date range for historical data
start_date = "2015-01-01"
end_date   = "2023-12-31"

# Try to load price data from a local CSV, otherwise download using yfinance
data_file = "prices.csv"
try:
    prices_df = pd.read_csv(data_file, index_col=0, parse_dates=True)
    print("Loaded data from", data_file)
except FileNotFoundError:
    print("Downloading price data for tickers:", tickers)
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

# In[37]:


# Construct the new action space.
# Each action is a tuple (src, dst, amt) where:
# - src and dst are indices in assets_plus_cash (0..5)
# - amt is in [1, 2, 3, 4, 5] representing a 1%-5% transfer.
# We allow transferring funds between any two different assets.

# Define all possible actions (including no-action) as tuples
all_actions = []
cash_idx = num_assets            # index 5 is for cash
assets_plus_cash = list(range(num_assets)) + [cash_idx]  # [0,1,2,3,4,5]
percents = [1,2,3,4,5]
for pct in percents:
    for src in assets_plus_cash:
        for dst in assets_plus_cash:
            if src != dst:
                all_actions.append((src, dst, pct))
all_actions.append((None, None, 0))  # No action
action_count = len(all_actions) # 151
# print("Number of actions:", action_count)
# Create a mapping from action tuple to index in all_actions list
action_to_index = {act: idx for idx, act in enumerate(all_actions)}

def get_valid_actions(state):
    """
    Given a state (tuple of 6 integers summing to 100 representing allocations for 5 stocks and cash),
    return a list of valid actions (from all_actions_new).

    Rules:
      - If src is not cash, state[src] must be >= amt (i.e. you must have enough allocation to sell).
      - Optionally, we can restrict that if dst is not cash, the allocation after transfer should not exceed 100.
        (Here we assume no single asset can have more than 100%).
      - The no-action (None, None, 0) is always valid.
    """
    valid = []
    for act in all_actions:
        if act == (None, None, 0):
            valid.append(act)
        else:
            src, dst, amt = act
            # Check: if src is not cash, then it must have at least 'amt'
            if src != cash_idx and state[src] < amt:
                continue
            # Check: if dst is not cash, ensure it does not exceed 100.
            if dst != cash_idx and state[dst] + amt > 100:
                continue
            valid.append(act)
    return valid


def apply_action(state, action):
    """
    Apply an action (src, dst, amt) to the state.
    - state: tuple of 6 integers (allocations in % for 5 stocks and cash; sum to 100)
    - action: (src, dst, amt). For example, (0,5,3) means transfer 3% from asset 0 to cash.
      (None, None, 0) means no action.
    Returns the new state as a tuple of 6 integers.
    """
    state = list(state)
    if action == (None, None, 0):
        return tuple(state)
    src, dst, amt = action
    # If src is not cash, reduce its allocation by amt (clip at 0)
    if src is not None and src != cash_idx and state[src] >= amt:
        state[src] = state[src] - amt
    # If dst is not cash, increase its allocation by amt (clip at 100)
    if dst is not None and dst != cash_idx and state[dst] + amt <= 100:
        state[dst] = state[dst] + amt
    # If src is cash, subtract amt from cash, if dst is cash, add amt to cash.
    if src == cash_idx:
        state[cash_idx] = state[cash_idx] - amt
    if dst == cash_idx:
        state[cash_idx] = state[cash_idx] + amt
    # Optionally, enforce that the new state's sum remains 100. Here, if clipping occurred,
    # you might choose to normalize or simply allow slight deviations.
    # For now, assume actions are valid and state remains valid.
    return tuple(state)

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
    for k in range(len(assets_plus_cash)):
        if k == cash_idx:
            ratio = 1.0
        else:
            ratio = price_next[k] / price_today[k]
        growth_factor += weights_frac[k] * ratio
    # Reward is the log of the growth factor
    return math.log(growth_factor)


# ### Reward Shaping using a Rolling Sharpe Ratio

# In[38]:


class SharpeRewardShaper:
    def __init__(self, window=30, epsilon=1e-6):
        self.window = window
        self.rewards_history = []
        self.epsilon = epsilon

    def shape(self, raw_reward):
        self.rewards_history.append(raw_reward)
        if len(self.rewards_history) > self.window:
            self.rewards_history.pop(0)
        avg_reward = np.mean(self.rewards_history)
        std_reward = np.std(self.rewards_history) + self.epsilon
        sharpe = avg_reward / std_reward
        return sharpe

reward_shaper = SharpeRewardShaper(window=30)


# ### Replay Buffer for Experience Replay

# In[39]:


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


# ### Neural Network for Q-value Function

# In[40]:


class DQNNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        """
        Neural network that approximates Q(s,a) for all actions a given state s.
        state_dim: dimensionality of state input (e.g. 5)
        action_dim: number of possible actions (e.g. 21)
        """
        super(DQNNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, action_dim)

        # hidden1 = 128
        # hidden2 = 128
        # self.fc1 = nn.Linear(state_dim, hidden1)
        # self.fc2 = nn.Linear(hidden1, hidden2)
        # self.fc3 = nn.Linear(hidden2, action_dim)


    def forward(self, x):
        # x is a tensor of shape [batch_size, state_dim]

        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        x = F.relu(self.fc3(x))
        q_vals = self.fc4(x)
        return q_vals

        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # q_values = self.fc3(x)  # outputs Q-values for each action
        # return q_values


# ### DQN Agent Training Setup

# In[ ]:


# Hyperparameters
gamma = 0.99            # discount factor for future rewards
learning_rate = 5e-4  # learning rate for optimizer
epsilon_start = 1.0     # initial exploration rate
epsilon_min   = 0.2     # minimum exploration rate
epsilon_decay = 0.995    # multiplicative decay factor per episode
episodes = 150          # number of training episodes
batch_size = 128         # mini-batch size for replay updates
target_update_freq = 5 # how often (in episodes) to update the target network
replay_capacity = 10000 # capacity of the replay buffer
use_double_dqn = True  # use Double DQN for      
state_dim = num_assets + 1  # 5 assets + cash
action_dim = action_count 
ensemble_size = 2

# Initialize replay memory, policy network, target network, optimizer
replay_buffer = ReplayBuffer(replay_capacity)
# prioritized_replay_buffer = PrioritizedReplayBuffer(replay_capacity)

# policy_net = DQNNetwork(state_dim, action_dim)
# target_net = DQNNetwork(state_dim, action_dim)
# Copy initial weights to target network
# target_net.load_state_dict(policy_net.state_dict())
# target_net.eval()  # target network in evaluation mode (not strictly necessary)
# optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)


# Create ensemble of networks and their corresponding target networks
ensemble_nets = [DQNNetwork(state_dim, action_dim) for _ in range(ensemble_size)]
ensemble_targets = [DQNNetwork(state_dim, action_dim) for _ in range(ensemble_size)]
for net, target in zip(ensemble_nets, ensemble_targets):
    target.load_state_dict(net.state_dict())
    target.eval()
# Combine parameters of all ensemble networks in one optimizer
ensemble_optimizer = optim.Adam([p for net in ensemble_nets for p in net.parameters()], lr=learning_rate)

# For action selection and training, we take the average Q-values across ensemble members.
def ensemble_q_values(state_input):
    # Temporarily store the training state of each network.
    original_modes = [net.training for net in ensemble_nets]

    # Switch networks to eval mode for inference (to avoid BN issues with batch size 1)
    for net in ensemble_nets:
        net.eval()

    # Compute Q-values from each network and average them
    q_vals_list = [net(state_input) for net in ensemble_nets]  # shape: [ensemble_size, batch, action_dim]
    avg_q_vals = torch.stack(q_vals_list, dim=0).mean(dim=0)

    # Restore the original training mode of each network
    for net, mode in zip(ensemble_nets, original_modes):
        if mode:
            net.train()
        else:
            net.eval()

    return avg_q_vals

# Helper function: select action using epsilon-greedy policy
def select_action(state, epsilon):
    """
    Choose an action using epsilon-greedy strategy for the new state/action format.
    - state: current state as a tuple of 6 integers (sum to 100).
    - epsilon: current exploration rate.
    Returns: (action_idx, action_tuple)
    """
    valid_actions = get_valid_actions(state)
    if random.random() < epsilon:
        # Exploration: random valid action
        action = random.choice(valid_actions)
    else:
        # Exploitation: choose best action according to Q-network
        # Normalize state: now state values are percentages out of 100
        state_input = torch.FloatTensor([inc/100.0 for inc in state]).unsqueeze(0)
        with torch.no_grad():
          q_values = ensemble_q_values(state_input)  # shape [1, action_dim]
          q_values = q_values.numpy().squeeze()  # shape [action_dim]
        # Mask out invalid actions by setting their Q-value very low
        # (So they won't be chosen as max)
        invalid_actions = set(all_actions) - set(valid_actions)
        for act in invalid_actions:
            # if act in action_to_index:
            q_values[action_to_index[act]] = -1e9  # large negative to disable
        best_idx = int(np.argmax(q_values))
        action = all_actions[best_idx]

    # Return both the index and the tuple representation
    return action_to_index[action], action

# Training loop
# Use the new state representation: (stock1, stock2, stock3, stock4, stock5, cash)
# For instance, an equal weight in stocks (15% each) and 25% cash:
initial_state = (15, 15, 15, 15, 15, 25)
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
        weights_new = [x/100.0 for x in new_state]  # convert increments to fractions
        reward = compute_reward(weights_new, train_prices[t], train_prices[t+1])
        reward = reward_shaper.shape(reward)

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
            state_tensor = torch.FloatTensor([ [x/100.0 for x in s] for s in states_batch ])
            next_state_tensor = torch.FloatTensor([ [x/100.0 for x in s] for s in next_states_batch ])
            action_tensor = torch.LongTensor(actions_batch)
            reward_tensor = torch.FloatTensor(rewards_batch)
            done_tensor   = torch.BoolTensor(dones_batch)

            # Compute current Q values for each state-action in the batch
            # policy_net(state_tensor) has shape [batch, action_dim]; gather along actions
            q_values = ensemble_q_values(state_tensor)  # [batch, action_count]
            state_action_values = q_values.gather(1, action_tensor.unsqueeze(1)).squeeze(1)
            # Compute target Q values using target network
            with torch.no_grad():
                # next_q_values = target_net(next_state_tensor)  # [batch, action_count]
                if use_double_dqn:
                    # Step 1: For each next_state, select the best action using the online network
                    # Online ensemble selects best action:
                    online_next_q = ensemble_q_values(next_state_tensor)  # avg Q from ensemble networks
                    best_actions = online_next_q.argmax(dim=1, keepdim=True)  # best action indices from online net
                    # Step 2: Evaluate these actions using the target network
                    # For evaluation, take average target Q from target ensemble
                    q_vals_targets_list = []
                    for target_net in ensemble_targets:
                        q_vals_targets_list.append(target_net(next_state_tensor))
                    target_next_q = torch.stack(q_vals_targets_list).mean(dim=0)
                    selected_q = target_next_q.gather(1, best_actions).squeeze(1)
                else:
                    # Standard DQN target: use the max Q-value from the target network directly
                    q_vals_targets_list = []
                    for target_net in ensemble_targets:
                        q_vals_targets_list.append(target_net(next_state_tensor))
                    target_next_q = torch.stack(q_vals_targets_list).mean(dim=0)
                    selected_q = target_next_q.max(dim=1)[0]

            selected_q = selected_q * (1 - done_tensor.float())
            target_values = reward_tensor + gamma * selected_q

            # target_tensor = torch.FloatTensor(target_values)
            # losses = F.smooth_l1_loss(state_action_values, target_values, reduction='none')
            # loss = (losses * torch.FloatTensor(weights)).mean() if 'weights' in locals() else losses.mean()
            # loss = losses.mean()
            # Optimize the model: MSE loss between state_action_values and target_values
            loss = F.mse_loss(state_action_values, target_values)

            ensemble_optimizer.zero_grad()
            loss.backward()
            ensemble_optimizer.step()

    # Decay epsilon after each episode
    epsilon = max(epsilon * epsilon_decay, epsilon_min)
    # Update target network periodically
    if ep % target_update_freq == 0:
        for net, target in zip(ensemble_nets, ensemble_targets):
            target.load_state_dict(net.state_dict())
    if ep % 10 == 0 or ep == episodes:
        print(f"Episode {ep}/{episodes} completed, epsilon={epsilon:.3f}")
print("Training completed.")


# ### Policy Evaluation on Test Data

# In[46]:


def evaluate_policy(price_array, model, initial_state):
    """
    Simulate the portfolio value over time on given price data using the provided model (greedy policy).
    Returns a list of portfolio values for each day in the price data.
    """
    days = price_array.shape[0]
    state = initial_state
    portfolio_value = 1.0  # start with $1.0
    values = [portfolio_value]

    for t in range(days - 1):
        # Agent action: choose greedy (highest Q) action for current state
        state_input = torch.FloatTensor([x /100.0 for x in state]).unsqueeze(0)
        with torch.no_grad():
            ensemble_qs = model(state_input)
            q_vals = ensemble_qs.numpy().squeeze()
        # Mask invalid actions for current state
        valid_acts = get_valid_actions(state)
        invalid_acts = set(all_actions) - set(valid_acts)
        for act in invalid_acts:
            # if act in action_to_index:
            q_vals[action_to_index[act]] = -1e9
        best_act_idx = int(np.argmax(q_vals))
        best_action = all_actions[best_act_idx]
        # Rebalance portfolio according to best action
        state = apply_action(state, best_action)
        # Compute portfolio growth factor from day t to t+1 for agent
        weights = [x / 100.0 for x in state]
        # growth_factor = 0.0
        # for k in range(num_assets):
        #     growth_factor += weights[k] * (price_array[t+1][k] / price_array[t][k])
        growth_factor = sum([weights[k] * (price_array[t+1][k] / price_array[t][k]) for k in range(num_assets)]) + weights[cash_idx]
        portfolio_value *= growth_factor
        values.append(portfolio_value)
        # Update baseline value (its shares just appreciate with market, no rebalance)
        # Baseline (buy-and-hold equal weights) for comparison
        baseline_value = 1.0
        # Compute initial shares for baseline (with equal weights)
        baseline_weights = [0.2] * num_assets  # 20% each
        baseline_shares = [baseline_weights[i] * baseline_value / price_array[0][i] for i in range(num_assets)]
        baseline_portfolio_val = 0.0
        for k in range(num_assets):
            baseline_portfolio_val += baseline_shares[k] * price_array[t+1][k]
        baseline_value = baseline_portfolio_val
    final_return = (portfolio_value - 1.0) * 100.0 # in %
    baseline_return = (baseline_value - 1.0) * 100.0
    # After iterating, 'values' list contains portfolio value from start to end of period
    print(f"Test period: Agent final portfolio value = {portfolio_value:.4f} (Return = {final_return:.2f}%)")
    print(f"Test period: Baseline final value = {baseline_value:.4f} (Return = {baseline_return:.2f}%)")
    return values, baseline_weights

# Evaluate the trained model on test data
agent_values, baseline_weights = evaluate_policy(test_prices, ensemble_q_values, initial_state)


# In[ ]:





# In[ ]:




