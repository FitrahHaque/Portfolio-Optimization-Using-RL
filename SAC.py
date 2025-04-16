import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from collections import deque
import random

class ActorNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        # Output logits for each action
        return self.fc3(x)

class CriticNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(CriticNetwork, self).__init__()
        # Q1 architecture
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        
        # Q2 architecture
        self.fc4 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc5 = nn.Linear(hidden_dim, hidden_dim)
        self.fc6 = nn.Linear(hidden_dim, 1)
        
    def forward(self, state, action):
        # Concatenate state and action
        x = torch.cat([state, action], dim=1)
        
        # Q1
        q1 = F.relu(self.fc1(x))
        q1 = F.relu(self.fc2(q1))
        q1 = self.fc3(q1)
        
        # Q2
        q2 = F.relu(self.fc4(x))
        q2 = F.relu(self.fc5(q2))
        q2 = self.fc6(q2)
        
        return q1, q2
    
    def Q1(self, state, action):
        x = torch.cat([state, action], dim=1)
        q1 = F.relu(self.fc1(x))
        q1 = F.relu(self.fc2(q1))
        q1 = self.fc3(q1)
        return q1

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)

class SACAgent:
    def __init__(self, state_dim, action_dim, hidden_dim=256, 
                 learning_rate=3e-4, gamma=0.99, tau=0.005, 
                 alpha=0.2, target_entropy=None):
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        
        # Initialize networks
        self.actor = ActorNetwork(state_dim, action_dim, hidden_dim)
        self.critic = CriticNetwork(state_dim, action_dim, hidden_dim)
        self.critic_target = CriticNetwork(state_dim, action_dim, hidden_dim)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # Initialize optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=learning_rate)
        
        # Target entropy for automatic temperature tuning
        if target_entropy is None:
            self.target_entropy = -action_dim
        else:
            self.target_entropy = target_entropy
            
        # Temperature (alpha) optimizer
        self.log_alpha = torch.tensor(np.log(alpha), requires_grad=True)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=learning_rate)
        
    def select_action(self, state, valid_actions, action_to_index, all_actions):
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action_logits = self.actor(state)
            action_probs = F.softmax(action_logits, dim=1)
            
            # Mask invalid actions
            action_probs = action_probs.numpy().squeeze()
            invalid_actions = set(all_actions) - set(valid_actions)
            for act in invalid_actions:
                action_probs[action_to_index[act]] = 0.0
            
            # Normalize probabilities
            action_probs = action_probs / (np.sum(action_probs) + 1e-9)
            
            # Sample action
            action_idx = np.random.choice(len(action_probs), p=action_probs)
            return action_idx, all_actions[action_idx]
    
    def update(self, batch):
        state, action, reward, next_state, done = batch
        
        # Convert to tensors
        state = torch.FloatTensor(state)
        action = torch.LongTensor(action)
        reward = torch.FloatTensor(reward)
        next_state = torch.FloatTensor(next_state)
        done = torch.FloatTensor(done)
        
        # Update critic
        with torch.no_grad():
            # Get next state action probabilities
            next_action_logits = self.actor(next_state)
            next_action_probs = F.softmax(next_action_logits, dim=1)
            
            # Compute target Q values
            next_q1, next_q2 = self.critic_target(next_state, next_action_probs)
            next_q = torch.min(next_q1, next_q2)
            target_q = reward + (1 - done) * self.gamma * next_q
        
        # Get current Q values
        current_q1, current_q2 = self.critic(state, F.one_hot(action, num_classes=len(all_actions)).float())
        
        # Compute critic loss
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        
        # Optimize critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Update actor
        action_logits = self.actor(state)
        action_probs = F.softmax(action_logits, dim=1)
        log_probs = F.log_softmax(action_logits, dim=1)
        
        # Compute Q values for current state and actions
        q1, q2 = self.critic(state, action_probs)
        q = torch.min(q1, q2)
        
        # Compute actor loss
        actor_loss = (self.alpha * log_probs - q).mean()
        
        # Optimize actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Update temperature
        alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
        
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        
        self.alpha = self.log_alpha.exp()
        
        # Update target networks
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data) 

def train_sac(env, agent, replay_buffer, num_episodes=1000, batch_size=256, 
              warmup_steps=10000, update_interval=1):
    """Train the SAC agent"""
    total_steps = 0
    episode_rewards = []
    
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            # Select action
            if total_steps < warmup_steps:
                # Random action during warmup
                valid_actions = env.get_valid_actions(state)
                action_idx = np.random.randint(len(valid_actions))
                action = valid_actions[action_idx]
            else:
                action_idx, action = agent.select_action(
                    state, 
                    env.get_valid_actions(state),
                    env.action_to_index,
                    env.all_actions
                )
            
            # Take action
            next_state, reward, done, _ = env.step(action)
            
            # Store transition
            replay_buffer.push(state, action_idx, reward, next_state, done)
            
            # Update agent
            if len(replay_buffer) > batch_size and total_steps % update_interval == 0:
                batch = replay_buffer.sample(batch_size)
                agent.update(batch)
            
            state = next_state
            episode_reward += reward
            total_steps += 1
        
        episode_rewards.append(episode_reward)
        print(f"Episode {episode + 1}/{num_episodes}, Reward: {episode_reward:.2f}")
    
    return episode_rewards

def evaluate_sac(env, agent, num_episodes=10):
    """Evaluate the trained SAC agent"""
    episode_rewards = []
    
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            _, action = agent.select_action(
                state,
                env.get_valid_actions(state),
                env.action_to_index,
                env.all_actions
            )
            
            next_state, reward, done, _ = env.step(action)
            state = next_state
            episode_reward += reward
        
        episode_rewards.append(episode_reward)
        print(f"Evaluation Episode {episode + 1}/{num_episodes}, Reward: {episode_reward:.2f}")
    
    return np.mean(episode_rewards), np.std(episode_rewards)

if __name__ == "__main__":
    # Environment parameters
    num_assets = 5
    state_dim = num_assets  # Current allocations
    action_dim = 21  # Number of possible actions (including no-action)
    
    # Create environment (using your existing environment)
    env = PortfolioEnv()  # You'll need to adapt your existing environment
    
    # SAC parameters
    hidden_dim = 256
    learning_rate = 3e-4
    gamma = 0.99
    tau = 0.005
    alpha = 0.2
    
    # Create agent and replay buffer
    agent = SACAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        learning_rate=learning_rate,
        gamma=gamma,
        tau=tau,
        alpha=alpha
    )
    
    replay_buffer = ReplayBuffer(capacity=1000000)
    
    # Training parameters
    num_episodes = 1000
    batch_size = 256
    warmup_steps = 10000
    update_interval = 1
    
    # Train the agent
    print("Starting training...")
    episode_rewards = train_sac(
        env, agent, replay_buffer,
        num_episodes=num_episodes,
        batch_size=batch_size,
        warmup_steps=warmup_steps,
        update_interval=update_interval
    )
    
    # Evaluate the trained agent
    print("\nEvaluating trained agent...")
    mean_reward, std_reward = evaluate_sac(env, agent)
    print(f"Mean reward: {mean_reward:.2f} ± {std_reward:.2f}") 