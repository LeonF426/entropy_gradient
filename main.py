import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random

# Actor Model
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, action_dim)
        self.max_action = max_action
        
    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        action = torch.tanh(self.fc3(x)) * self.max_action  # Action scaled between [-1, 1]
        return action

# Critic Model
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, 1)
        
    def forward(self, state, action):
        x = torch.cat([state, action], 1)  # Concatenate state and action
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        q_value = self.fc3(x)
        return q_value

# DDPG Agent
class DDPGAgent:
    def __init__(self, state_dim, action_dim, max_action):
        self.actor = Actor(state_dim, action_dim, max_action).cuda()
        self.critic = Critic(state_dim, action_dim).cuda()
        self.target_actor = Actor(state_dim, action_dim, max_action).cuda()
        self.target_critic = Critic(state_dim, action_dim).cuda()

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=0.0001)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=0.001)

        self.replay_buffer = deque(maxlen=100000)
        self.max_action = max_action
        self.gamma = 0.99  # Discount factor
        self.tau = 0.005   # For soft update of target network

    # Store experience in replay buffer
    def store_transition(self, transition):
        self.replay_buffer.append(transition)

    # Sample a batch from the replay buffer
    def sample_batch(self, batch_size):
        batch = random.sample(self.replay_buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return np.array(state), np.array(action), np.array(reward), np.array(next_state), np.array(done)

    # Update Actor and Critic networks
    def update(self, batch_size):
        if len(self.replay_buffer) < batch_size:
            return
        
        state, action, reward, next_state, done = self.sample_batch(batch_size)

        state = torch.FloatTensor(state).cuda()
        action = torch.FloatTensor(action).cuda()
        reward = torch.FloatTensor(reward).unsqueeze(1).cuda()
        next_state = torch.FloatTensor(next_state).cuda()
        done = torch.FloatTensor(done).unsqueeze(1).cuda()

        # Update Critic Network
        with torch.no_grad():
            next_action = self.target_actor(next_state)
            target_q_value = self.target_critic(next_state, next_action)
            target_q_value = reward + (1 - done) * self.gamma * target_q_value

        current_q_value = self.critic(state, action)
        critic_loss = nn.MSELoss()(current_q_value, target_q_value)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update Actor Network
        actor_loss = -self.critic(state, self.actor(state)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Soft update of target networks
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

# Train the DDPG Agent in the environment
def train_ddpg(env, agent, num_episodes=1000, batch_size=64):
    max_action = float(env.action_space.high[0])
    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        while not done:
            # Select action with added noise for exploration
            action = agent.actor(torch.FloatTensor(state).cuda()).cpu().detach().numpy()
            action = action + np.random.normal(0, 0.1, size=env.action_space.shape[0])
            action = np.clip(action, env.action_space.low, env.action_space.high)
            
            next_state, reward, done, _ = env.step(action)
            agent.store_transition((state, action, reward, next_state, done))
            state = next_state
            episode_reward += reward
            
            agent.update(batch_size)
        
        print(f"Episode {episode+1}, Reward: {episode_reward}")
    
    env.close()


if __name__ == "__main__":
    # Load environment
    env = gym.make("LunarLanderContinuous-v2")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    # Initialize agent
    agent = DDPGAgent(state_dim, action_dim, max_action)
    
    # Train the agent
    #train_ddpg(env, agent)
    print(agent.actor)
