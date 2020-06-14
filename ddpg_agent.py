import numpy as np
import random
import copy
from collections import namedtuple, deque
import time

from model import Actor, Critic
# from colorama import Fore, Back, Style 

import torch
import torch.nn.functional as F
import torch.optim as optim

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = "cpu"
# Conversion from numpy to tensor
def ten(x): return torch.from_numpy(x).float().to(device)
    
class ddpg_Agent():
    """Interacts with and learns from the environment."""
    
    def __init__(self, env, config):
        """Initialize an Agent object.
        
        Params
        ======
            env : environment to be handled
            config : configuration given a variety of parameters
        """
        self.env = env
        self.config = config

        # set parameter for ML
        self.set_parameters(config)
        # Q-Network
        self.create_networks()
        # Noise process
        self.noise = OUNoise(self.action_size, self.seed)
        # Replay memory
        self.memory = ReplayBuffer(self.action_size, self.buffer_size, self.batch_size, self.seed)
    
    def set_parameters(self, config):
        # Base agent parameters
        self.gamma = config['gamma']                    # discount factor 
        self.tau = config['tau']
        self.max_episodes = config['max_episodes']      # max numbers of episdoes to train
        self.env_file_name = config['env_file_name']    # name and path for env app
        self.brain_name = config['brain_name']          # name for env brain used in step
        self.num_agents = config['num_agents']
        self.state_size = config['state_size']
        self.action_size = config['action_size']
        self.hidden_size = config['hidden_size']
        self.buffer_size = config['buffer_size']
        self.batch_size = config['batch_size']
        self.dropout = config['dropout']
        self.critic_learning_rate = config['critic_learning_rate']
        self.actor_learning_rate = config['actor_learning_rate']
        self.seed = (config['seed'])
        # Some debug flags
        self.DoDebugEpisodeLists = False        
        
    def create_networks(self):
        # Actor Network (local & Target Network)
        self.actor_local = Actor(self.state_size, self.hidden_size, self.action_size, self.seed).to(device)
        self.actor_target = Actor(self.state_size, self.hidden_size, self.action_size, self.seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=self.actor_learning_rate)

        # Critic Network (local & Target Network)
        self.critic_local = Critic(self.state_size, self.hidden_size, self.action_size, self.seed, self.dropout).to(device)
        self.critic_target = Critic(self.state_size, self.hidden_size, self.action_size, self.seed, self.dropout).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=self.critic_learning_rate)

    def step(self, state, action, reward, next_state, done):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        # print('step : Next States : ',next_state.shape)
        self.memory.add(state, action, reward, next_state, done)
        # print('New step added to memory, length : ',len(self.memory))

        # Learn, if enough samples are available in memory
        if len(self.memory) > self.batch_size:
            experiences = self.memory.sample()
            self.learn(experiences, self.gamma)

    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy."""
        state = ten(state)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            action += self.noise.sample()
            # ToDo check if tanh works better
        return np.clip(action, -1, 1)

    def train(self):
        all_rewards = []
        reward_window = deque(maxlen=100)
        print('Running on device : ',device)
        for i_episode in range(self.max_episodes): 
            tic = time.time()
            # Reset the enviroment
            env_info = self.env.reset(train_mode=True)[self.brain_name]
            state = env_info.vector_observations
            total_reward = np.zeros(self.num_agents)
            t = 0
            done = np.zeros(self.num_agents, dtype = bool)

            # loop over episode time steps
            while all(done==False): #  t < self.tmax:
                # act and collect data
                action = self.act(state)
                env_info = self.env.step(action)[self.brain_name]
                next_state = env_info.vector_observations
                reward = np.asarray(env_info.rewards)
                if False:
                    if sum(reward) > 0.001:
                        print('-----------------------------------------------------')
                        print('-----------------------------------------------------')
                        print('-----------------------------------------------------')
                        print(env_info.rewards)
                    print(reward)
                    print(done)
                    print(type(next_state))
                done = np.asarray(env_info.local_done)
                np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
                # print('Episode {} step {} taken action {} reward {} and done is {}'.format(i_episode,t,action,reward,done))
                # increment stuff
                t += 1
                total_reward += reward
                # Proceed agent step
                self.step(state, action, reward, next_state, done)
                # prepare for next round
                state = next_state
            # while not done
            # keep track of rewards:
            all_rewards.append(np.mean(total_reward))
            reward_window.append(np.mean(total_reward))
            
            # Output Episode info : 
            #if total_reward < 0.001:
            toc = time.time()
            print('Episode {} || Total Reward : {:6.3f} || average reward : {:6.3f} || Used {:5.3f} seconds'.format(i_episode,np.mean(total_reward),np.mean(reward_window),toc-tic))
            # else:
                # print(Back.RED + 'Episode {} || Total Reward : {:6.3f} || average reward : {:6.3f}'.format(i_episode,total_reward,np.mean(reward_window)))
                # print(Style.RESET_ALL)
            
        return all_rewards

    def reset(self):
        self.noise.reset()

    def learn(self, experiences, gamma):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            self.gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        # print('learn : Next States : ',next_states.shape)
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        # print('learn : Actions : ',actions_next.shape)
        # print('learn : Q_target_next : ',Q_targets_next.shape)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, self.tau)
        self.soft_update(self.actor_local, self.actor_target, self.tau)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            self.batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        # print('Adding experiences to memory : ')
        # print('State shape : ',state.shape)
        # print(type(state))
        for a in np.arange(state.shape[0]):
            e = self.experience(state[a,], action[a,], reward[a,], next_state[a,], done[a,])
            self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        # print('Sample : state : ',experiences[0].state.shape)
        # print('Sample : next_state : ',experiences[0].next_state.shape)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
        
        # print('Sample : states : ',states.shape)
        # print('Sample : next_states : ',next_states.shape)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)