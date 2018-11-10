import numpy as np
import random
import copy
from collections import namedtuple, deque, defaultdict
import os
import logging
import json
from datetime import datetime
# Pytorch
import torch
import torch.nn.functional as F
import torch.optim as optim
# Code
from models import Actor, Critic
from utils import OUNoise, ReplayBuffer
from utils import save_to_json, save_to_txt

# Hyper parameters
config = {
    "USE_BATCHNORM": False,      # whether to use batch norm (paper used it to learn across many different games)
    "BUFFER_SIZE": int(2e6),     # replay buffer size
    "BATCH_SIZE": 256,           # minibatch size
    "GAMMA": 0.9,                # discount factor
    "TAU": 1e-3,                 # for soft update of target parameters
    "LR_ACTOR": 1e-3,            # learning rate of the actor 
    "LR_CRITIC": 1e-3,           # learning rate of the critic
    "WEIGHT_DECAY": 0,           # L2 weight decay
    # "SCALE_REWARD": 0.1,       # scaling factor applied to rewards (http://arxiv.org/abs/1604.06778)
    "SCALE_REWARD": 1.0,         # default scaling factor
    "SIGMA": 0.01,
    "FC1": 32,
    "FC2": 16,
}

# Create logger
logger = logging.getLogger("ddpg_multi")
# Device setup
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logger.debug('Device Info:{}'.format(device))


class DDPGMultiAgent():
    """Interacts with and learns from the environment."""
    
    def __init__(self, state_size, action_size, num_agents, writer, random_seed, dirname, print_every=100, model_path=None, eval_mode=False):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            num_agents (int): number of agents
            writer (object): visdom visualiser for realtime visualisations            
            random_seed (int): random seed
            dirname (string): output directory to store config, losses
            print_every (int): how often to print progress
            model_path (string): if defined, load saved model to resume training
        """
        self.state_size = state_size
        self.action_size = action_size
        self.num_agents = num_agents
        self.seed = random.seed(random_seed)  
        self.dirname = dirname
        self.print_every = print_every      

        # save config params
        save_to_json(config, '{}/hyperparams.json'.format(self.dirname))


        # Actor Networks (w/ Target Networks)
        self.actor_local = [
            Actor(state_size, action_size, random_seed, fc1_units=config['FC1'], fc2_units=config['FC2'], use_bn=config["USE_BATCHNORM"]).to(device) 
            for _ in range(num_agents)
        ]
        
        self.actor_target = [
            Actor(state_size, action_size, random_seed, fc1_units=config['FC1'], fc2_units=config['FC2'], use_bn=config["USE_BATCHNORM"]).to(device) 
            for _ in range(num_agents)
        ]

        self.actor_optimizer = [
            optim.Adam(self.actor_local[i].parameters(), lr=config["LR_ACTOR"]) 
            for i in range(num_agents)
        ]

        # Critic Networks (w/ Target Networks)
        self.critic_local = [
            Critic(state_size, action_size, random_seed, fc1_units=config['FC1'], fc2_units=config['FC2'], use_bn=config["USE_BATCHNORM"]).to(device)
            for _ in range(num_agents)
            ]
        self.critic_target = [
            Critic(state_size, action_size, random_seed, fc1_units=config['FC1'], fc2_units=config['FC2'], use_bn=config["USE_BATCHNORM"]).to(device)
            for _ in range(num_agents)
            ]
        self.critic_optimizer = [
            optim.Adam(self.critic_local[i].parameters(), lr=config["LR_CRITIC"], weight_decay=config["WEIGHT_DECAY"])
            for i in range(num_agents)
            ]

        # Load saved model (if available)
        if model_path:
            logger.info('Loading model from {}'.format(model_path))
            for i in range(self.num_agents):                 
                self.actor_local[i].load_state_dict(torch.load('{}/checkpoint_actor_{}.pth'.format(model_path, i)))
                self.actor_target[i].load_state_dict(torch.load('{}/checkpoint_actor_{}.pth'.format(model_path, i)))
                self.critic_local[i].load_state_dict(torch.load('{}/checkpoint_critic_{}.pth'.format(model_path, i)))
                self.critic_target[i].load_state_dict(torch.load('{}/checkpoint_critic_{}.pth'.format(model_path, i)))
                if eval_mode:
                    logger.info('agent {} set to eval mode')
                    self.actor_local[i].eval()
    

        # Noise process
        self.noise = [
            OUNoise(action_size, random_seed, sigma=config['SIGMA'])
            for _ in range(num_agents)
            ]

        # Replay memory
        self.memory = ReplayBuffer(action_size, config["BUFFER_SIZE"], config["BATCH_SIZE"], random_seed)

        # Record losses
        self.actor_losses = []
        self.critic_losses = []
        self.learn_count = []
        self.learn_step = 0
        # Initialise visdom writer
        self.writer = writer
        logger.info("Initialised with random seed: {}".format(random_seed))
    
    def step(self, states, actions, rewards, next_states, dones):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        # self.memory.add(state, action, reward, next_state, done)                
        for i in range(self.num_agents):
            state = states[i]
            action = actions[i]
            reward = rewards[i]
            next_state = next_states[i]
            done = dones[i]
            scaled_rewards = config['SCALE_REWARD'] * reward
            self.memory.add(state, action, scaled_rewards, next_state, done)

        # Learn, if enough samples are available in memory
        if len(self.memory) > config["BATCH_SIZE"]:
            # each agent should get different sample
            # experiences = self.memory.sample()            
            self.learn(config["GAMMA"])

    def act(self, states, add_noise=True):
        """Returns actions for given states as per policies."""
        actions = []
        for i in range(self.num_agents):
            actions.append(self._act(self.actor_local[i], states[i], add_noise, self.noise[i]))
        return actions

    def _act(self, actor_local, state, add_noise, noise):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(np.expand_dims(state, 0)).float().to(device)
        actor_local.eval()
        with torch.no_grad():
            action = actor_local(state).cpu().data.numpy()
        actor_local.train()
        if add_noise:
            action += noise.sample()
        return np.clip(action, -1, 1)

    def reset(self):
        """Resets the noise"""
        for i in range(self.num_agents):
            self.noise[i].reset()

    def learn(self, gamma):
        """Learn from experiences"""
        actor_losses = []
        critic_losses = []        
        self.learn_step += 1
        for i in range(self.num_agents):
            experiences = self.memory.sample()
            actor_loss, critic_loss = self._learn(experiences, gamma, 
                self.actor_local[i], self.actor_target[i],
                self.critic_local[i], self.critic_target[i],
                self.actor_optimizer[i], self.critic_optimizer[i]
                )
            actor_losses.append(actor_loss)
            critic_losses.append(critic_loss)

        if self.learn_step % self.print_every == 0:
            self.writer.text('critic loss: {}'.format(np.mean(critic_losses)), "Critic Multi Agent")
            save_to_txt(np.mean(critic_losses), '{}/critic_losses_multi.txt'.format(self.dirname))
            self.writer.push(np.mean(critic_losses), "Loss(critic)")
            self.writer.text('actor loss: {}'.format(np.mean(actor_losses)), "Actor Multi Agent")
            save_to_txt(np.mean(actor_losses), '{}/actor_losses_multi.txt'.format(self.dirname))
            self.writer.push(np.mean(actor_losses), "Loss(actor)")
            


    def _learn(self, experiences, gamma, actor_local, actor_target, critic_local, critic_target, actor_optimizer, critic_optimizer):
        """Update policy and value parameters using given batch of experience tuples.
            Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
            actor_local (object): instance of local actor
            actor_target (object): instance of target actor
            critic_local (object): instance of local critic
            critic_target (object): instance of target critic
            actor_optimizer (object): instance of actor optimizer
            critic_optimizer (object): instance of critic optimizer
        """
        states, actions, rewards, next_states, dones = experiences
        critic_loss_value = None
        actor_loss_value = None

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = actor_target(next_states)
        Q_targets_next = critic_target(next_states, actions_next)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected = critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        critic_loss_value = critic_loss.item()
        # Minimize the loss
        critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = actor_local(states)
        actor_loss = -critic_local(states, actions_pred).mean()
        actor_loss_value = actor_loss.item()

        # Minimize the loss
        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        if self.learn_step == 1:
            # One time only, start local and target with same parameters
            self._copy_weights(critic_local, critic_target)
            self._copy_weights(actor_local, actor_target)
        else:
            self.soft_update(critic_local, critic_target, config["TAU"])
            self.soft_update(actor_local, actor_target, config["TAU"])        

        return actor_loss_value, critic_loss_value

    def _copy_weights(self, source_network, target_network):
        """Copy source network weights to target"""
        for target_param, source_param in zip(target_network.parameters(), source_network.parameters()):
            target_param.data.copy_(source_param.data)

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


    def checkpoint(self):
        """Checkpoint actor and critic models"""
        if not os.path.exists('{}/multi'.format(self.dirname)):
            os.makedirs('{}/multi'.format(self.dirname))
        for i in range(self.num_agents):
            torch.save(self.actor_local[i].state_dict(), '{}/multi/checkpoint_actor_{}.pth'.format(self.dirname, i))
            torch.save(self.critic_local[i].state_dict(), '{}/multi/checkpoint_critic_{}.pth'.format(self.dirname, i))