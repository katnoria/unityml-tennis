import os
import random
import logging
import numpy as np
# pytorch
import torch
import torch.nn.functional as F
import torch.optim as optim
# repo code
from models import Actor, CentralCritic, Critic
from utils import save_to_txt, ReplayBuffer, OUNoise, flatten

# Hyper parameters
config = {
    "USE_BATCHNORM": False,      # whether to use batch norm (paper used it to learn across many different games)
    "BUFFER_SIZE": int(1e6),     # replay buffer size
    "BATCH_SIZE": 512,           # minibatch size
    "GAMMA": 0.9,               # discount factor
    # "TAU": 1e-3,               # for soft update of target parameters
    "TAU": 1e-2,                 # for soft update of target parameters
    "LR_ACTOR": 1e-3,            # learning rate of the actor 
    "LR_CRITIC": 1e-3,           # learning rate of the critic
    "WEIGHT_DECAY": 0,           # L2 weight decay
    "SIGMA": 0.1,
    "LEARN_STEP": 1,           # how often to learn
    "FC1": 64,
    "FC2": 64,
}

# Create logger
logger = logging.getLogger(__name__)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logger.info('Using device: {}'.format(device))

class MADDPGAgent():
    """Multi Agent DDPG Implementation"""
    def __init__(self, state_size, action_size, num_agents, agent_index, writer, random_seed, dirname, print_every=1000, model_path=None, eval_mode=False):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)
        self.agent_index = agent_index
        self.writer = writer
        self.dirname = dirname
        self.print_every = print_every
        # Create Critic network
        self.local_critic = Critic(self.state_size * num_agents, self.action_size * num_agents, random_seed, fc1_units=config['FC1'], fc2_units=config['FC2']).to(device)
        self.target_critic = Critic(self.state_size * num_agents, self.action_size * num_agents, random_seed, fc1_units=config['FC1'], fc2_units=config['FC2']).to(device)
        self.critic_optimizer = optim.Adam(self.local_critic.parameters(), lr=config['LR_CRITIC'], weight_decay=config['WEIGHT_DECAY'])
        # Create Actor network
        self.local_actor = Actor(self.state_size, self.action_size, random_seed, fc1_units=config['FC1'], fc2_units=config['FC2']).to(device)
        self.target_actor = Actor(self.state_size, self.action_size, random_seed, fc1_units=config['FC1'], fc2_units=config['FC2']).to(device)
        self.actor_optimizer = optim.Adam(self.local_actor.parameters(), lr=config['LR_ACTOR'])

        self.noise = OUNoise(self.action_size, random_seed, sigma=config['SIGMA'])
        self.learn_step = 0

    def act(self, state, add_noise=True):
        state = torch.from_numpy(state).float().to(device)
        # Run inference in eval mode
        self.local_actor.eval()
        with torch.no_grad():
            action = self.local_actor(state).cpu().data.numpy()
        self.local_actor.train()
        # add noise if true
        if add_noise:
            action += self.noise.sample()
        return np.clip(action, -1, 1)

    def reset(self):
        self.noise.reset()

    def learn(self, agents, experience, gamma):
        num_agents = len(agents)
        states, actions, rewards, next_states, dones = experience
        # ---------------central critic-------------------
        # use target actor to get action        
        next_actions = torch.zeros((len(states), num_agents, self.action_size)).to(device)
        for i, agent in enumerate(agents):            
            next_actions[:, i] = agent.target_actor(states[:, i, :])
            # if i == self.agent_index:
            #     next_actions[:, i] = agent.target_actor(states[:, i, :]) 
            # else:
            #     agent.target_actor.eval()
            #     with torch.no_grad():
            #         next_actions[:, i] = agent.target_actor(states[:, i, :]).cpu().data
            #     agent.target_actor.train()
         
        next_actions = flatten(next_actions)
        critic_states = flatten(next_states)

        # calculate target and expected
        Q_targets_next = self.target_critic(critic_states, next_actions)
        Q_targets = rewards[:, self.agent_index, :] + (gamma * Q_targets_next * (1 - dones[:, self.agent_index, :]))
        Q_expected = self.local_critic(flatten(states), flatten(actions))

        # use mse
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        critic_loss_value = critic_loss.item()
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        for param in self.local_critic.parameters():
            # import pdb; pdb.set_trace()
            param.grad.data.clamp_(-0.5, 0.5)
        self.critic_optimizer.step()

        # ---------------actor---------------------
        # Just update the predicted action of current agent
        # predicted_actions = actions.clone()
        predicted_actions = torch.zeros((len(states), num_agents, self.action_size)).to(device)
        predicted_actions.data.copy_(actions.data)
        predicted_actions[:, self.agent_index] = self.local_actor(states[:, self.agent_index])
        # predicted_actions[:, self.agent_index] = self.local_actor(states[:, self.agent_index])
        actor_loss = -self.local_critic(flatten(states), flatten(predicted_actions)).mean()
        # Kept to remind myself about the mistake that several tooks hours of investigation
        # and was only found when I looked at grads from self.local_actor.parameters()
        # actor_loss = -self.local_critic(flatten(states), flatten(actions)).mean()

        # What if we use the predicted action from all agents, will that confuse
        # the critic 
        # predicted_actions = torch.zeros((len(states), num_agents, self.action_size)).to(device)
        # for i, agent in enumerate(agents):
        #     predicted_actions[:, i] = agent.local_actor(states[:, i]) 
        # actor_loss = -self.local_critic(flatten(states), flatten(predicted_actions)).mean()

        actor_loss_value = actor_loss.item()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        for param in self.local_actor.parameters():
            # import pdb; pdb.set_trace()
            param.grad.data.clamp_(-0.5, 0.5)
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        if self.learn_step == 0:
            # One time only, start local and target with same parameters
            self._copy_weights(self.local_critic, self.target_critic)
            self._copy_weights(self.local_actor, self.target_actor)
        else:
            self.soft_update(self.local_critic, self.target_critic, config["TAU"])
            self.soft_update(self.local_actor, self.target_actor, config["TAU"])

        self.learn_step += 1
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
        torch.save(self.local_critic.state_dict(), '{}/multi/checkpoint_critic_{}.pth'.format(self.dirname, self.agent_index)) 
        torch.save(self.local_actor.state_dict(), '{}/multi/checkpoint_actor_{}.pth'.format(self.dirname, self.agent_index))


class MADDPGAgentTrainer(): 
    def __init__(self, state_size, action_size, num_agents, writer, random_seed, dirname, print_every=1000, model_path=None, eval_mode=False):
        self.state_size = state_size
        self.action_size = action_size
        self.num_agents = num_agents
        self.writer = writer
        self.print_every = print_every
        self.dirname = dirname
        self.agents = [
            MADDPGAgent(state_size, action_size, self.num_agents, agent_index=i, writer=self.writer, random_seed=random_seed, dirname=dirname, print_every=print_every, model_path=model_path) 
            for i in range(self.num_agents)]
        self.memory = ReplayBuffer(self.action_size, config['BUFFER_SIZE'], config['BATCH_SIZE'], random_seed)
        self.learn_step = 0        

    def act(self, states, add_noise=True):
        actions = []
        for i, agent in enumerate(self.agents):
            action = agent.act(states[i], add_noise)
            actions.append(action)
        return actions

    def step(self, states, actions, rewards, next_states, dones):
        # import pdb; pdb.set_trace()
        self.learn_step += 1
        states = np.expand_dims(states, 0)
        actions = np.expand_dims(np.array(actions).reshape(self.num_agents, self.action_size),0)
        rewards = np.expand_dims(np.array(rewards).reshape(self.num_agents, -1),0)
        dones = np.expand_dims(np.array(dones).reshape(self.num_agents, -1),0)
        next_states = np.expand_dims(np.array(next_states).reshape(self.num_agents, -1), 0)
        # import pdb; pdb.set_trace()
        self.memory.add(states, actions, rewards, next_states, dones)
        if len(self.memory) < config['BATCH_SIZE']:
            return
        if not self.learn_step % config['LEARN_STEP'] == 0:
            return

        experiences = self.memory.sample()
        actor_losses = []
        critic_losses = []
        for agent in self.agents:
            actor_loss, critic_loss = agent.learn(self.agents, experiences, config['GAMMA'])
            actor_losses.append(actor_loss)
            critic_losses.append(critic_loss)
        
        if self.learn_step % self.print_every == 0:
            # Save Critic loss
            save_to_txt(critic_losses, '{}/critic_losses.txt'.format(self.dirname))
            self.writer.text('critic loss: {}'.format(critic_losses), "Critic")
            self.writer.push(critic_losses, "Loss(critic)")
            # Save Actor loss
            save_to_txt(actor_losses, '{}/actor_losses.txt'.format(self.dirname))
            self.writer.text('actor loss: {}'.format(actor_losses), "Actor")
            self.writer.push(actor_losses, "Loss(actor)")

    def reset(self):
        for agent in self.agents:
            agent.reset()

    def checkpoint(self):
        """Checkpoint actor and critic models"""
        for agent in self.agents:
            agent.checkpoint()
