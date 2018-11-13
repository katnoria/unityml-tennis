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
from utils import save_to_txt, save_to_json, flatten
from utils import ReplayBuffer, OUNoise

# Hyper parameters
config = {
    "USE_BATCHNORM": False,      # whether to use batch norm (paper used it to learn across many different games)
    "BUFFER_SIZE": int(1e6),     # replay buffer size
    "BATCH_SIZE": 512,           # minibatch size
    "GAMMA": 0.95,               # discount factor
    "TAU": 1e-2,                 # for soft update of target parameters
    "LR_ACTOR": 1e-3,            # learning rate of the actor 
    "LR_CRITIC": 1e-3,           # learning rate of the critic
    "WEIGHT_DECAY": 0,           # L2 weight decay
    "SIGMA": 0.1,                # std of noise 
    "LEARN_STEP": 1,             # how often to learn (paper updates every 100 steps but 1 worked best here)
    "CLIP_GRADS": True,          # Whether to clip gradients
    "CLAMP_VALUE": 1,            # Clip value
    "FC1": 64,                   # First linear layer size
    "FC2": 64,                   # Second linear layer size
}

# Create logger
logger = logging.getLogger(__name__)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logger.info('Using device: {}'.format(device))

class MADDPGAgent():
    """
    Multi Agent DDPG Implementation
    Paper: https://arxiv.org/abs/1706.02275
    I used their code to understand how the agents were implemented https://github.com/openai/maddpg
    """
    def __init__(self, state_size, action_size, num_agents, agent_index, writer, random_seed, dirname, print_every=1000, model_path=None, eval_mode=False):
        """Initialize an Agent object.
        
        Parameters:    
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            num_agents (int): number of agents
            agent_index (int): index (id) of current agent
            writer (object): visdom visualiser for realtime visualisations            
            random_seed (int): random seed
            dirname (string): output directory to store config, losses
            print_every (int): how often to print progress
            model_path (string): if defined, load saved model to resume training
            eval_mode (bool): whether to use eval mode
        """        
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)
        self.agent_index = agent_index
        self.writer = writer
        self.dirname = dirname
        self.print_every = print_every
        # save config params
        save_to_json(config, '{}/hyperparams.json'.format(self.dirname))

        # Create Critic network
        self.local_critic = Critic(self.state_size * num_agents, self.action_size * num_agents, random_seed, fc1_units=config['FC1'], fc2_units=config['FC2']).to(device)
        self.target_critic = Critic(self.state_size * num_agents, self.action_size * num_agents, random_seed, fc1_units=config['FC1'], fc2_units=config['FC2']).to(device)
        self.critic_optimizer = optim.Adam(self.local_critic.parameters(), lr=config['LR_CRITIC'], weight_decay=config['WEIGHT_DECAY'])
        # Create Actor network
        self.local_actor = Actor(self.state_size, self.action_size, random_seed, fc1_units=config['FC1'], fc2_units=config['FC2']).to(device)
        self.target_actor = Actor(self.state_size, self.action_size, random_seed, fc1_units=config['FC1'], fc2_units=config['FC2']).to(device)
        self.actor_optimizer = optim.Adam(self.local_actor.parameters(), lr=config['LR_ACTOR'])

        # Load saved model (if available)
        if model_path:
            logger.info('Loading model from {}'.format(model_path))
            self.local_actor.load_state_dict(torch.load('{}/checkpoint_actor_{}.pth'.format(model_path, self.agent_index)))
            self.target_actor.load_state_dict(torch.load('{}/checkpoint_actor_{}.pth'.format(model_path, self.agent_index)))
            self.local_critic.load_state_dict(torch.load('{}/checkpoint_critic_{}.pth'.format(model_path, self.agent_index)))
            self.target_critic.load_state_dict(torch.load('{}/checkpoint_critic_{}.pth'.format(model_path, self.agent_index)))
            if eval_mode:
                logger.info('agent {} set to eval mode')
                self.actor_local.eval()
    
        self.noise = OUNoise(self.action_size, random_seed, sigma=config['SIGMA'])
        self.learn_step = 0

    def act(self, state, add_noise=True, noise_weight=1):
        """Get the actions to take under the supplied states

        Parameters:
            state (array_like): Game state provided by the environment
            add_noise (bool): Whether we should apply the noise
            noise_weight (int): How much weight should be applied to the noise
        """
        state = torch.from_numpy(state).float().to(device)
        # Run inference in eval mode
        self.local_actor.eval()
        with torch.no_grad():
            action = self.local_actor(state).cpu().data.numpy()
        self.local_actor.train()
        # add noise if true
        if add_noise:
            action += self.noise.sample() * noise_weight
        return np.clip(action, -1, 1)

    def reset(self):
        """Resets the noise"""
        self.noise.reset()

    def learn(self, agents, experience, gamma):
        """Use the experience to allow agents to learn. 
        The critic of each agent can see the actions taken by all agents 
        and incorporate that in the learning.

        Parameters:
            agents (MADDPGAgent): instance of all the agents
            experience (Tuple[torch.Tensor]):  tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        num_agents = len(agents)
        states, actions, rewards, next_states, dones = experience
        # ---------------central critic-------------------
        # use target actor to get action, here we get target actors from 
        # all agents to predict the next action
        next_actions = torch.zeros((len(states), num_agents, self.action_size)).to(device)
        for i, agent in enumerate(agents):            
            next_actions[:, i] = agent.target_actor(states[:, i, :])
        
        # Flatten state and action
        # e.g from state (100,2,24) --> (100, 48)
        critic_states = flatten(next_states)
        next_actions = flatten(next_actions)

        # calculate target and expected
        Q_targets_next = self.target_critic(critic_states, next_actions)
        Q_targets = rewards[:, self.agent_index, :] + (gamma * Q_targets_next * (1 - dones[:, self.agent_index, :]))
        Q_expected = self.local_critic(flatten(states), flatten(actions))

        # use mse loss 
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        critic_loss_value = critic_loss.item()
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        if config['CLIP_GRADS']:
            for param in self.local_critic.parameters():
                param.grad.data.clamp_(-1 * config['CLAMP_VALUE'], config['CLAMP_VALUE'])
        self.critic_optimizer.step()

        # ---------------actor---------------------
        # Only update the predicted action of current agent
        predicted_actions = torch.zeros((len(states), num_agents, self.action_size)).to(device)
        predicted_actions.data.copy_(actions.data)
        predicted_actions[:, self.agent_index] = self.local_actor(states[:, self.agent_index])
        actor_loss = -self.local_critic(flatten(states), flatten(predicted_actions)).mean()
        # Kept to remind myself about the mistake that several tooks hours of investigation
        # and was only found when I looked at grads from self.local_actor.parameters()
        # actor_loss = -self.local_critic(flatten(states), flatten(actions)).mean()

        actor_loss_value = actor_loss.item()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        if config['CLIP_GRADS']:
            for param in self.local_actor.parameters():
                # import pdb; pdb.set_trace()
                param.grad.data.clamp_(-1 * config['CLAMP_VALUE'], config['CLAMP_VALUE'])
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
    """Manages the interaction between the agents and the environment"""
    def __init__(self, state_size, action_size, num_agents, writer, random_seed, dirname, print_every=1000, model_path=None, eval_mode=False):
        """Initialise the trainer object

        Parameters:    
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            num_agents (int): number of agents
            agent_index (int): index (id) of current agent
            writer (object): visdom visualiser for realtime visualisations            
            random_seed (int): random seed
            dirname (string): output directory to store config, losses
            print_every (int): how often to print progress
            model_path (string): if defined, load saved model to resume training
            eval_mode (bool): whether to use eval mode
        """
        self.state_size = state_size
        self.action_size = action_size
        self.num_agents = num_agents
        self.writer = writer
        self.print_every = print_every
        self.dirname = dirname
        # initialise all agents
        self.agents = [
            MADDPGAgent(state_size, action_size, self.num_agents, agent_index=i, writer=self.writer, random_seed=random_seed, dirname=dirname, print_every=print_every, model_path=model_path) 
            for i in range(self.num_agents)]
        self.memory = ReplayBuffer(self.action_size, config['BUFFER_SIZE'], config['BATCH_SIZE'], random_seed)
        self.learn_step = 0        

    def act(self, states, add_noise=True):
        """Executes act on all the agents

        Parameters:
            states (list): list of states, one for each agent
            add_noise (bool): whether to apply noise to the actions
        """
        actions = []
        for i, agent in enumerate(self.agents):
            action = agent.act(states[i], add_noise)
            actions.append(action)
        return actions

    def step(self, states, actions, rewards, next_states, dones):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        self.learn_step += 1
        # store a single entry for each step i.e the experience of each agent for a step 
        # gets stored as single entry.
        states = np.expand_dims(states, 0)
        actions = np.expand_dims(np.array(actions).reshape(self.num_agents, self.action_size),0)
        rewards = np.expand_dims(np.array(rewards).reshape(self.num_agents, -1),0)
        dones = np.expand_dims(np.array(dones).reshape(self.num_agents, -1),0)
        next_states = np.expand_dims(np.array(next_states).reshape(self.num_agents, -1), 0)
        # Use debugger to explore the shape
        # import pdb; pdb.set_trace()
        self.memory.add(states, actions, rewards, next_states, dones)

        # Get agent to learn from experience if we have enough data/experiences in memory
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
        
        # Plot real-time graphs and store losses
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
        """Resets the noise for each agent"""
        for agent in self.agents:
            agent.reset()

    def checkpoint(self):
        """Checkpoint actor and critic models"""
        for agent in self.agents:
            agent.checkpoint()
