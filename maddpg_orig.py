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
    "BUFFER_SIZE": int(2e6),     # replay buffer size
    "BATCH_SIZE": 1024,           # minibatch size
    "GAMMA": 0.95,                # discount factor
    "TAU": 1e-3,                 # for soft update of target parameters
    "LR_ACTOR": 1e-3,            # learning rate of the actor 
    "LR_CRITIC": 1e-3,           # learning rate of the critic
    "WEIGHT_DECAY": 0,           # L2 weight decay
    # "SCALE_REWARD": 0.1,       # scaling factor applied to rewards (http://arxiv.org/abs/1604.06778)
    "SCALE_REWARD": 1.0,         # default scaling factor
    "SIGMA": 0.01,
    "FC1": 64,
    "FC2": 64,
}

# Create logger
logger = logging.getLogger(__name__)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logger.info('Using device: {}'.format(device))

class MADDPGAgent():
    """Multi Agent DDPG Implementation"""
    def __init__(self, state_size, action_size, num_agents, writer, random_seed, dirname, print_every=1000, model_path=None, eval_mode=False):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)
        self.num_agents = num_agents
        self.writer = writer
        self.dirname = dirname
        self.print_every = print_every

        self.local_critic = Critic(self.state_size * self.num_agents, self.action_size * self.num_agents, random_seed, fc1_units=config['FC1'], fc2_units=config['FC2']).to(device)
        self.critic_optimizer = optim.Adam(self.local_critic.parameters(), lr=config['LR_CRITIC'], weight_decay=config['WEIGHT_DECAY'])
        self.target_critic = Critic(self.state_size * self.num_agents, self.action_size * self.num_agents, random_seed, fc1_units=config['FC1'], fc2_units=config['FC2']).to(device)

        self.local_actors = []
        self.target_actors = []
        self.actor_optimizers = []
        self.noise = []
        # Noise process
        for i in range(self.num_agents):
            self.local_actors.append(Actor(self.state_size, self.action_size, random_seed, fc1_units=config['FC1'], fc2_units=config['FC2']).to(device))
            self.target_actors.append(Actor(self.state_size, self.action_size, random_seed, fc1_units=config['FC1'], fc2_units=config['FC2']).to(device))
            self.actor_optimizers.append(optim.Adam(self.local_actors[i].parameters(), lr=config['LR_ACTOR']))
            self.noise.append(OUNoise(self.action_size, random_seed, sigma=config['SIGMA']))

        self.memory = ReplayBuffer(self.action_size, config['BUFFER_SIZE'], config['BATCH_SIZE'], random_seed)
        self.learn_step = 0


    def act(self, states, add_noise=True):
        actions = []
        for i in range(self.num_agents):
            action = self._act(self.local_actors[i], states[i], add_noise, self.noise[i])
            actions.append(action)
        return actions

    def _act(self, actor, state, add_noise, noise):
        state = torch.from_numpy(np.expand_dims(state, 0)).float().to(device)
        # Run inference in eval mode
        actor.eval()
        with torch.no_grad():
            action = actor(state).cpu().data.numpy()
        actor.train()
        # add noise if true
        if add_noise:
            action += noise.sample()
        return np.clip(action, -1, 1)

    def reset(self):
        for i in range(self.num_agents):
            self.noise[i].reset()

    # def step(self, states, actions, rewards, next_states, dones):
    #     states = np.expand_dims(states, 0)
    #     actions = np.expand_dims(np.array(actions).reshape(self.num_agents, self.action_size),0)
    #     rewards = np.expand_dims(np.array(rewards),0)
    #     dones = np.expand_dims(np.array(dones),0)
    #     next_states = np.expand_dims(next_states, 0)
    #     self.memory.add(states, actions, rewards, next_states, dones)
    #     # import pdb; pdb.set_trace()
    #     if len(self.memory) > config['BATCH_SIZE']:
    #         experiences = self.memory.sample()
    #         self.learn(config['GAMMA'])

    # def learn(self, gamma):
    #     self.learn_step += 1
    #     # ---------------central critic-------------------
    #     experiences = self.memory.sample()
    #     states, actions, rewards, next_states, dones = experiences
    #     # use target actor to get action        
    #     next_actions = torch.zeros((len(states), self.num_agents, self.action_size)).to(device)
    #     for i in range(len(states)):         
    #         for j in range(self.num_agents):
    #             next_actions[i][j] = self.target_actors[j](states[i][j])
         
    #     # next_actions = torch.cat(next_actions, dim=1)
    #     # import pdb; pdb.set_trace()
    #     next_actions = torch.reshape(next_actions, (next_actions.shape[0], -1,))
    #     critic_states = torch.reshape(next_states, (next_states.shape[0], -1,))

    #     # calculate target and expected
    #     Q_targets_next = self.target_critic(critic_states, next_actions)
    #     Q_targets = flatten(rewards).mean(dim=1) + (gamma * Q_targets_next * (1 - flatten(dones).mean(dim=1).unsqueeze(1)))


    #     Q_expected = self.local_critic(flatten(states), flatten(actions))

    #     # use mse
    #     critic_loss = F.mse_loss(Q_expected, Q_targets)        
    #     critic_loss_value = critic_loss.item()
    #     if self.learn_step % self.print_every == 0:
    #         save_to_txt(critic_loss, '{}/critic_loss.txt'.format(self.dirname))
    #         self.writer.text('critic loss: {}'.format(critic_loss), "Critic")
    #         self.writer.push(critic_loss.item(), "Loss(critic)")

    #     self.critic_optimizer.zero_grad()
    #     critic_loss.backward()
    #     self.critic_optimizer.step()

    #     # ---------------actors---------------------
    #     # predicted_actions = []
    #     predicted_actions = torch.zeros((len(states), self.num_agents, self.action_size)).to(device)
    #     for i in range(len(states)):
    #         for j in range(self.num_agents):
    #             predicted_actions[i][j]=self.local_actors[j](states[i][j]) 

    #     actor_losses = -self.local_critic(flatten(states), flatten(predicted_actions)).mean()
    #     if self.learn_step % self.print_every == 0:
    #         save_to_txt(actor_losses, '{}/actor_loss.txt'.format(self.dirname))
    #         self.writer.text('actor loss: {}'.format(actor_losses), "Actor")
    #         self.writer.push(actor_losses.item(), "Loss(actor)")

    #     actor_losses_value = actor_losses.item()
    #     # import pdb; pdb.set_trace()

    #     for i in range(self.num_agents):
    #         self.actor_optimizers[i].zero_grad()

    #     actor_losses.backward()

    #     for i in range(self.num_agents):
    #         self.actor_optimizers[i].step()

    #     # ----------------------- update target networks ----------------------- #
    #     if self.learn_step == 1:
    #         # One time only, start local and target with same parameters
    #         self._copy_weights(self.local_critic, self.target_critic)
    #         for actor_local, actor_target in zip(self.local_actors, self.target_actors):
    #             self._copy_weights(actor_local, actor_target)
    #     else:
    #         self.soft_update(self.local_critic, self.target_critic, config["TAU"])                    
    #         for actor_local, actor_target in zip(self.local_actors, self.target_actors):
    #             self.soft_update(actor_local, actor_target, config["TAU"]) 

    #     return actor_losses_value, critic_loss_value

    def step(self, states, actions, rewards, next_states, dones):
        # import pdb; pdb.set_trace()
        states = np.expand_dims(states, 0)
        actions = np.expand_dims(np.array(actions).reshape(self.num_agents, self.action_size),0)
        rewards = np.expand_dims(np.array(rewards).reshape(self.num_agents, -1),0)
        dones = np.expand_dims(np.array(dones).reshape(self.num_agents, -1),0)
        next_states = np.expand_dims(np.array(next_states).reshape(self.num_agents, -1), 0)
        # import pdb; pdb.set_trace()
        self.memory.add(states, actions, rewards, next_states, dones)
        if len(self.memory) > config['BATCH_SIZE']:
            experiences = self.memory.sample()
            self.learn(config['GAMMA'])

    def learn(self, gamma):
        self.learn_step += 1
        actor_losses = []
        critic_losses = []
        for i in range(self.num_agents):
            actor_loss, critic_loss = self.learn_single_agent(i, gamma)
            actor_losses.append(actor_loss)
            critic_losses.append(critic_loss)
        if self.learn_step % self.print_every == 0:            
            self.writer.text('actor{} loss: {}'.format(i+1, actor_loss), "Actor {}".format(i+1))

        if self.learn_step % self.print_every == 0:            
            save_to_txt(actor_losses, '{}/actor{}_loss.txt'.format(self.dirname, i+1))
            self.writer.push(actor_losses, "Loss(actor)")

        return actor_losses, critic_losses

    def learn_single_agent(self, agent_index, gamma):
        # ---------------central critic-------------------
        experiences = self.memory.sample()
        states, actions, rewards, next_states, dones = experiences
        # use target actor to get action        
        next_actions = torch.zeros((len(states), self.num_agents, self.action_size)).to(device)
        for j in range(self.num_agents):
            next_actions[:, j] = self.target_actors[j](states[:, j, :])
         
        # next_actions = torch.cat(next_actions, dim=1)
        # import pdb; pdb.set_trace()
        next_actions = torch.reshape(next_actions, (next_actions.shape[0], -1,))
        critic_states = torch.reshape(next_states, (next_states.shape[0], -1,))

        # calculate target and expected
        Q_targets_next = self.target_critic(critic_states, next_actions)
        Q_targets = rewards[:, agent_index, :] + (gamma * Q_targets_next * (1 - dones[:, agent_index, :]))
        Q_expected = self.local_critic(flatten(states), flatten(actions))

        # use mse
        critic_loss = F.mse_loss(Q_expected, Q_targets)        
        critic_loss_value = critic_loss.item()
        if self.learn_step % self.print_every == 0:
            save_to_txt(critic_loss, '{}/critic_loss.txt'.format(self.dirname))
            self.writer.text('critic loss: {}'.format(critic_loss), "Critic")
            self.writer.push(critic_loss.item(), "Loss(critic)")

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ---------------actor---------------------
        # predicted_actions = self.local_actors[agent_index](states[:, agent_index, :])
        # import pdb; pdb.set_trace()
        # actor_loss = -self.local_critic(states[:, agent_index, :], predicted_actions).mean()
        predicted_actions = torch.zeros((len(states), self.num_agents, self.action_size)).to(device)
        for j in range(self.num_agents):
            predicted_actions[:, j]=self.local_actors[agent_index](states[:, j]) 
        actor_loss = -self.local_critic(flatten(states), flatten(predicted_actions)).mean()
        # actor_loss = -self.local_critic(flatten(states), flatten(actions)).mean()

        # if self.learn_step % self.print_every == 0:
        #     save_to_txt(actor_loss, '{}/actor_loss.txt'.format(self.dirname))
        #     self.writer.text('actor loss: {}'.format(actor_loss), "Actor")
        #     self.writer.push(actor_loss.item(), "Loss(actor)")

        actor_loss_value = actor_loss.item()
        # import pdb; pdb.set_trace()

        self.actor_optimizers[agent_index].zero_grad()
        actor_loss.backward()
        self.actor_optimizers[agent_index].step()

        # ----------------------- update target networks ----------------------- #
        if self.learn_step == 1:
            # One time only, start local and target with same parameters
            self._copy_weights(self.local_critic, self.target_critic)
            self._copy_weights(self.local_actors[agent_index], self.target_actors[agent_index])
        else:
            self.soft_update(self.local_critic, self.target_critic, config["TAU"])
            self.soft_update(self.local_actors[agent_index], self.target_actors[agent_index], config["TAU"])

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

        torch.save(self.local_critic.state_dict(), '{}/multi/checkpoint_critic.pth'.format(self.dirname))            
        for i in range(self.num_agents):
            torch.save(self.local_actors[i].state_dict(), '{}/multi/checkpoint_actor_{}.pth'.format(self.dirname, i))