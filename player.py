import os
import argparse
import logging
import numpy as np
from ddpg import DDPGMultiAgent
from maddpg import MADDPGAgentTrainer
from utils import VisWriter
from unityagents import UnityEnvironment

# Setup logger
logger = logging.getLogger(__name__)
dirname = 'logs'
if not os.path.exists(dirname):
    os.makedirs(dirname)

def play(env, brain_name, num_agents, agent, num_episodes=10):
    """Execute policy in specified environment

    Args:
        env: Unity environment object
        brain_name: A string parameter indicating name of brain
        num_agents: An integer representing number of agents
        agent: An instance of agent (DDPGMultiAgent or MADDPGAgentTrainer)
    """

    best_score = -np.inf
    scores = []
    env_info = env.reset(train_mode=False)[brain_name]
    states = env_info.vector_observations
    for episode in range(num_episodes):
        episode_scores = []
        score = np.zeros(num_agents)
        while True:
            actions = agent.act(states)
            env_info = env.step(actions)[brain_name]
            next_states = env_info.vector_observations
            rewards = env_info.rewards
            dones = env_info.local_done
            score += env_info.rewards
            states = next_states
            if np.any(dones):
                break

        episode_scores.append(score)
        logger.info('Episode Score: {:.2f}'.format(np.mean(scores)))
        scores.append(score)
    logger.info('Final Score: {:.2f}'.format(np.mean(scores)))

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--env", type=str, help="Full path of environment")
    parser.add_argument("--agent", type=str, help="Choose implemntation [maddpg,ddpg]")
    parser.add_argument("--model", type=str, help="Model checkpoint path, use if you wish to continue training from a checkpoint")
    parser.add_argument("--num_episodes", type=int, help="Number of episodes")


    args = parser.parse_args()
    env = UnityEnvironment(file_name=args.env)
    # brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    # reset the environment
    env_info = env.reset(train_mode=False)[brain_name]

    # number of agents
    num_agents = len(env_info.agents)
    print('Number of agents:', num_agents)

    state = env_info.vector_observations
    state_shape = state.shape[1]
    action_size = brain.vector_action_space_size    

    writer = VisWriter(vis=False)
    if args.agent == 'ddpg':
        agent = DDPGMultiAgent(state_shape, action_size, num_agents, writer=writer, random_seed=10, dirname=dirname, print_every=100, model_path=args.model, eval_mode=True)
    elif args.agent == 'maddpg':
        agent = MADDPGAgentTrainer(state_shape, action_size, num_agents, writer=writer, random_seed=10, dirname=dirname, print_every=100, model_path=args.model, eval_mode=True)

    play(env, brain_name, num_agents, agent, num_episodes=args.num_episodes)

if __name__ == "__main__":
    main()