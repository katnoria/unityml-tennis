import os
import argparse
import logging
import numpy as np
from ddpg import DDPGAgent
from ddpg_multi import DDPGMultiAgent
from utils import VisWriter
from unityagents import UnityEnvironment

logger = logging.getLogger(__name__)

dirname = 'logs'
if not os.path.exists(dirname):
    os.makedirs(dirname)

def play(env, brain_name, num_agents, agent):
    """Execute policy in specified environment

    Params
    ======
        env (object): Unity environment
        brain_name (string): Name of brain
        num_agents (int): Number of agents
        agent (object): Pretrained agent
    """
    best_score = -np.inf
    scores = []
    env_info = env.reset(train_mode=False)[brain_name]
    states = env_info.vector_observations
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

    scores.append(score)
    logger.info('Final Score: {:.2f}'.format(np.mean(scores)))

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--env", type=str, help="Full path of environment")
    parser.add_argument("--model", type=str, help="Model checkpoint path, use if you wish to continue training from a checkpoint")
    parser.add_argument("--agent", type=int, help="Number of agents. Specify either 1 or 20")


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
    
    if args.agent == 1:
        agent = DDPGAgent(state_shape, action_size, writer=writer, dirname=dirname, random_seed=42, print_every=100, model_path=args.model, eval_mode=True)
    elif args.agent == 20:
        agent = DDPGMultiAgent(state_shape, action_size, num_agents, writer=writer, random_seed=10, dirname=dirname, print_every=100, model_path=args.model, eval_mode=True)
    else:
        raise ValueError('Invalid number of agents specified: {}. Your choices are 1 or 20'.format(args.agent))

    play(env, brain_name, num_agents, agent)

if __name__ == "__main__":    
    main()