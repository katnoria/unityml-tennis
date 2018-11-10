import os
from datetime import datetime
import logging
from collections import deque
import argparse
import numpy as np
import torch
from ddpg_multi import DDPGMultiAgent
from unityagents import UnityEnvironment
from utils import VisWriter, save_to_txt

# Create logger
logger = logging.getLogger("ddpg_multi")
formatter = logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s') 
logger.setLevel(logging.DEBUG)

now = datetime.now().strftime('%Y-%m-%d-%H%M%S')
dirname = 'runs/{}'.format(now)
if not os.path.exists(dirname):
    os.makedirs(dirname)

filehandler = logging.FileHandler(filename='{}/ddpg_multi.log'.format(dirname))
filehandler.setFormatter(formatter)
filehandler.setLevel(logging.DEBUG)
logger.addHandler(filehandler)
# Uncomment to enable console logger
steamhandler = logging.StreamHandler()
steamhandler.setFormatter(formatter)
steamhandler.setLevel(logging.INFO)
logger.addHandler(steamhandler)

def ddpg(env, brain_name, num_agents, agent, writer, n_episodes=300, max_t=1000, print_every=50, stop_on_solve=True):
    """Train DDPG Agent

    Params    
    ======
        env (object): Unity environment instance
        brain_name (string): name of brain
        num_agents (int): number of agents
        agent (DDPGMultiAgent): agent instance
        writer (VisWriter): Visdom visualiser for realtime plots
        n_episodes (int): number of episodes to train the network
        max_t (int): number of timesteps in each episode
        print_every (int): how often to print the progress
        stop_on_solve (bool): whether to stop training as soon as environment is solved
    """
    best_score = -np.inf
    scores_deque = deque(maxlen=100)
    scores = []
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name]
        states = env_info.vector_observations
        agent.reset()
        score = np.zeros(num_agents)
        for t in range(max_t):
            actions = agent.act(states)
            env_info = env.step(actions)[brain_name]           # send all actions to tne environment
            next_states = env_info.vector_observations         # get next state (for each agent)
            rewards = env_info.rewards                         # get reward (for each agent)            
            dones = env_info.local_done                        # see if episode finished
            score += env_info.rewards                          # update the score (for each agent)
            agent.step(states, actions, rewards, next_states, dones)
            states = next_states                               # roll over states to next time step
            if np.any(dones):                                  # exit loop if episode finished
                break
        
        scores_deque.append(np.max(score))
        scores.append(np.max(score))
        current_score = np.mean(scores_deque)
        # keep storing current score (incase we terminate, we'll have data for plotting/comparison)
        save_to_txt(current_score, '{}/scores_multi.txt'.format(dirname))
        # Publish and save
        writer.text('Episode {}/{}: Average score(100): {}'.format(i_episode, n_episodes, current_score), "Average 100 episodes")
        writer.push(np.mean(scores_deque), "Average Score")
        logger.info('Episode {}\tAverage Score: {:.2f}'.format(i_episode, current_score))

        if len(scores) > 0:
            writer.push(scores[-1], "Score")

        if current_score >= best_score:
            logger.info('Best score found, old: {}, new: {}'.format(best_score, current_score))
            best_score = current_score
            agent.checkpoint()

        if i_episode % print_every == 0:
            logger.info('Episode {}\tAverage Score: {:.2f}'.format(i_episode, current_score))

        # check environment solved
        if current_score >= 0.5:            
            logger.info('Environment solved in {} episodes'.format(i_episode))
            if stop_on_solve:
                logger.info('Terminating agent training')
                break
            
    logger.info('Final Average Score: {:.2f}'.format(current_score))
    return scores

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--num_episodes", type=int, default=1000, help="Total number of episodes to train")
    parser.add_argument("--max_t", type=int, default=1000, help="Max timestep in a single episode")
    parser.add_argument("--vis", type=bool, default=True, help="Whether to use visdom to visualise training")
    parser.add_argument("--model", type=str, default=None, help="Model checkpoint path, use if you wish to continue training from a checkpoint")
    parser.add_argument("--info", type=str, default="", help="Use this to attach notes to your runs")
    parser.add_argument("--stop_on_solve", type=bool, default=True, help="Stop as soon as the environment is solved")

    args = parser.parse_args()

    # visualiser
    writer = VisWriter(vis=args.vis)
    # save info/comments about the experiment
    save_to_txt(args.info, '{}/info.txt'.format(dirname))

    # Unity Env
    env = UnityEnvironment(file_name='env/Tennis_Linux_NoVis/Tennis.x86_64')    
    # brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    # reset the environment
    env_info = env.reset(train_mode=True)[brain_name]
    # number of agents
    num_agents = len(env_info.agents)
    print('Number of agents:', num_agents)

    state = env_info.vector_observations
    state_shape = state.shape[1]
    action_size = brain.vector_action_space_size    

    agent = DDPGMultiAgent(state_shape, action_size, num_agents, writer=writer, random_seed=10, dirname=dirname, print_every=100, model_path=args.model)
    scores = ddpg(env, brain_name, num_agents, agent, writer, n_episodes=args.num_episodes, max_t=args.max_t, stop_on_solve=args.stop_on_solve)
    # save all scores
    save_to_txt('\n'.join([score.tolist() for score in scores]), '{}/scores_multi_full.txt'.format(dirname))
    # save_to_txt('\n'.join(list(scores)), '{}/scores_multi_full.txt'.format(dirname))


if __name__ == "__main__":    
    main()