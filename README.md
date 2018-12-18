[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135623-e770e354-7d12-11e8-998d-29fc74429ca2.gif "Trained Agent"
[image2]: https://user-images.githubusercontent.com/10624937/42135622-e55fb586-7d12-11e8-8a54-3c31da15a90a.gif "Soccer"


# Multi-Agent Environment: Collaboration and Competition

### Introduction

In this repo, we are going to solve the [Tennis](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis) environment.

![Trained Agent][image1]

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1.  If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01.  Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation.  Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping. 

The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
- This yields a single **score** for each episode.

The environment is considered solved, when the average (over 100 episodes) of those **scores** is at least +0.5.

### Solving the Environment

The solution is explained in the reports below:

👉 [Click here for Multi-Agent DDPG solution](Report.md)

👉 [Click here for DDPG solution](Report_DDPG.md)


### Getting Started

1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
    - Linux (headless): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux_NoVis.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

2. Place the file in env directory of this repo, and unzip (or decompress) the file. 

3. (Optional but recommended) create a conda environment
```
conda create -n myenv python=3.6
```

4. Install dependencies
```
conda activate myenv
pip install .
```

5. Install unity ml-agents using the [instructions](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Installation.md) here.


### Instructions

In order to train your agent, first start visdom 
```
conda activate myenv
visdom
```
then launch the training

Default command line arguments 
```
conda activate myenv
cd src
python ddpg_trainer.py --help

usage: ddpg_trainer.py [-h] [--num_episodes NUM_EPISODES] [--max_t MAX_T]
                       [--vis VIS] [--model MODEL] [--info INFO]
                       [--stop_on_solve STOP_ON_SOLVE]

optional arguments:
  -h, --help            show this help message and exit
  --num_episodes NUM_EPISODES
                        Total number of episodes to train (default: 1000)
  --max_t MAX_T         Max timestep in a single episode (default: 1000)
  --vis                 Use visdom to visualise training (default: True)
  --no-vis              Do not use visdom to visualise training (default:
                        True)
  --model MODEL         Model checkpoint path, use if you wish to continue
                        training from a checkpoint (default: None)
  --info INFO           Use this to attach notes to your runs (default: )
  --stop_on_solve       Stop as soon as the environment is solved (default:
                        True)
  --no-stop_on_solve    Continue even after the environment is solved
                        (default: True)
```

For example, this is what I use the following command to train an agent using MADDPG implementaion
```
python maddpg_trainer.py --max_t 5000 --num_episodes 10000
```
and DDPG implementaion
```
python ddpg_trainer.py --max_t 5000 --num_episodes 10000
```

### Real-time monitoring

Open your web browser to view the realtime training plots @ http://127.0.0.1:8097

Every time you run the trainer, a new directory is created under src/runs with following contents:

* log file
* hyperparams.json : contains the configuration used 
* actor_losses.txt (actor_losses_multi.txt for 20 agents env): contains the loss for actor
* critic_losses.txt (critic_losses_multi.txt for 20 agents env): contains the loss for critic
* scores.txt : contains the entire score history
* scores_full.txt: Also contains the entire history but above file is updated at every episode so if you terminate before completing all episodes, this file will not be generated.
* checkpoint_actor.pth: Best weights for actor model
* checkpoint_critic.pth: Best weights for critic model


![image](data/ddpg/ddpg_visdom.jpg)

### Play

To see the players in action, use the uploaded model from checkpoints directory.

```
conda activate myenv
cd src
python player.py --help
usage: player.py [-h] [--env ENV] [--model MODEL] [--agent AGENT]

optional arguments:
  -h, --help     show this help message and exit
  --env ENV      Full path of environment (default: None)
  --model MODEL  Model checkpoint path, use if you wish to continue training
                 from a checkpoint (default: None)
  --agent AGENT  Number of agents. Specify either 1 or 20 (default: None)
```

For example
```
python player.py --env ./env/Tennis_Linux/Tennis.x86_64 --agent maddpg --model ./checkpoint/maddpg/multi
```
