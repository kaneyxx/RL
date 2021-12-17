from agent import DQN
from model import ValueNet
from replay import ExperienceReplay
from env import CartPole
from trainer import run_episode, test_episode
import gym
import argparse

# arguments
parser = argparse.ArgumentParser()
parser.add_argument('--env', type=str, default='CartPole-v0', help='env = CartPole-v0')
parser.add_argument('--replay', type=int, default=100000, help='Experience replay storage count')
parser.add_argument('--episodes', type=int, default=1000, help='How many episodes gotta train')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training each step')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for training')
parser.add_argument('--epsilon', type=float, default=0.1, help='Epsilon greedy sample action, modify it when you dont want it decay by episode')
parser.add_argument('--epsilon_decay', type=float, default=0.995, help='Epsilon will multiply decay rate')
parser.add_argument('--epsilon_min', type=float, default=0.001, help='Minimal epsilon value')
parser.add_argument('--gamma', type=float, default=0.99, help='Discount rate for estimating future value')
parser.add_argument('--replace_iter', type=int, default=20, help='Update target network once every n episodes')
parser.add_argument('--use_pretrained', type=str, default=None, help='Load pretrained weights')
parser.add_argument('--render', default=False, action='store_true', help='Render agent/env interaction')
parser.add_argument('--test', type=str, default=None, help='Test on specific policy file')

def main():
    args = parser.parse_args()
    env = gym.make(args.env)
    # env = CartPole(env)
    memory_buffer = ExperienceReplay(args.replay)
    n_actions = env.action_space.n
    n_states = env.observation_space.shape[0]

    # Hyper parameters
    n_episodes = args.episodes
    batch_size = args.batch_size
    lr = args.lr
    epsilon = args.epsilon
    epsilon_decay = args.epsilon_decay
    epsilon_min = args.epsilon_min
    gamma = args.gamma
    target_replace_iter = args.replace_iter 
    memory_capacity = args.replay

    # init agent
    agent = DQN(ValueNet,
            memory_buffer,
            n_states,
            n_actions,
            batch_size,
            epsilon,
            epsilon_decay,
            epsilon_min,
            gamma, 
            lr,
            target_replace_iter,
            memory_capacity)
    if args.use_pretrained is not None:
        agent.load(args.use_pretrained)
        print("Pretrained weights loaded successfully!")
    
    if args.test is None:
        run_episode(agent, env, n_episodes, args.render)
    else:
        agent.load(args.test)
        print("Testing weights loaded successfully!")
        test_episode(agent, env, args.render) # default test episodes = 100


if __name__ == "__main__":
    main()