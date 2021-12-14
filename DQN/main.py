from agent import DQN
from model import ValueNet
from replay import ExperienceReplay
from env import CartPole
from trainer import run_episode, test_episode
import gym
import argparse

# arguments
parser = argparse.ArgumentParser()
parser.add_argument('--env', type=str, default='CartPole-v0', help='env name = CartPole-v0')
parser.add_argument('--replay', type=int, default=10000, help='Experience replay storage count')
parser.add_argument('--episodes', type=int, default=1000, help='how many episodes gotta train')
parser.add_argument('--batch_size', type=int, default=16, help='batch size for training each step')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate for training')
parser.add_argument('--epsilon', type=float, default=1.0, help='epsilon greedy sample action, modify it when you dont want it decay by episode')
parser.add_argument('--epsilon_decay', type=float, default=0.999, help='epsilon will multiply decay rate for each step')
parser.add_argument('--epsilon_min', type=float, default=0.01, help='minimum epsilon value')
parser.add_argument('--gamma', type=float, default=0.9, help='discount rate for estimating future value')
parser.add_argument('--replace_iter', type=int, default=10, help='update target network once every n episodes')
parser.add_argument('--use_pretrained', default=False, action='store_true', help='add this arg if wanna use pretrained weight')
parser.add_argument('--render', default=False, action='store_true', help='render agent interaction')
parser.add_argument('--test', type=str, default=None, help='use the testing weight file path for testing')


def main():
    args = parser.parse_args()
    env = gym.make(args.env)
    env = CartPole(env)
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
    memory_capacity = len(memory_buffer)

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
    if args.use_pretrained:
        """
        Default model architecture:
        self.fc1 = nn.Linear(in_features=n_states, out_features=24, bias=False)
        self.fc2 = nn.Linear(in_features=24, out_features=48, bias=False)
        self.fc3 = nn.Linear(in_features=48, out_features=n_actions, bias=False)
        """
        agent.load("./dqn_cartpole_pretrained.pth")
        print("Pretrained weights loaded successfully!")
    
    if args.test is None:
        run_episode(agent, env, n_episodes, args.render)
    else:
        agent.load(args.test)
        test_episode(agent, env, args.render) # default test episodes = 100


if __name__ == "__main__":
    main()