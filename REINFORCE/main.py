import argparse
import torch
from env import create_env
from model import Policy
from trainer import train

parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
parser.add_argument('--env-name', type=str, default='CartPole-v1', metavar='G',
                    help='name of the environment to run (default: CartPole-v1)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 543)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='interval between training status logs (default: 10)')
parser.add_argument('--max-timesteps', type=int, default=1000, metavar='N',
                    help='max timesteps per episode (default: 1000)')
parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                    help='learning rate (default: 1e-3)')
args = parser.parse_args()

env = create_env(args.env_name, args.seed)
torch.manual_seed(args.seed)

state_space = env.observation_space.shape[0]
action_space = env.action_space.n

policy = Policy(state_space, action_space)
optimizer = torch.optim.Adam(policy.parameters(), lr=args.lr)

if __name__ == '__main__':
    train(env, policy, optimizer, args)



