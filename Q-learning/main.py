from env import CliffWalkingWrapper, FrozenLakeWapper, GridWorld
from agent import QLearningAgent
from train import run_episode, test_episode
import argparse
import gym
import matplotlib.pyplot as plt
from plot import plot


parser = argparse.ArgumentParser()
parser.add_argument('--env', type=str, required=True, help='environment_name = FrozenLake, CliffWalking or GridWorld')
parser.add_argument('--episode', default=1000, type=int, help='episodes')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--gamma', default=0.9, type=float, help='discount rate')
parser.add_argument('--epislon', default=0.1, type=float, help='epislon')
parser.add_argument('--slippery', default=False, type=bool, help='slippery')
parser.add_argument('--render', default=False, action="store_true", help='render')


def main():
    args = parser.parse_args()
    print(args)

    if args.env == 'FrozenLake':
        envName = "FrozenLake"
        env = gym.make("FrozenLake-v1", is_slippery=False)  # 0 left, 1 down, 2 right, 3 up
        env = FrozenLakeWapper(env)
    
    elif args.env == 'CliffWalking':
        envName = "CliffWalking"
        env = gym.make("CliffWalking-v0")  # 0 up, 1 right, 2 down, 3 left
        env = CliffWalkingWrapper(env)
    
    elif args.env == "GridWorld":
        envName = "GridWorld"
        # desc = ['SFFHF', 
        #         'FHFFF',
        #         'FFFFH',
        #         'FHFFF',
        #         'FFHFG']
        desc2 = ['SFFF', 'FHFH', 'FFFH', 'HFFG']
        env = GridWorld(gridmap=desc2, is_slippery=False)

    

    agent = QLearningAgent(
        obs_n=env.observation_space.n,
        act_n=env.action_space.n,
        learning_rate=args.lr,
        gamma=args.gamma,
        e_greed=args.epislon)

    global_steps = []
    global_rewards = []
    is_render = args.render
    for episode in range(args.episode):
        episode += 1
        ep_reward, ep_steps = run_episode(env, agent, is_render)
        global_steps.append(ep_steps)
        global_rewards.append(ep_reward)
        print('Episode %s: steps = %s , reward = %.1f' % (episode, ep_steps, ep_reward))
    
    # plotting
    episode = [i for i in range(0, args.episode)]
    plot(episode=episode, steps=global_steps, rewards=global_rewards, env=envName)
    agent.save(env_name=envName)

    # testing
    test_episode(env, agent)

if __name__ == "__main__":
    main()