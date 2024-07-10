from itertools import count
from agent import select_action, finish_episode

def train(env, policy, optimizer, args):
    running_reward = -200
    for i_episode in count(1):
        state = env.reset()
        ep_reward = 0
        for t in range(1, args.max_timesteps + 1):
            action = select_action(policy, state)
            state, reward, done, info = env.step(action)
            if args.render:
                env.render()
            policy.rewards.append(reward)
            ep_reward += reward
            if done:
                break

        running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward
        finish_episode(policy, optimizer, args.gamma)
        if i_episode % args.log_interval == 0:
            print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                  i_episode, ep_reward, running_reward))
        
        if running_reward > -110:
            print("Solved! Running reward is now {} and "
                  "the last episode runs to {} time steps!".format(running_reward, t))
            break

