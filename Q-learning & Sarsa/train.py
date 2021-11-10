import gym
from env import CliffWalkingWrapper, FrozenLakeWapper, GridWorld
from agent import QLearningAgent
import time

def run_episode(env, agent, render=False):
    total_steps = 0  # record steps per episode
    total_reward = 0 # record total reward per episode

    obs = env.reset()  # restart an episode

    while True:
        
        action = agent.sample(obs)  # pick an action
        next_obs, reward, done, _ = env.step(action)  # interact with env
        
        # training
        agent.learn(obs, action, reward, next_obs, done)

        obs = next_obs  # store observation
        total_reward += reward
        total_steps += 1  # steps
        if render:
            env.render()  # render graph
        if done:
            break
            
        # time.sleep(0.01)
    return total_reward, total_steps


def test_episode(env, agent):
    total_reward = 0
    total_steps = 0
    obs = env.reset()
    while True:
        action = agent.predict(obs)  # greedy
        next_obs, reward, done, _ = env.step(action)
        total_reward += reward
        total_steps += 1
        obs = next_obs
        time.sleep(0.5)
        env.render()
        if done:
            print('test steps = %s, reward = %.1f' % (total_steps, total_reward))
            break
        