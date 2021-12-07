import gym
from env import CliffWalkingWrapper, FrozenLakeWapper, GridWorld
from agent import Agent
import time

def run_episode(env, agent, render=False):
    total_steps = 0  # record steps per episode
    total_reward = 0 # record total reward per episode

    obs = env.reset()  # restart an episode

    while True:
        
        action = agent.sample(obs)  # pick an action
        next_obs, reward, done, _ = env.step(action)  # interact with env
        next_action = agent.sample(next_obs) # for Sarsa
        
        # training
        agent.learn(obs, action, reward, next_obs, next_action, done)

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

if __name__ == "__main__":
    
    # for testing presentation
    envName = "CliffWalking"
    env = gym.make("CliffWalking-v0")  # 0 up, 1 right, 2 down, 3 left
    env = CliffWalkingWrapper(env)
    agent = SarsaAgent(
            obs_n=env.observation_space.n,
            act_n=env.action_space.n,
            learning_rate=0.01,
            gamma=0.9,
            e_greed=0.1)
    agent.restore(npy_file="./qtable_CliffWalking.npy")
    test_episode(env, agent)