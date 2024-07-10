import gym

def create_env(env_name, seed):
    env = gym.make(env_name)
    env.reset()
    env.action_space.seed(seed)
    return env
