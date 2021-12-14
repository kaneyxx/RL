import gym

class CartPole(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self.env = env

    def step(self, action):
        next_state, reward, terminal, info = self.env.step(action)

        return next_state, reward, terminal, info

if __name__ == "__main__":
    env = gym.make("CartPole-v0")
    env = CartPole(env)
    env.reset()
    print(env.step(1))