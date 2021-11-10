# RL

### Q-learning & Sarsa

command: python main.py \--[parameters]

* env : "FrozenLake", "CliffWalking", "GridWorld"
* agent : "Q-Learning", "Sarsa"
* episode : how many episodes you want the agent to learn
* lr : learning rate
* gamma : discount rate
* epsilon : low prob. for random action to make sure you will not only pick one action
* slippery : for FrozenLake and GridWorld env
* render : default is False, there will be a window show up if True