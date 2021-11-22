# RL

### Q-learning & Sarsa
<br/> 

Command:
```python
python main.py [--parameters]
```
Example 1 (Training):
```python
python main.py --env "CliffWalking" --agent "Sarsa" --episode 500 --render
```
Example 2 (Testing):
```python
python main.py --env "CliffWalking" --agent "Sarsa" --test "./qtable_CliffWalking_Sarsa.npy"
```

<br/>  

Parameters:
* env : "FrozenLake", "CliffWalking", "GridWorld"
* agent : "Q-Learning", "Sarsa"
* episode : How many episodes you want the agent to learn
* lr : Learning rate
* gamma : Discount rate
* epsilon : Low prob. for random action to make sure you will not only pick one action
* slippery : Only for FrozenLake and GridWorld env, default = False
* render : There will be a window show up if True, default = False
* test : Test on specific table file (input file path), default = None