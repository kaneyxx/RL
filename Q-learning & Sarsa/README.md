### Q-learning & Sarsa
<br/> 

Example:
```python
python main.py [--parameters]
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