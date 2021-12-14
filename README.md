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
* agent : "Q-Learning", "Sarsa", "SarsaLambda"
* episode : How many episodes you want the agent to learn
* lr : Learning rate
* gamma : Discount rate
* lambda : Decaying rate for eligibility traces (only implemented in Sarsa lambda algorithm currently)
* epsilon : Low prob. for random action to make sure you will not only pick one action
* slippery : Only for FrozenLake and GridWorld env, default = False
* render : There will be a window show up if True, default = False
* test : Test on specific table file (input file path), default = None

### Deep Q Network (DQN)
<br/> 

Command:
```python
python main.py [--parameters]
```
Example 1 (Training with default settings):
```python
python main.py
```
Example 2 (Training with customized settings):
```python
python main.py --episodes 500 --batch_size 64 --replace_iter 5 --use_pretrained --render
```
Example 3 (Testing):
```python
python main.py --test "./dqn.pth" --render
```

<br/>  

Parameters:
* env : 'CartPole-v0'
* replay : Experience replay storage capacity
* episodes : How many episodes you want the agent to learn
* batch_size : Sampled batch size for each step
* lr : Learning rate
* epsilon : Prob. for random action to make sure you can explore the environment
* epsilon_decay : Epsilon decay rate for each step
* epsilon_min : Minimum epsilon
* replace_iter : Update target network once every n episodes
* use_pretrained : Add this arg if wanna use pretrained weight, default = False
* render : There will be a window show up if True, default = False
* test : Test on specific table file (input file path), default = None