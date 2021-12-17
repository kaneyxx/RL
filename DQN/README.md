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
* episodes : Episodes you want the agent to learn
* batch_size : Sampled batch size for each step
* lr : Learning rate
* epsilon : Prob. for random action to make sure the agent can explore the environment
* epsilon_decay : Epsilon decay rate (for every 20 episodes)
* epsilon_min : Minimal epsilon
* gamma : Discount rate for estimating future value
* replace_iter : Update target network once every n episodes
* use_pretrained : Load pretrained weights, default = False
* render : There will be a window show up if True, default = False
* test : Test on specific policy file (input file path), default = None