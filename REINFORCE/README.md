### REINFORCE

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
python main.py --env-name MountainCar-v0 --gamma 0.99 --seed 123 --log-interval 10 --max-timesteps 1000 --lr 1e-3
```
Example 3 (Training with rendering enabled):
```python
python main.py --env-name CartPole-v1 --render
```

<br/>  

Parameters:
* env-name : Environment to run ('CartPole-v0', 'CartPole-v1', 'MountainCar-v0', etc.), default = 'CartPole-v1'
* gamma : Discount factor, default = 0.99
* seed : Random seed, default = 543
* render : Render the environment during training, default = False
* log-interval : Interval between training status logs, default = 10
* max-timesteps : Max timesteps per episode, default = 1000
* lr : Learning rate, default = 1e-3