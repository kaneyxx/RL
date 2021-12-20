import torch
import torch.nn as nn
import numpy as np
from replay import ExperienceReplay

class DQN():
    def __init__(self, 
                model,
                ExperienceReplay,
                n_states=None,
                n_actions=None,
                batch_size=None,
                epsilon=None,
                epsilon_decay=None,
                epsilon_min=None,
                gamma=None, 
                lr=None,
                target_replace_iter=None,
                memory_capacity=None):

        self.eval_net = model(n_states, n_actions)
        self.target_net = model(n_states, n_actions)
        self.memory = ExperienceReplay
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=lr)
        # self.optimizer = torch.optim.RMSprop(self.eval_net.parameters(), lr=lr, momentum=0)
        # self.loss_function = nn.MSELoss()
        self.loss_function = nn.SmoothL1Loss()
        self.learn_step_counter = 0 # count to update target network
        self.memory_counter = 0 # count to save transition 
        
        self.n_states = n_states
        self.n_actions = n_actions
        self.batch_size = batch_size
        self.lr = lr
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.gamma = gamma
        self.target_replace_iter = target_replace_iter
        self.memory_capacity = memory_capacity
    
    def decay(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def sample_action(self, states):
        x = torch.FloatTensor(states)

        # epsilon-greedy
        if np.random.uniform() < self.epsilon:
            action = np.random.randint(0, self.n_actions)
        else: # sample best action based on current policy
            action_value = self.eval_net(x).view(-1, 2) # estimate each action value
            action = torch.max(action_value, 1)[1].data.numpy()[0] # pick action with max estimated value

        return action

    def predict(self, states):
        self.target_net.eval()
        x = torch.unsqueeze(torch.FloatTensor(states), 0)
        with torch.no_grad():
            actions_value = self.target_net(x)
            action = torch.max(actions_value, 1)[1].data.numpy()[0]
        return action
    
    def store_transition(self, state, action, reward, next_state, terminal):
        # pack experience
        transition = np.hstack((state, action, reward, next_state, terminal))

        # store experience
        self.memory.append(transition)

    def learn(self):
        self.eval_net.to("cpu")
        self.target_net.to("cpu")
        self.target_net.eval()

        # sample batch_size experiences
        state_batch, action_batch, reward_batch, next_state_batch, terminal_batch = self.memory.sample(self.batch_size)

        # calculate the loss
        q_eval = self.eval_net(state_batch).gather(1, action_batch) # compute the Q value based on state batch by eval_net
        q_next = self.target_net(next_state_batch).max(1)[0].unsqueeze(1).detach() # detach for not to update target_net
        q_target = reward_batch + self.gamma * q_next * terminal_batch # compute the Q value based on state batch by target_net
        loss = self.loss_function(q_eval, q_target)

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        # for param in self.eval_net.parameters():
        #     param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        # update target_net for every update_iter
        self.learn_step_counter += 1
        # print(self.learn_step_counter)
        if self.learn_step_counter % self.target_replace_iter == 0:
            for target_param, eval_param in zip(self.target_net.parameters(), self.eval_net.parameters()):
                target_param.data.copy_(0.01*eval_param.data + 0.99*target_param)
            # self.target_net.load_state_dict(self.eval_net.state_dict())
    
    def save(self):
        self.target_net.load_state_dict(self.eval_net.state_dict())
        torch.save(self.target_net, "./dqn.pth")

    def load(self, PATH):
        self.target_net = torch.load(PATH)
        self.eval_net = torch.load(PATH)

    def _get_epsilon(self):
        return self.epsilon
