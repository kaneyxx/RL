import numpy as np


class QLearningAgent(object):
    def __init__(self,
                 obs_n,
                 act_n,
                 learning_rate=0.01,
                 gamma=0.9,
                 e_greed=0.1):
        self.act_n = act_n  # actions
        self.lr = learning_rate  # learning rate
        self.gamma = gamma  # reward discount rate
        self.epsilon = e_greed  # low prob. for random actions
        self.Q = np.zeros((obs_n, act_n))

    # sample action while training
    def sample(self, obs):
        if np.random.uniform(0, 1) < (1.0 - self.epsilon):  # pick action based on Q-table
            action = self.predict(obs)
        else:
            action = np.random.choice(self.act_n)  # make sure a low prob. to pick a random action
        return action

    # pick actions
    def predict(self, obs):
        Q_list = self.Q[obs, :]
        maxQ = np.max(Q_list) # greedy
        action_list = np.where(Q_list == maxQ)[0]  # actions are corresponding to maxQ
        action = np.random.choice(action_list)
        return action

    # learning algorithm (Q-table)
    def learn(self, obs, action, reward, next_obs, next_action, done):
        """ 
        off-policy
        obs: observation before interaction, s_t
        action: action picked for this interaction, a_t
        reward: reward got, r_t+1
        next_obs: observation after interaction, s_t+1
        done: if episode done (achieve the goal)
        """
        predict_Q = self.Q[obs, action]
        if done:
            target_Q = reward  # achieve the goal
        else:
            target_Q = reward + self.gamma * np.max(self.Q[next_obs, :])  # Q-learning
        self.Q[obs, action] += self.lr * (target_Q - predict_Q)  # modify q

    # save Q-table to numpy file
    def save(self, env_name=None):
        if env_name is None:
            npy_file = 'q_table.npy'
            np.save(npy_file, self.Q)
            print(npy_file + ' saved.')
            
        else:
            npy_file = 'qtable_{}.npy'.format(env_name)
            np.save(npy_file, self.Q)
            print(npy_file + ' saved.')
            

    # restore Q-table for agent
    def restore(self, npy_file=None):
        if npy_file is None:
            npy_file='q_table.npy'
            self.Q = np.load(npy_file)
            print(npy_file + ' loaded.')
        else:
            self.Q = np.load(npy_file)
            print(npy_file + ' loaded.')

class SarsaAgent(object):
    def __init__(self,
                 obs_n,
                 act_n,
                 learning_rate=0.01,
                 gamma=0.9,
                 e_greed=0.1):
        self.act_n = act_n  # actions
        self.lr = learning_rate  # learning rate
        self.gamma = gamma  # reward discount rate
        self.epsilon = e_greed  # low prob. for random actions
        self.Q = np.zeros((obs_n, act_n))
         
    # sample action while training
    def sample(self, obs):
        if np.random.uniform(0, 1) < (1.0 - self.epsilon):  # pick action based on table
            action = self.predict(obs)
        else:
            action = np.random.choice(self.act_n)  # make sure a low prob. to pick a random action
        return action

    # predict actions based on observations
    def predict(self, obs):
        Q_list = self.Q[obs, :]
        maxQ = np.max(Q_list)
        action_list = np.where(Q_list == maxQ)[0]  # actions are corresponding to maxQ
        action = np.random.choice(action_list)
        return action

    # learning algorithm (updating Q table)
    def learn(self, obs, action, reward, next_obs, next_action, done):
        """ 
        on-policy
        obs: observation before interaction, s_t
        action: action picked for this interaction, a_t
        reward: reward got, r_t+1
        next_obs: observation after interaction, s_t+1
        next_action: next action corresponding to next obs based on current table, a_t+1
        done: is episode done
        """
        predict_Q = self.Q[obs, action]
        if done:
            target_Q = reward  # achieve the goal
        else:
            target_Q = reward + self.gamma * self.Q[next_obs, next_action]  # Sarsa
        self.Q[obs, action] += self.lr * (target_Q - predict_Q)  # modify Q

    # save Q-table to numpy file
    def save(self, env_name=None):
        if env_name is None:
            npy_file = 'q_table.npy'
            np.save(npy_file, self.Q)
            print(npy_file + ' saved.')
            
        else:
            npy_file = 'qtable_{}.npy'.format(env_name)
            np.save(npy_file, self.Q)
            print(npy_file + ' saved.')

    # restore Q-table for agent
    def restore(self, npy_file=None):
        if npy_file is None:
            npy_file='q_table.npy'
            self.Q = np.load(npy_file)
            print(npy_file + ' loaded.')
        else:
            self.Q = np.load(npy_file)
            print(npy_file + ' loaded.')

class SarsaLambdaAgent(object):
    def __init__(self,
                 obs_n,
                 act_n,
                 learning_rate=0.1, # lr = alpha in some algorithm annotation
                 gamma=0.9,
                 lambda_=0.9,
                 e_greed=0.1):
        self.act_n = act_n  # actions
        self.lr = learning_rate  # learning rate
        self.gamma = gamma  # reward discount rate
        self.lambda_ = lambda_ # decaying rate for eligibility traces
        self.epsilon = e_greed  # low prob. for random actions
        self.Q = np.zeros((obs_n, act_n)) # table for recording state-action table
        self.E = np.zeros((obs_n, act_n)) # table for recording eligibility traces
         
    # sample action while training (might change to other decaying epsilon strategy later)
    def sample(self, obs):
        if np.random.uniform(0, 1) < (1.0 - self.epsilon):  # pick action based on table
            action = self.predict(obs)
        else:
            action = np.random.choice(self.act_n)  # make sure a low prob. to pick a random action
        return action

    # predict actions based on observations
    def predict(self, obs):
        Q_list = self.Q[obs, :]
        maxQ = np.max(Q_list)
        action_list = np.where(Q_list == maxQ)[0]  # actions are corresponding to maxQ
        action = np.random.choice(action_list)
        return action

    # learning algorithm (updating Q table)
    def learn(self, obs, action, reward, next_obs, next_action, done):
        """ 
        backwards view for Sarsa lambda
        obs: observation before interaction, s_t
        action: action picked for this interaction, a_t
        reward: reward got, r_t+1
        next_obs: observation after interaction, s_t+1
        next_action: next action corresponding to next obs based on current table, a_t+1
        done: is episode done
        """
        predict_Q = self.Q[obs, action]
        if done:
            target_Q = reward  # achieve the goal
        else:
            target_Q = reward + self.gamma * self.Q[next_obs, next_action]  # Sarsa
        sarsa_error = target_Q - predict_Q

        # E(s, a) <- E(s, a) + 1
        self.E[obs, action] += 1

        for i in range(self.E.shape[0]):
            for j in range(self.E.shape[1]):
                self.Q[i, j] += self.lr * sarsa_error * self.E[i, j]
                self.E[i, j] = self.gamma * self.lambda_ * self.E[i, j]

    # save Q-table to numpy file
    def save(self, env_name=None):
        if env_name is None:
            npy_file = 'q_table.npy'
            np.save(npy_file, self.Q)
            print(npy_file + ' saved.')
            
        else:
            npy_file = 'qtable_{}.npy'.format(env_name)
            np.save(npy_file, self.Q)
            print(npy_file + ' saved.')

    # restore Q-table for agent
    def restore(self, npy_file=None):
        if npy_file is None:
            npy_file='q_table.npy'
            self.Q = np.load(npy_file)
            print(npy_file + ' loaded.')
        else:
            self.Q = np.load(npy_file)
            print(npy_file + ' loaded.')