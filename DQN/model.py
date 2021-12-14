import torch.nn as nn
import torch.nn.functional as F

class ValueNet(nn.Module):
    def __init__(self, n_states, n_actions):
        super(ValueNet, self).__init__()
        self.fc1 = nn.Linear(in_features=n_states, out_features=24, bias=False)
        self.fc2 = nn.Linear(in_features=24, out_features=48, bias=False)
        self.fc3 = nn.Linear(in_features=48, out_features=n_actions, bias=False)
    
    # def _create_weights(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
    #             nn.init.uniform_(m.weight, -0.1, 0.1)
                # nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        Q = self.fc3(h2)
        # return q-value for each action
        return Q

