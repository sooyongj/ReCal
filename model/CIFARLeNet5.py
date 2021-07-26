import torch.nn as nn
import torch.nn.functional as F


##
## LeNet5
##
class LeNet5(nn.Module):
    
    def __init__(self, n_labels=10):
        super().__init__()

        self.n_labels = n_labels

        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.lin1 = nn.Linear(16 * 5 * 5, 120)
        self.lin2 = nn.Linear(120, 84)
        self.lin3 = nn.Linear(84, n_labels)
        
    def forward(self, x):
        x = self.feature(x)
        x = self.lin3(x)
        return x

    def feature(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2), 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        return x
