import torch
import torch.nn as nn
import torch.nn.functional as F


class MNISTModel(nn.Module):
    def __init__(self,cfg):
        super(MNISTModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=cfg.kernel_size, padding=cfg.padding)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=cfg.kernel_size, padding=cfg.padding)
        self.fc1 = nn.Linear(7 * 7 * 64, cfg.hidden_dim)
        self.fc2 = nn.Linear(cfg.hidden_dim, cfg.out_features)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 7 * 7 * 64)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
    

