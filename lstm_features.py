import sys
sys.setrecursionlimit(15000)
import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
from torch.autograd import Variable
import torchvision.models as models
from torch.optim import Adam
#from torchnet.engine import Engine
#from torchnet.logger import VisdomPlotLogger, VisdomLogger
from torchvision.utils import make_grid
from torchvision.datasets.mnist import MNIST
from tqdm import tqdm
#import torchnet as tnt


class LSTMNN(nn.Module):

    def __init__(self):
        super(LSTMNN, self).__init__()

        self.lstm = nn.LSTM(input_size=256, hidden_size=256, num_layers=2)
        self.fc1 = nn.Linear(256, 128)
        self.dropout = nn.Dropout(p=0.3)
        self.fc2 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        hidden = None
        for t in range(x.size(1)):

            xc = F.relu(x[:, t, :, :].squeeze())
            xc = xc.reshape(xc.size(0), -1)
            out, hidden = self.lstm(xc.unsqueeze(0), hidden)

        xfc = self.dropout(self.fc1(out[-1, :, :]))
        xfc = F.relu(xfc)
        xfc = self.fc2(xfc)
        xfc_out = self.sigmoid(xfc).squeeze()

        return xfc_out





