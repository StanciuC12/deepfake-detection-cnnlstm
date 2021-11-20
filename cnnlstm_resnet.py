import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
import torch.nn.functional as F
from torchvision.models import resnet18, resnet101
from Xception_model import xception


class CNNLSTM(nn.Module):

    def __init__(self, pretrained=True, finetuning=False, frozen_params=50):
        super(CNNLSTM, self).__init__()

        self.resnet = models.resnet50(pretrained=pretrained)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential()
        self.fc_xc = nn.Sequential(nn.Linear(num_ftrs, 512), nn.ReLU(), nn.Linear(512, 256), nn.ReLU())

        self.lstm = nn.LSTM(input_size=256, hidden_size=256, num_layers=2)
        self.fc1 = nn.Linear(256, 128)
        self.dropout = nn.Dropout(p=0.7)
        self.fc2 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

        if finetuning:

            i = 0
            for param in self.resnet.parameters():
                i += 0
                if i < frozen_params:
                    param.requires_grad = False
                else:
                    param.requires_grad = True


    def forward(self, x_3d):

        hidden = None
        for t in range(x_3d.size(1)):

            x = self.resnet(x_3d[:, t, :, :, :])
            x = self.fc_xc(x)

            out, hidden = self.lstm(x.unsqueeze(0), hidden)

        x = self.dropout(self.fc1(out[-1, :, :]))
        x = F.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)

        return x


if __name__ == "__main__":

    model = CNNLSTM(pretrained=True)
