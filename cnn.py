import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
import torch.nn.functional as F
from torchvision.models import resnet18, resnet101
from Xception_model import xception


class CNN(nn.Module):

    def __init__(self, pretrained=True, finetuning=False, architecture='Xception', frozen_params=50):
        super(CNN, self).__init__()

        if architecture == 'Xception':
            self.model = xception(pretrained=pretrained)
            self.model.fc = nn.Sequential()
            self.fc_xc = nn.Sequential(nn.Linear(2048, 512), nn.ReLU(), nn.Linear(512, 256), nn.ReLU(),
                                       nn.Dropout(p=0.7), nn.Linear(256, 1), nn.Sigmoid())

        elif architecture == 'resnet-50':
            self.model = models.resnet50(pretrained=pretrained)
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Sequential()
            self.fc_xc = nn.Sequential(nn.Linear(num_ftrs, 512), nn.ReLU(), nn.Linear(512, 256), nn.ReLU(),
                                       nn.Dropout(p=0.7), nn.Linear(256, 1), nn.Sigmoid())

        if finetuning:
            # 154 parameters for XceptionNET
            i = 0
            for param in self.model.parameters():
                i += 0
                if i < frozen_params:
                    param.requires_grad = False
                else:
                    param.requires_grad = True


    def forward(self, x):

        x = self.model(x)
        out = self.fc_xc(x)

        return out


if __name__ == "__main__":

    model = CNN(pretrained=True, architecture='Xception')
    print(model)
    i = 0
    for param in model.named_parameters():
        i += 1
        print(param[0], i)
