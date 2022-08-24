import sys
sys.setrecursionlimit(15000)

import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
from torch.autograd import Variable
import torchvision.models as models

epsilon = 0.000000000001


class VggExtractor(nn.Module):
    def __init__(self, train=False, freeze_gradient=True, total_used_layers=18):
        super(VggExtractor, self).__init__()

        self.vgg_1 = self.Vgg(models.vgg19(pretrained=True), 0, total_used_layers)
        if train:
            self.vgg_1.train(mode=True)
            if freeze_gradient:
                self.freeze_gradient()
        else:
            self.vgg_1.eval()

    def Vgg(self, vgg, begin, end):
        features = nn.Sequential(*list(vgg.features.children())[begin:(end+1)])
        return features

    def freeze_gradient(self, begin=0, end=9):
        for i in range(begin, end+1):
            self.vgg_1[i].requires_grad = False

    def forward(self, input):
        return self.vgg_1(input)


class CapsuleLayer(nn.Module):
    def __init__(self, num_capsules=2, num_route_nodes=-1, in_channels=256, out_channels=32, kernel_size=9, stride=2,
                 num_iterations=3, capsules=None):
        super(CapsuleLayer, self).__init__()

        self.num_route_nodes = num_route_nodes
        self.num_iterations = num_iterations

        self.num_capsules = num_capsules

        if num_route_nodes != -1:
            self.route_weights = nn.Parameter(torch.randn(num_capsules, num_route_nodes, in_channels, out_channels))
        else:
            self.capsules = nn.ModuleList(
                [nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=0) for _ in
                 range(num_capsules)])

    def squash(self, tensor, dim=-1):
        squared_norm = (tensor ** 2).sum(dim=dim, keepdim=True)
        scale = squared_norm / (1 + squared_norm + epsilon)
        return scale * tensor / torch.sqrt(squared_norm + epsilon)

    def forward(self, x):
        if self.num_route_nodes != -1:
            priors = x[None, :, :, None, :] @ self.route_weights[:, None, :, :, :]

            logits = Variable(torch.zeros(*priors.size())).cuda()
            for i in range(self.num_iterations):
                probs = self._softmax(logits, dim=2)
                outputs = self.squash((probs * priors).sum(dim=2, keepdim=True))

                if i != self.num_iterations - 1:
                    delta_logits = (priors * outputs).sum(dim=-1, keepdim=True)
                    logits = logits + delta_logits
        else:
            outputs = [capsule(x).view(x.size(0), -1, 1) for capsule in self.capsules]  # astea sunt ELEMENZTE DE CAPSULE??????
            outputs = torch.cat(outputs, dim=-1)
            outputs = self.squash(outputs)

        return outputs

    def _softmax(self, input, dim=1):
        transposed_input = input.transpose(dim, len(input.size()) - 1)
        softmaxed_output = F.softmax(transposed_input.contiguous().view(-1, transposed_input.size(-1)), dim=-1)
        return softmaxed_output.view(*transposed_input.size()).transpose(dim, len(input.size()) - 1)


class CapsuleNet(nn.Module):
    def __init__(self, architecture, dataset, freeze_gradient_extractor=True):
        super(CapsuleNet, self).__init__()

        self.architecture = architecture
        self.dataset = dataset
        self.cl = CapsuleLayer()

        if self.architecture == 'capsule_timecaps':
            self.conv1 = VggExtractor(freeze_gradient=freeze_gradient_extractor)

            self.primary_capsules = CapsuleLayer(num_capsules=8, num_route_nodes=-1, in_channels=256, out_channels=32,
                                                 kernel_size=9, stride=4)
            self.secondary_capsules = CapsuleLayer(num_capsules=16, num_route_nodes=2048, in_channels=8,
                                               out_channels=16)
            self.out_layers = nn.Sequential(nn.Linear(256, 128), nn.ReLU())

            self.time_capsules = CapsuleLayer(num_capsules=8, num_route_nodes=32, in_channels=128,
                                              out_channels=16)

            self.out_fc = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 1), nn.Sigmoid())

        elif self.architecture == 'capsule_timecaps_simple' or self.architecture == 'capsule_timecaps_simple_test':

            self.conv1 = VggExtractor(freeze_gradient=freeze_gradient_extractor)
            self.conv2 = nn.Conv2d(in_channels=256, out_channels=32, kernel_size=7, stride=4)
            self.time_capsules = CapsuleLayer(num_capsules=16, num_route_nodes=32, in_channels=2048,
                                              out_channels=32)
            self.out_fc = nn.Sequential(nn.Linear(16*32, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 1), nn.Sigmoid())
            self.out_fc_2 = nn.Sequential(nn.Linear(2048, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 1),
                                        nn.Sigmoid())


        else:
            raise ValueError("Not a known architecture!")

    def forward(self, x):

        if self.architecture == 'capsule_timecaps':

            x = F.relu(self.conv1(x), inplace=True)
            x = self.primary_capsules(x)
            x = self.secondary_capsules(x)
            x = x.squeeze().transpose(0, 1)
            x = x.reshape(x.shape[0], -1)

            primary_time_capsules = self.out_layers(x)
            primary_time_capsules = primary_time_capsules.reshape(1,
                                                                  primary_time_capsules.shape[0],
                                                                  primary_time_capsules.shape[1])
            primary_time_capsules = self.cl.squash(tensor=primary_time_capsules)
            secondary_time_capsules = self.time_capsules(primary_time_capsules).squeeze()
            out = self.out_fc(secondary_time_capsules.reshape(-1))

            return out

        elif self.architecture == 'capsule_timecaps_simple':

            x = F.relu(self.conv1(x), inplace=True)
            x = F.relu(self.conv2(x))
            x = x.reshape(x.shape[0], -1)

            primary_time_capsules = x.reshape(1, x.shape[0], x.shape[1])
            primary_time_capsules = self.cl.squash(tensor=primary_time_capsules)

            secondary_time_capsules = self.time_capsules(primary_time_capsules).squeeze()
            out = self.out_fc(secondary_time_capsules.reshape(-1))

            return out

        elif self.architecture == 'capsule_timecaps_simple_test':

            x = F.relu(self.conv1(x), inplace=True)
            x = F.relu(self.conv2(x))
            x = x.reshape(x.shape[0], -1)

            out = self.out_fc_2(x).mean()

            return out


class CapsuleLoss(nn.Module):

    def __init__(self):
        super(CapsuleLoss, self).__init__()

    def forward(self, labels, classes):

        len_images = labels.size(0)
        left = F.relu(0.9 - classes, inplace=True) ** 2
        right = F.relu(classes - 0.1, inplace=True) ** 2

        margin_loss = labels * left + 0.5 * (1. - labels) * right
        margin_loss = margin_loss.sum()

        return (margin_loss) / len_images


if __name__ == "__main__":

    a = VggExtractor()