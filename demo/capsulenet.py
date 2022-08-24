import sys
sys.setrecursionlimit(15000)

import torch
import torch.nn.functional as F
from torch import nn
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
                 num_iterations=3):
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

        if self.architecture == 'capsule' and self.dataset == 'celebDF':
            self.conv1 = VggExtractor(freeze_gradient=freeze_gradient_extractor)
            self.primary_capsules = CapsuleLayer(num_capsules=8, num_route_nodes=-1, in_channels=256, out_channels=32,
                                                 kernel_size=9, stride=2)
            self.digit_capsules = CapsuleLayer(num_capsules=10, num_route_nodes=32 * 15 * 15, in_channels=8,
                                               out_channels=16)

            self.out_layers = nn.Sequential(nn.Linear(160, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 1), nn.Sigmoid())

        elif self.architecture == 'capsule' and self.dataset == 'FF':
            self.conv1 = VggExtractor(freeze_gradient=freeze_gradient_extractor)
            self.primary_capsules = CapsuleLayer(num_capsules=8, num_route_nodes=-1, in_channels=256, out_channels=32,
                                                 kernel_size=9, stride=2)
            self.digit_capsules = CapsuleLayer(num_capsules=2, num_route_nodes=32 * 15 * 15, in_channels=8,
                                               out_channels=16)

        elif self.architecture == 'capsule-LSTM':
            self.conv1 = VggExtractor(freeze_gradient=freeze_gradient_extractor)
            self.primary_capsules = CapsuleLayer(num_capsules=8, num_route_nodes=-1, in_channels=256, out_channels=32,
                                                 kernel_size=9, stride=4)
            self.digit_capsules = CapsuleLayer(num_capsules=16, num_route_nodes=32 * 8 * 8, in_channels=8,
                                               out_channels=16)
            self.lstm = nn.LSTM(input_size=256, hidden_size=256, num_layers=2)
            self.fc1 = nn.Linear(256, 128)
            self.dropout = nn.Dropout(p=0.3)
            self.fc2 = nn.Linear(128, 1)
            self.sigmoid = nn.Sigmoid()

        elif self.architecture == 'test':
            self.conv1 = VggExtractor(freeze_gradient=freeze_gradient_extractor)
            self.conv2 = nn.Conv2d(in_channels=256, out_channels=32, kernel_size=7, stride=4)
            self.primary_capsules = CapsuleLayer(num_capsules=1024, num_route_nodes=-1, in_channels=32, out_channels=8,
                                                 kernel_size=5, stride=2, num_iterations=5)  # Am adaugata num_iterations=5 mai parziu pt experiment
            self.digit_capsules = CapsuleLayer(num_capsules=16, num_route_nodes=32, in_channels=1024,
                                               out_channels=16, num_iterations=5)
            self.out_layers = nn.Sequential(nn.Linear(16*16, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 1), nn.Sigmoid())


        elif self.architecture == 'capsule_features':
            self.conv1 = VggExtractor(freeze_gradient=freeze_gradient_extractor)
            self.primary_capsules = CapsuleLayer(num_capsules=8, num_route_nodes=-1, in_channels=256, out_channels=32,
                                                 kernel_size=9, stride=2, num_iterations=5)  # Am adaugata num_iterations=5 mai parziu pt experiment
            self.digit_capsules = CapsuleLayer(num_capsules=16, num_route_nodes=32 * 15 * 15, in_channels=8,
                                               out_channels=16, num_iterations=5)
            self.out_layers = nn.Sequential(nn.Linear(16*16, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 1), nn.Sigmoid())

        elif self.architecture == 'capsule_low_param':
            self.conv1 = VggExtractor(freeze_gradient=freeze_gradient_extractor)
            self.conv2 = nn.Conv2d(in_channels=256, out_channels=64, kernel_size=5, stride=2)
            self.primary_capsules = CapsuleLayer(num_capsules=8, num_route_nodes=-1, in_channels=64, out_channels=32,
                                                 kernel_size=5, stride=2, num_iterations=5)
            self.digit_capsules = CapsuleLayer(num_capsules=16, num_route_nodes=1568, in_channels=8,
                                               out_channels=16, num_iterations=5)
            self.out_layers = nn.Sequential(nn.Linear(16*16, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 1), nn.Sigmoid())

        elif self.architecture == r'capsule_features_no_dropout':

            self.conv1 = VggExtractor(freeze_gradient=freeze_gradient_extractor)
            self.primary_capsules = CapsuleLayer(num_capsules=8, num_route_nodes=-1, in_channels=256, out_channels=32,
                                                 kernel_size=9, stride=2)
            self.digit_capsules = CapsuleLayer(num_capsules=16, num_route_nodes=32 * 15 * 15, in_channels=8,
                                               out_channels=16)
            self.out_layers = nn.Sequential(nn.Linear(16*16, 128), nn.ReLU(), nn.Linear(128, 1), nn.Sigmoid())

        elif self.architecture == 'capsule_low_param_8caps':
            self.conv1 = VggExtractor(freeze_gradient=freeze_gradient_extractor)
            self.conv2 = nn.Conv2d(in_channels=256, out_channels=64, kernel_size=5, stride=2)
            self.primary_capsules = CapsuleLayer(num_capsules=8, num_route_nodes=-1, in_channels=64, out_channels=32,
                                                 kernel_size=5, stride=2, num_iterations=5)
            self.digit_capsules = CapsuleLayer(num_capsules=16, num_route_nodes=1568, in_channels=8,
                                               out_channels=8)
            self.out_layers = nn.Sequential(nn.Linear(8*16, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 1), nn.Sigmoid())

        elif self.architecture == 'capsule_lower_param_4caps':  # asta e low param din paper
            # cu weight_decay = 0
            self.conv1 = VggExtractor(freeze_gradient=freeze_gradient_extractor, total_used_layers=10)
            self.conv2 = nn.Sequential(nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=2),
                                       nn.ReLU(), nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=2))
            self.primary_capsules = CapsuleLayer(num_capsules=8, num_route_nodes=-1, in_channels=64, out_channels=16,
                                                 kernel_size=5, stride=2, num_iterations=5)
            self.digit_capsules = CapsuleLayer(num_capsules=8, num_route_nodes=784, in_channels=8,
                                               out_channels=16, num_iterations=5)
            self.out_layers = nn.Sequential(nn.Dropout(0.2), nn.Linear(128, 1), nn.Sigmoid())

        elif self.architecture == 'capsule_lowest_param':
            # cu weight_decay = 0
            self.conv1 = VggExtractor(freeze_gradient=freeze_gradient_extractor, total_used_layers=8)
            self.conv2 = nn.Conv2d(in_channels=128, out_channels=16, kernel_size=7, stride=4)
            self.conv3 = nn.Conv2d(in_channels=16, out_channels=8, kernel_size=5, stride=3)
            self.digit_capsules = CapsuleLayer(num_capsules=8, num_route_nodes=121, in_channels=8,
                                               out_channels=16)
            self.out_layers = nn.Sequential(nn.Dropout(0.2), nn.Linear(128, 1), nn.Sigmoid())

        else:
            raise ValueError("Not a known architecture!")

    def forward(self, x):

        if self.architecture == 'capsule':
            x = F.relu(self.conv1(x), inplace=True)  # TODO: is relu here?
            x = self.primary_capsules(x)
            x = self.digit_capsules(x).squeeze().transpose(0, 1)

            if self.dataset == 'FF':
                if len(x.shape) > 2:
                    classes = (x ** 2).sum(dim=-1) ** 0.5
                else:
                    classes = (x ** 2).sum(dim=-2) ** 0.5
                classes = F.softmax(classes, dim=-1)
                return classes

            if self.dataset == 'celebDF':
                if len(x.shape) > 2:
                    x = x.reshape(x.size(0), -1)
                else:
                    x = x.reshape(-1)
                out = self.out_layers(x)

                return out

        elif self.architecture == 'capsule-LSTM':

            hidden = None
            for t in range(x.size(1)):

                xc = F.relu(self.conv1(x[:, t, :, :, :]))
                xc = self.primary_capsules(xc)
                xc = self.digit_capsules(xc).squeeze().transpose(0, 1)
                xc = xc.reshape(xc.size(0), -1)

                out, hidden = self.lstm(xc.unsqueeze(0), hidden)

            xfc = self.dropout(self.fc1(out[-1, :, :]))
            xfc = F.relu(xfc)
            xfc = self.fc2(xfc)
            xfc_out = self.sigmoid(xfc)

            return xfc_out

        elif self.architecture == 'capsule_low_param' or self.architecture == 'capsule_low_param_8caps' \
                or self.architecture == 'capsule_lower_param_4caps':

            x = F.relu(self.conv1(x), inplace=True)
            x = F.relu(self.conv2(x))
            x = self.primary_capsules(x)
            x = self.digit_capsules(x).squeeze().transpose(0, 1)
            features = x

            if len(x.shape) > 2:
                x = x.reshape(x.size(0), -1)
            else:
                x = x.reshape(-1)
            out = self.out_layers(x)

            return out, features

        elif self.architecture == 'capsule_features' or self.architecture == 'capsule_features_no_dropout':

            x = F.relu(self.conv1(x), inplace=True)  # TODO: is relu here?
            x = self.primary_capsules(x)
            x = self.digit_capsules(x).squeeze().transpose(0, 1)
            features = x

            #  x[0] -iamginea 0; x[0][0] - capsula 0 imaginea 0;
            # x[:, :, 0] - iau feature-ul 0 de la toate imaginile + toate capsulele
            #  torch.Size([16, 7200, 8, 16]) - features
            #torch.Size([16, 18, 256, 16])

            if len(x.shape) > 2:
                x = x.reshape(x.size(0), -1)
            else:
                x = x.reshape(-1)
            out = self.out_layers(x)

            return out, features

        elif self.architecture == 'test':  #  arhitectura normala cu multe capsule mici

            x = F.relu(self.conv1(x), inplace=True)
            x = F.relu(self.conv2(x), inplace=True)
            x = self.primary_capsules(x)
            x = self.digit_capsules(x).squeeze().transpose(0, 1)


            if len(x.shape) > 2:
                x = x.reshape(x.size(0), -1)
            else:
                x = x.reshape(-1)
            out = self.out_layers(x)

            return out, None

        elif self.architecture == 'capsule_lowest_param':

            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))
            # primary capsules
            x = x.reshape(x.shape[0], -1, 8)
            x = self.digit_capsules(x).squeeze().transpose(0, 1)
            features = x

            if len(x.shape) > 2:
                x = x.reshape(x.size(0), -1)
            else:
                x = x.reshape(-1)
            out = self.out_layers(x)

            return out, features


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