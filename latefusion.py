import torch
import time
import os
import copy
from data_loader import DeepFakeDataset
from torchvision import transforms
from cnnlstm import CNNLSTM
import torch.nn as nn
import numpy as np
from util import AverageMeter
from sklearn.metrics import roc_auc_score
import pandas as pd


class Network(nn.Module):

    def __init__(self, nr_hidden_layers=3, starting_pow=6, in_size=3, add_dropout=True, dropout_p=0.7):
        super(Network, self).__init__()

        layers = [nn.Linear(in_size, 2**starting_pow), nn.ReLU()]
        for i in range(nr_hidden_layers):
            if not add_dropout:
                layers = layers + [nn.Linear(2**(starting_pow-i), 2**(starting_pow - i - 1)), nn.ReLU()]
            else:
                layers = layers + [nn.Linear(2 ** (starting_pow - i), 2 ** (starting_pow - i - 1)),
                                   nn.ReLU(), nn.Dropout(p=dropout_p)]
        layers = layers + [nn.Linear(2 ** (starting_pow - nr_hidden_layers), 1)]
        self.layers = nn.Sequential(*layers)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        x = self.layers(x)
        out = self.sigmoid(x)

        return out


results_param_dict = {'FF': [r'E:\saved_model\FF_raw_mouth\latefusion_FF_mouth_224_177.xlsx',
                           r'E:\saved_model\FF_raw_eyes\latefusion_FF_eyes_224_1725.xlsx',
                           r'E:\saved_model\FF_raw_nose\latefusion_FF_nose_224_1744.xlsx'],
                    'celebDF' : [r'E:\saved_model\CelebDF_mouth\latefusion_celebDF_mouth_224_206.xlsx',
                                 r'E:\saved_model\CelebDF_eyes\latefusion_celebDF_eyes_224_216.xlsx',
                                 r'E:\saved_model\CelebDF_nose\latefusion_celebDF_nose_224_2210.xlsx']
                    }

test_results_param_dict = {'FF': [r'E:\saved_model\FF_raw_mouth\outputs_test_mouth_epoch_12_519_533.xlsx',
                                  r'E:\saved_model\FF_raw_eyes\outputs_test_eyes_epoch_12_319_630.xlsx',
                                  r'E:\saved_model\FF_raw_nose\outputs_test_nose_epoch_14_919_543.xlsx'],
                    'celebDF' :  [r'E:\saved_model\CelebDF_mouth\outputs_test_mouth_epoch_12_celebDF_214_528.xlsx',
                                  r'E:\saved_model\CelebDF_eyes\outputs_test_eyes_epoch_14_celebDF_214_2026.xlsx',
                                  r'E:\saved_model\CelebDF_nose\outputs_test_nose_epoch_12_celebDF_224_1129.xlsx']
                        }

dataset = 'celebDF'
dataset_adr = r'E:\saved_img'
train_file_path = r'train_test_combined.xlsx'

# Hyperparameters
################################################################
lr = 1e-3
lr_decay = 0.9
nr_epochs = 100
train_batch_size = 32
test_batch_size = 4

# NN parameters
nr_hidden_layers = 5
starting_pow = 10
add_dropout = False
dropout_p = 0.8
##################################################################


# Train data
train_mouth = pd.read_excel(results_param_dict[dataset][0])
train_eyes = pd.read_excel(results_param_dict[dataset][1])
train_nose = pd.read_excel(results_param_dict[dataset][2])

data_train_in = torch.Tensor(np.stack([train_mouth['prediction'], train_eyes['prediction'], train_nose['prediction']], axis=1))
data_labels = torch.Tensor(train_mouth['GT'])

# Test data
test_mouth = pd.read_excel(test_results_param_dict[dataset][0])
test_eyes = pd.read_excel( test_results_param_dict[dataset][1])
test_nose = pd.read_excel (test_results_param_dict[dataset][2])

data_test_in = torch.Tensor(np.stack([test_mouth['Pred'], test_eyes['Pred'], test_nose['Pred']], axis=1))
test_data_labels = torch.Tensor(test_mouth['GT'])


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Device: ', device)

model = Network(nr_hidden_layers=nr_hidden_layers, starting_pow=starting_pow, add_dropout=add_dropout, dropout_p=dropout_p)
model.to(device)
print(model)

params = list(model.parameters())
optimizer = torch.optim.Adam(params, lr=lr)

criterion = nn.BCELoss()

for epoch in range(nr_epochs):
    print('EPOCH: ', epoch)

    targets_vect = []
    predictions_vect = []
    auc_train = 0
    auc_test = 0

    model.train()
    for i in range(int(len(data_labels)/train_batch_size)):

        in_data = data_train_in[i*train_batch_size:(i+1)*train_batch_size].to(device)
        targets = data_labels[i*train_batch_size:(i+1)*train_batch_size].reshape((train_batch_size, 1)).to(device)
        out_data = model(in_data)

        loss = criterion(out_data, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        targets_vect.append(targets.to('cpu'))
        predictions_vect.append(out_data.to('cpu').detach())

        if len(torch.unique(torch.stack(targets_vect).flatten())) > 1:
            auc_train = roc_auc_score(torch.stack(targets_vect).flatten(), torch.stack(predictions_vect).flatten())
        else:
            auc_train = '-'

    torch.save(model.state_dict(),
               os.path.join(r'E:\saved_model', 'latefusion_epoch_' + str(epoch) + '_' + dataset +  '.pkl'))

    # Testing
    targets_vect = []
    predictions_vect = []

    model.eval()
    with torch.no_grad():
        for i in range(int(len(test_data_labels) / test_batch_size)):

            in_data = data_test_in[i * test_batch_size:(i + 1) * test_batch_size].to(device)
            targets = test_data_labels[i * test_batch_size:(i + 1) * test_batch_size].reshape((test_batch_size, 1)).to(device)
            out_data = model(in_data)

            targets_vect.append(targets.to('cpu'))
            predictions_vect.append(out_data.to('cpu').detach())

            if len(torch.unique(torch.stack(targets_vect).flatten())) > 1:
                auc_test = roc_auc_score(torch.stack(targets_vect).flatten(), torch.stack(predictions_vect).flatten())
            else:
                auc_test = '-'


        print('\n\n========================================================' +
                '\n Dataset: ' + dataset +
                '\n TRAIN Epoch: ' + str(epoch) +
                '\n TRAIN AUC total: ' + str(auc_train) +
                '\n TEST Epoch: ' + str(epoch) +
                '\n TEST AUC total: ' + str(auc_test) +
                '\n========================================================\n')

        f = open(r"E:\saved_model\latefusion_results_" + dataset + ".txt", "a")
        f.write('\n\n========================================================' +
                '\n TRAIN Epoch: ' + str(epoch) +
                '\n TRAIN AUC total: ' + str(auc_train) +
                '\n TEST Epoch: ' + str(epoch) +
                '\n TEST AUC total: ' + str(auc_test) +
                '\n========================================================\n')
        f.close()

    lr = lr * lr_decay
    for g in optimizer.param_groups:
        g['lr'] = lr




























