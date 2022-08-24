import torch
import time
import os
import copy
from torchvision import transforms
import torch.nn as nn
import numpy as np
from util import AverageMeter
from sklearn.metrics import roc_auc_score
import pandas as pd
from data_loader_pictures import DeepFakeDataset
from capsulenet import CapsuleNet, CapsuleLoss
# import torchfunc
from lstm_features import LSTMNN


def preprocess_features(features):

    for i in range(len(features)):

        feature = features[i]

        if feature.shape[0] > 256:
            feature = feature[0:256]
        if feature.shape[0] < 256:
            feature = torch.cat([feature, torch.zeros(256 - feature.shape[0], feature.shape[1], feature.shape[2])])

        features[i] = feature

    return features

##################################################
lr = 1e-3
weight_decay = 0.001
nr_epochs = 15
lr_decay = 0.2
test_data_frequency = 1
train_batch_size = 16
test_batch_size = 2
gradient_clipping_value = None #1
##################################################

train_data_adr = r'E:\saved_model\features_capsule_features_fullface_epoch_15_celebDF_162_524.pkl'
train_data_targets_adr = r'E:\saved_model\targets_capsule_features_fullface_epoch_15_celebDF_162_525.pkl'
test_data_adr = r'E:\saved_model\features_test_capsule_features_fullface_epoch_15_celebDF_162_917.pkl'
test_data_targets_adr = r'E:\saved_model\targets_test_capsule_features_fullface_epoch_15_celebDF_162_917.pkl'

print('Reading data...')
#train
train_data = torch.load(train_data_adr)
train_data_targets = torch.load(train_data_targets_adr)
#test
test_data = torch.load(test_data_adr)
test_data_targets = torch.load(test_data_targets_adr)

#train
train_data = torch.stack(preprocess_features(train_data))
train_data_targets = torch.stack(train_data_targets).squeeze()
#test
test_data = torch.stack(preprocess_features(test_data))
test_data_targets = torch.stack(test_data_targets).squeeze()
print('Data read')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Device: ', device)

model = LSTMNN()
model.to(device)

print("# parameters:", sum(param.numel() for param in model.parameters()))

epoch_done = 0
criterion = nn.BCELoss()

params = list(model.parameters())
optimizer = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)

for epoch in range(epoch_done + 1, nr_epochs+1):

        print('Epoch: ', epoch)
        train_loss = 0.0
        losses = AverageMeter()
        accuracies = AverageMeter()
        times = AverageMeter()
        test_losses = AverageMeter()
        predictions_vect = []
        targets_vect = []
        prediction_df = pd.DataFrame(columns=['GT', 'prediction'])

        model.train()
        for i in range(int(len(train_data)//train_batch_size)):

                t = time.time()
                data = train_data[i*train_batch_size: (i+1)*train_batch_size]
                targets = train_data_targets[i*train_batch_size: (i+1)*train_batch_size]
                data, targets = data.to(device), targets.to(device)

                outputs_gpu = model(data)

                outputs = outputs_gpu.to('cpu').flatten()
                targets = targets.to('cpu')

                loss = criterion(outputs, targets)

                optimizer.zero_grad()
                loss.backward()
                if gradient_clipping_value:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping_value)
                optimizer.step()

                predictions_vect.append(outputs.detach())
                targets_vect.append(targets)

                outputs_values = copy.copy(outputs.detach())
                outputs[outputs >= 0.5] = 1
                outputs[outputs < 0.5] = 0
                acc = np.count_nonzero(outputs == targets) / len(targets)

                train_loss += loss.item()
                losses.update(loss.item(), data.size(0))
                accuracies.update(acc, data.size(0))

                batch_t = time.time() - t
                times.update(batch_t, 1)
                avg_loss = train_loss

                if len(torch.unique(torch.cat(targets_vect).flatten())) > 1:
                        auc_train = roc_auc_score(torch.cat(targets_vect).flatten(), torch.cat(predictions_vect).flatten())
                else:
                        auc_train = '-'

                print('Minibatch: ' + str(i) + '/' + str(len(train_data)//train_batch_size) + ' Loss: ' + str(avg_loss) +
                      ' Acc: ' + str(accuracies.avg) + ' AUC total: ' + str(auc_train) +
                      ' Est time/Epoch: ' + str(int(times.avg * len(train_data)//train_batch_size // 3600)) + 'h' +
                      str(int((times.avg * len(train_data) - 3600 * (times.avg * len(train_data)//train_batch_size // 3600)) // 60)) + 'm')
                train_loss = 0.0
                # print('Outputs: ', outputs_values)
                # print('Targets: ', targets)

        # Saving model
        torch.save(model.state_dict(),
                   os.path.join(r'E:\saved_model', 'lstm_features_epoch_' + str(epoch) +
                                str(time.gmtime()[2]) + str(time.gmtime()[1]) + '_' + str(time.gmtime()[3]) + str(time.gmtime()[4]) + '.pkl'))

        try:
                prediction_df['GT'] = torch.cat(targets_vect).flatten().numpy()
                prediction_df['prediction'] = torch.cat(predictions_vect).flatten().numpy()
                prediction_df.to_excel(r'E:\saved_model\outputs_train_' + 'lstm_features_epoch_' + str(epoch) +
                                       str(time.gmtime()[2]) + str(time.gmtime()[1]) + '_' + str(time.gmtime()[3]) + str(time.gmtime()[4]) +'.xlsx')
        except:
                pass

        # Testing
        if epoch % test_data_frequency == 0:
                print('Starting TEST')

                test_predictions_vect = []
                test_targets_vect = []
                model.eval()

                for i in range(len(test_data)//test_batch_size):

                        print(f'{i}/{len(test_data)//test_batch_size}')
                        data = test_data[i * test_batch_size: (i + 1) * test_batch_size]
                        targets = test_data_targets[i * test_batch_size: (i + 1) * test_batch_size]
                        data, targets = data.to(device), targets.to(device)

                        with torch.no_grad():

                                outputs_gpu = model(data)
                                outputs = outputs_gpu.to('cpu').flatten()
                                targets = targets.to('cpu')
                                loss = criterion(outputs, targets)
                                test_losses.update(loss.item(), data.size(0))
                                test_predictions_vect.append(outputs.detach())
                                test_targets_vect.append(targets)

                if len(torch.unique(torch.cat(test_targets_vect).flatten())) > 1:
                        auc_test = roc_auc_score(torch.cat(test_targets_vect).flatten(), torch.cat(test_predictions_vect).flatten())
                else:
                        auc_test = '-'

                print('\n\n\n\n========================================================' +
                      '\n TRAIN Epoch: ' + str(epoch) +'\n TRAIN Loss: ' + str(losses.avg) +
                      '\n TRAIN Accuracy: ' + str(accuracies.avg) + '\n TRAIN AUC total: ' + str(auc_train) +
                      '\n TEST Epoch: ' + str(epoch) + '\n TEST Loss: ' + str(test_losses.avg) +
                      '\n TEST AUC total: ' + str(auc_test) +
                      '\n========================================================')

                f = open(r"E:\saved_model\results_features_lstm" + ".txt", "a")
                f.write('\n\n\n\n========================================================' +
                      '\n' 
                      '\n TRAIN Epoch: ' + str(epoch) +'\n TRAIN Loss: ' + str(losses.avg) +
                      '\n TRAIN Accuracy: ' + str(accuracies.avg) + '\n TRAIN AUC total: ' + str(auc_train) +
                      '\n TEST Epoch: ' + str(epoch) + '\n TEST Loss: ' + str(test_losses.avg) +
                      '\n TEST AUC total: ' + str(auc_test) +
                      '\n========================================================\n')
                f.close()

                try:
                        test_df = pd.DataFrame()
                        test_df['GT'] = torch.cat(test_targets_vect).flatten().numpy()
                        test_df['Pred'] = torch.cat(test_predictions_vect).flatten().numpy()
                        test_df.to_excel(r'E:\saved_model\outputs_test_features_lstm_epoch_' + str(epoch) + '_' +
                                         str(time.gmtime()[2]) + str(time.gmtime()[1]) + '_' + str(time.gmtime()[3]) + str(time.gmtime()[4]) + '.xlsx')
                except:
                        print('NU A MERS EXCELU PT TEST')


        else:

                print('\n\n\n\n========================================================' +
                      '\n TRAIN Epoch: ' + str(epoch) +
                      '\n TRAIN Loss: ' + str(losses.avg) +
                      '\n TRAIN Accuracy: ' + str(accuracies.avg) +
                      '\n TRAIN AUC total: ' + str(auc_train) +
                      '\n========================================================')

                f = open(r"E:\saved_model\results_features_lstm" + ".txt", "a")
                f.write('\n\n\n\n========================================================' +
                              '\n ' 
                              '\n TRAIN Epoch: ' + str(epoch) +'\n TRAIN Loss: ' + str(losses.avg) +
                              '\n TRAIN Accuracy: ' + str(accuracies.avg) + '\n TRAIN AUC total: ' + str(auc_train) +
                              '\n========================================================\n')
                f.close()

        lr = lr * lr_decay
        for g in optimizer.param_groups:
                g['lr'] = lr
