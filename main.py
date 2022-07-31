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



model_type = 'Xception-LSTM'
dataset_adr = r'E:\saved_img'
train_file_path = r'train_test_combined.xlsx'
img_type = 'fullface'
dataset= 'FF'
######################
lr = 5e-4
#####################
weight_decay = 1e-5
nr_epochs = 12
lr_decay = 0.9
test_data_frequency = 1
train_batch_size = 128
test_batch_size = 16
model_param_adr = r'E:\saved_model\ResNet-LSTM\FF\resnet-50-LSTM_fullface_epoch_4_param_FF_146_2358.pkl'  # None if new training


if model_type == 'Xception-LSTM':
        from cnnlstm import CNNLSTM
        from data_loader import DeepFakeDataset

        transf = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        frozen_params = 50

elif model_type == 'resnet-50-LSTM':
        from cnnlstm_resnet import CNNLSTM
        from data_loader import DeepFakeDataset

        transf = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # Resnet expects to have data normalized this way
        ])
        frozen_params = 72

elif model_type == 'Xception':

        from cnn import CNN
        from data_loader_pictures import DeepFakeDataset

        transf = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        frozen_params = 50

elif model_type == 'resnet-50':

        from cnn import CNN
        from data_loader_pictures import DeepFakeDataset

        transf = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # Resnet expects to have data normalized this way
        ])
        frozen_params = 72

else:
        raise Exception('Model type needs to be one of the following: resnet-50, resnet-50-LSTM, Xception, Xception-LSTM')


data_train = DeepFakeDataset(root_dir=dataset_adr, train_file=train_file_path, transform=transf, batch_size=train_batch_size, train=True, image_type=img_type, dataset=dataset)
data_test = DeepFakeDataset(root_dir=dataset_adr, train_file=train_file_path, transform=transf, batch_size=test_batch_size, train=False, image_type=img_type, dataset=dataset)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Device: ', device)

if 'LSTM' in model_type:
        model = CNNLSTM(pretrained=True, finetuning=True, frozen_params=frozen_params)
else:
        model = CNN(pretrained=True, finetuning=True, frozen_params=frozen_params, architecture=model_type)

model.to(device)
epoch_done = 0
if model_param_adr:
        model.load_state_dict(torch.load(model_param_adr))
        epoch_done = int(model_param_adr.split('_')[-5])  # Number of epoch done is always in that position by convention


crnn_params = list(model.parameters())
optimizer = torch.optim.Adam(crnn_params, lr=lr, weight_decay=weight_decay)

# scheduler = lr_scheduler.ReduceLROnPlateau(
# 	optimizer, 'min', patience=opt.lr_patience)

criterion = nn.BCELoss()


if epoch_done != 0:
        print('Starting from Epoch ', epoch_done+1)

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
        if dataset == 'celebDF':
                data_train.augment_dataset()
                data_train.shuffle()

        for i in range(len(data_train)):

                t = time.time()
                data, targets = data_train[i]
                data, targets = data.to(device), targets.to(device)

                outputs_gpu = model(data)

                outputs = outputs_gpu.to('cpu').flatten()
                targets = targets.to('cpu')

                loss = criterion(outputs, targets)

                optimizer.zero_grad()
                loss.backward()
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

                print('Minibatch: ' + str(i) + '/' + str(len(data_train))  + ' Loss: ' + str(avg_loss) +
                      ' Acc: ' + str(accuracies.avg) + ' AUC total: ' + str(auc_train) +
                      ' Est time/Epoch: ' + str(int(times.avg * len(data_train) // 3600)) + 'h' +
                      str(int((times.avg * len(data_train) - 3600 * (times.avg * len(data_train) // 3600)) // 60)) + 'm')
                train_loss = 0.0
                # print('Outputs: ', outputs_values)
                # print('Targets: ', targets)

        # Saving model
        torch.save(model.state_dict(),
                   os.path.join(r'E:\saved_model', model_type + '_' + img_type + '_epoch_' + str(epoch) + '_param_' + dataset + '_' +
                                str(time.gmtime()[2]) + str(time.gmtime()[1]) + '_' + str(time.gmtime()[3]) + str(time.gmtime()[4]) + '.pkl'))

        try:
                prediction_df['GT'] = torch.cat(targets_vect).flatten().numpy()
                prediction_df['prediction'] = torch.cat(predictions_vect).flatten().numpy()
                prediction_df.to_excel(r'E:\saved_model\outputs_train_' + model_type + '_' + img_type + '_epoch_' + str(epoch) + '_' + dataset + '_' +
                                       str(time.gmtime()[2]) + str(time.gmtime()[1]) + '_' + str(time.gmtime()[3]) + str(time.gmtime()[4]) +'.xlsx')
        except:
                pass


        # Testing
        if epoch % test_data_frequency == 0:
                print('Starting TEST')

                test_predictions_vect = []
                test_targets_vect = []
                test_df = copy.copy(data_test.label_df)
                model.eval()

                for i in range(len(data_test)):

                        data, targets = data_test[i]
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

                f = open(r"E:\saved_model\results_" + model_type + '_' + img_type + '_' + dataset + ".txt", "a")
                f.write('\n\n\n\n========================================================' +
                      '\n' + img_type +
                      '\n TRAIN Epoch: ' + str(epoch) +'\n TRAIN Loss: ' + str(losses.avg) +
                      '\n TRAIN Accuracy: ' + str(accuracies.avg) + '\n TRAIN AUC total: ' + str(auc_train) +
                      '\n TEST Epoch: ' + str(epoch) + '\n TEST Loss: ' + str(test_losses.avg) +
                      '\n TEST AUC total: ' + str(auc_test) +
                      '\n========================================================\n')
                f.close()

                try:
                        test_df['GT'] = torch.cat(test_targets_vect).flatten().numpy()
                        test_df['Pred'] = torch.cat(test_predictions_vect).flatten().numpy()
                        test_df.to_excel(r'E:\saved_model\outputs_test_' + model_type + '_' + img_type + '_epoch_' + str(epoch) + '_' + dataset + '_' +
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

                f = open(r"E:\saved_model\results_" + model_type + '_' + img_type + '_' + dataset + ".txt", "a")
                f.write('\n\n\n\n========================================================' +
                                '\n ' + img_type +
                              '\n TRAIN Epoch: ' + str(epoch) +'\n TRAIN Loss: ' + str(losses.avg) +
                              '\n TRAIN Accuracy: ' + str(accuracies.avg) + '\n TRAIN AUC total: ' + str(auc_train) +
                              '\n========================================================\n')
                f.close()

        lr = lr * lr_decay
        for g in optimizer.param_groups:
                g['lr'] = lr





