from datetime import datetime

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
from capsulenet import CapsuleNet, CapsuleLoss
from data_loader_pictures import DeepFakeDataset
# import torchfunc
#
# torchfunc.cuda.reset()

for mt in ['capsule_low_param', 'capsule_low_param_8caps', 'capsule_lower_param_4caps', 'capsule_lowest_param']:

        dataset_adr = r'F:\ff++\saved_images' # r'E:\saved_img'
        train_file_path = r'train_test_split.xlsx' #r'train_test_celebdf_corect.xlsx' #r'train_test_combined_final.xlsx'
        img_type = 'fullface'

        dataset = 'FF++'
        model_type = mt #'capsule_features' #'capsule_lower_param_4caps'
        ######################
        lr = 1e-3
        #####################
        weight_decay = 0
        nr_epochs = 7
        lr_decay = 0.8 #0.8 initial
        test_data_frequency = 1
        train_batch_size = 16
        test_batch_size = 4
        freeze_gradient = False
        gradient_clipping_value = None #1
        model_param_adr = None #r'E:\saved_model\capsule_features_fullface_epoch_3_param_celebDF_172_459.pkl'  #r'E:\saved_model\capsule_features_fullface_epoch_12_param_celebDF_132_208.pkl' #'E:\saved_model\capsule_features_fullface_epoch_13_param_celebDF_132_045.pkl' # None if new training

        transf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            # Resnet and VGG19 expects to have data normalized this way (because pretrained)
        ])

        data_train = DeepFakeDataset(root_dir=dataset_adr, train_file=train_file_path, transform=transf,
                                     batch_size=train_batch_size, train=True, image_type=img_type, dataset=dataset,
                                     nr_images_per_folder=16)
        data_test = DeepFakeDataset(root_dir=dataset_adr, train_file=train_file_path, transform=transf,
                                    batch_size=test_batch_size, train=False, image_type=img_type, dataset=dataset,
                                    nr_images_per_folder=8)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print('Device: ', device)

        model = CapsuleNet(architecture=model_type, dataset=dataset, freeze_gradient_extractor=freeze_gradient)
        model.to(device)

        print("# parameters:", sum(param.numel() for param in model.parameters()))


        epoch_done = 0
        if model_param_adr:
                model.load_state_dict(torch.load(model_param_adr))
                epoch_done = int(model_param_adr.split('_')[-5])  # Number of epoch done is always in that position by convention


        params = list(model.parameters())
        optimizer = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)

        # scheduler = lr_scheduler.ReduceLROnPlateau(
        # 	optimizer, 'min', patience=opt.lr_patience)

        #criterion = CapsuleLoss()
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

                for i in range(int(len(data_train))//2):  # aici era /2 pt ala mare

                        t = time.time()
                        data, targets = data_train[i]
                        data, targets = data.to(device), targets.to(device)

                        try:
                                outputs_gpu, _ = model(data)
                        except Exception as e:
                                print(f'Error {i}', e)
                                continue

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

                        print('Minibatch: ' + str(i) + '/' + str(int(len(data_train)/2)) + ' Loss: ' + str(avg_loss) +
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

                                print(f'{i}/{len(data_test)}')
                                data, targets = data_test[i]
                                data, targets = data.to(device), targets.to(device)

                                with torch.no_grad():
                                        try:
                                                outputs_gpu, _ = model(data)
                                        except:
                                                print(f'Failed in test {i}')
                                                continue
                                        outputs = outputs_gpu.to('cpu').flatten()
                                        targets = targets.to('cpu')
                                        loss = criterion(outputs, targets)
                                        test_losses.update(loss.item(), data.size(0))
                                        test_predictions_vect.append(outputs.detach())
                                        test_targets_vect.append(targets)
                        try:
                                if len(torch.unique(torch.cat(test_targets_vect).flatten())) > 1:
                                        auc_test = roc_auc_score(torch.cat(test_targets_vect).flatten(), torch.cat(test_predictions_vect).flatten())
                                else:
                                        auc_test = '-'
                        except:
                                auc_test = 'failed'

                        print('\n\n\n\n========================================================' +
                              '\n DateTime: ' + str(datetime.now()) +
                              '\n TRAIN Epoch: ' + str(epoch) +'\n TRAIN Loss: ' + str(losses.avg) +
                              '\n TRAIN Accuracy: ' + str(accuracies.avg) + '\n TRAIN AUC total: ' + str(auc_train) +
                              '\n TEST Epoch: ' + str(epoch) + '\n TEST Loss: ' + str(test_losses.avg) +
                              '\n TEST AUC total: ' + str(auc_test) +
                              '\n========================================================')

                        f = open(r"E:\saved_model\results_" + model_type + '_' + img_type + '_' + dataset + ".txt", "a")
                        f.write('\n\n\n\n========================================================' +
                              '\n DateTime: ' + str(datetime.now()) +
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
                              '\n DateTime: ' + str(datetime.now()) +
                              '\n TRAIN Epoch: ' + str(epoch) +
                              '\n TRAIN Loss: ' + str(losses.avg) +
                              '\n TRAIN Accuracy: ' + str(accuracies.avg) +
                              '\n TRAIN AUC total: ' + str(auc_train) +
                              '\n========================================================')

                        f = open(r"E:\saved_model\results_" + model_type + '_' + img_type + '_' + dataset + ".txt", "a")
                        f.write('\n\n\n\n========================================================' +
                                '\n ' + img_type +
                                '\n DateTime: ' + str(datetime.now()) +
                                '\n TRAIN Epoch: ' + str(epoch) +'\n TRAIN Loss: ' + str(losses.avg) +
                                '\n TRAIN Accuracy: ' + str(accuracies.avg) + '\n TRAIN AUC total: ' + str(auc_train) +
                                '\n========================================================\n')
                        f.close()

                lr = lr * lr_decay
                for g in optimizer.param_groups:
                        g['lr'] = lr





