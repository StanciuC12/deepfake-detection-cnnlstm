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


model_param_dict = {'FF': [r'E:\saved_model\FF_raw_mouth\mouth_epoch_12_param_519_5043.pkl',
                           r'E:\saved_model\FF_raw_eyes\eyes_epoch_12_param_319_46.pkl',
                           r'E:\saved_model\FF_raw_nose\nose_epoch_14_param_919_318.pkl'],
                    'celebDF' : [r'E:\saved_model\CelebDF_mouth\mouth_epoch_12_param_celebDF_214_459.pkl',
                                 r'E:\saved_model\CelebDF_eyes\eyes_epoch_14_param_celebDF_214_2012.pkl',
                                 r'E:\saved_model\CelebDF_nose\nose_epoch_12_param_celebDF_224_119.pkl']
                    }

for dataset in ['celebDF']:
    for img_type in ['mouth', 'eyes', 'nose']:

        print('ACUM FACEM ', dataset, img_type)
        dataset_adr = r'E:\saved_img'
        train_file_path = r'train_test_combined.xlsx'

        idx = 8472
        if img_type == 'mouth':
            idx = 0
        if img_type == 'eyes':
            idx = 1
        if img_type == 'nose':
            idx = 2

        model_param_adr = model_param_dict[dataset][idx]


        transf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

        data_train = DeepFakeDataset(root_dir=dataset_adr, train_file=train_file_path, transform=transf, batch_size=32,
                                     train=True, image_type=img_type, dataset=dataset)

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print('Device: ', device)

        model = CNNLSTM(pretrained=True, finetuning=True)
        model.to(device)

        if model_param_adr:
            model.load_state_dict(torch.load(model_param_adr))


        train_loss = 0.0
        losses = AverageMeter()
        accuracies = AverageMeter()
        times = AverageMeter()
        test_losses = AverageMeter()
        predictions_vect = []
        targets_vect = []
        prediction_df = copy.copy(data_train.label_df)

        model.eval()
        data_train.shuffle()
        with torch.no_grad():
            for i in range(len(data_train)):

                t = time.time()
                data, targets = data_train[i]
                data, targets = data.to(device), targets.to(device)

                outputs_gpu = model(data)
                outputs = outputs_gpu.to('cpu').flatten()
                targets = targets.to('cpu')

                predictions_vect.append(outputs.detach())
                targets_vect.append(targets)

                batch_t = time.time() - t
                times.update(batch_t, 1)

                print('Batch ', i+1, '/', len(data_train), 'Time remaining: ', int((times.avg * (len(data_train) - i - 1)) // 60), 'min')

        try:
            prediction_df['GT'] = torch.stack(targets_vect).flatten().numpy()
            prediction_df['prediction'] = torch.stack(predictions_vect).flatten().numpy()
            folder = '\\'.join(model_param_adr.split('\\')[0:-1])
            prediction_df.to_excel(
                os.path.join(folder, 'latefusion_' + dataset + '_' + img_type + '_' +
                str(time.gmtime()[2]) + str(time.gmtime()[1]) + '_' + str(time.gmtime()[3]) + str(
                    time.gmtime()[4]) + '.xlsx'), index=False)
        except:
            print('NU AM PUTUT SALVA BOSULE')
            f = open(r"E:\saved_model\errors.txt", "a")
            f.write('ERROR: ' + dataset + '  ' + img_type)
            f.close()
