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
from data_loader_pictures_fullvideo import DeepFakeDataset
# import torchfunc
#
# torchfunc.cuda.reset()


dataset_adr = r'F:\saved_celefdf_all' # r'E:\saved_img'
train_file_path = r'train_test_celebdf_corect.xlsx'
img_type = 'fullface'
dataset = 'celebDF'

dataset_model = 'FF++'
model_type = 'capsule_features'
test_batch_size = 1
test_split_batch_size = 4
model_param_adr = r'E:\saved_model\capsule_features_fullface_epoch_6_param_FF++_1410_142.pkl' #r'E:\saved_model\capsule_features_fullface_epoch_7_param_celebDF_172_1647.pkl'

transf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    # Resnet and VGG19 expects to have data normalized this way (because pretrained)
])

data_test = DeepFakeDataset(root_dir=dataset_adr, train_file=train_file_path, transform=transf,
                            batch_size=test_batch_size, train=False, image_type=img_type, dataset=dataset)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Device: ', device)

model = CapsuleNet(architecture=model_type, dataset=dataset_model)
model.to(device)

print("# parameters:", sum(param.numel() for param in model.parameters()))

epoch_done = None
if model_param_adr:
    model.load_state_dict(torch.load(model_param_adr))
    epoch_done = int(model_param_adr.split('_')[-5])


params = list(model.parameters())

criterion = nn.BCELoss()


# Testing

print('Starting TEST')
test_losses = AverageMeter()
losses = AverageMeter()
accuracies = AverageMeter()
times = AverageMeter()
test_predictions_vect = []
test_targets_vect = []
test_df = copy.copy(data_test.label_df)
model.eval()
t_start = time.time()

for i in range(len(data_test)):

    t1 = time.time()
    print(f'{i}/{len(data_test)}')
    data_total, targets = data_test[i]
    outputs_total = []
    loss_total = []

    for j in range(len(data_total)//test_split_batch_size):

        data = data_total[j * test_split_batch_size: (j+1) * test_split_batch_size]
        data, targets = data.to(device), targets.to(device)

        with torch.no_grad():
            try:
                outputs_gpu, _ = model(data)
            except Exception as e:
                print(e)
                print(f'Failed in test i={i}, j={j}')
                continue

            outputs = outputs_gpu.to('cpu').flatten()
            outputs_total.append(outputs.detach())


    outputs_total = torch.cat(outputs_total).flatten().mean()

    # test_losses.update(loss.item(), data.size(0))
    test_predictions_vect.append(outputs_total)
    test_targets_vect.append(targets.to('cpu').detach())

    if len(test_predictions_vect) > 1:
        # print(f'Predictions: {test_predictions_vect}')
        # print(f'Targets: {test_targets_vect}')

        try:
            loss = criterion(torch.stack(test_predictions_vect).flatten(), torch.stack(test_targets_vect).flatten())
            print(f'Loss: {loss}')
        except Exception as e:
            print(e)

    try:
        if len(torch.unique(torch.cat(test_targets_vect).flatten())) > 1:
                auc_test = roc_auc_score(torch.stack(test_targets_vect).flatten(), torch.stack(test_predictions_vect).flatten())
        else:
                auc_test = '-'
    except Exception as e:
        print(e)
        auc_test = 'failed'

    print(f'AUC: {auc_test}')

    t2 = time.time()
    duration_1_sample = t2 - t1
    times.update(duration_1_sample, 1)
    print('Est time/Epoch: ' + str(int(times.avg * (len(data_test)-i) // 3600)) + 'h' +
                      str(int((times.avg * (len(data_test)-i) - 3600 * (times.avg * (len(data_test)-i) // 3600)) // 60)) + 'm')


print('\n\n\n\n========================================================' +
      '\n TEST Epoch: ' + str(epoch_done) + '\n TEST Loss: ' + str(test_losses.avg) +
      '\n TEST AUC total: ' + str(auc_test) +
      '\n========================================================')

f = open(r"E:\saved_model\test_results_" + model_type + '_' + img_type + '_' + dataset + ".txt", "a")
f.write('\n\n\n\n========================================================' +
      '\n' + img_type +
      '\n TEST Epoch: ' + str(epoch_done) + '\n TEST Loss: ' + str(test_losses.avg) +
      '\n TEST AUC total: ' + str(auc_test) +
      '\n========================================================\n')
f.close()

try:
    test_df['GT'] = torch.stack(test_targets_vect).flatten().numpy()
    test_df['Pred'] = torch.stack(test_predictions_vect).flatten().numpy()
    test_df.to_excel(r'E:\saved_model\outputs_test_test_' + model_type + '_' + img_type + '_epoch_' + str(epoch_done) + '_' + dataset + '_' +
                     str(time.gmtime()[2]) + str(time.gmtime()[1]) + '_' + str(time.gmtime()[3]) + str(time.gmtime()[4]) + '.xlsx')

except:
    print('NU A MERS EXCELU PT TEST')




