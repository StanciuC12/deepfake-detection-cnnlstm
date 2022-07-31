from data_loader import DeepFakeDataset
import time
from torchvision.transforms import transforms
from functools import partial
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from timeCaps.capsulenet_timecaps import CapsuleNet, CapsuleLoss


if __name__ == "__main__":

    dataset_adr = r'F:\ff++\saved_images'  # r'E:\saved_img'
    train_file_path = r'train_test_split.xlsx'
    img_type = 'fullface'

    dataset = 'FF++'
    model_type = 'capsule_timecaps'
    ######################
    lr = 1e-3
    #####################3
    weight_decay = 0
    nr_epochs = 15
    lr_decay = 0.9
    test_data_frequency = 1
    train_batch_size = 8
    test_batch_size = 8
    gradient_clipping_value = None  # 1
    model_param_adr = None  # r'E:\saved_model\Capsule\celebDF\capsule_low_param_fullface_epoch_14_param_celebDF_271_319.pkl'    # None if new training

    transf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        # Resnet and VGG19 expects to have data normalized this way (because pretrained)
    ])

    data_train = DeepFakeDataset(root_dir=dataset_adr, train_file=train_file_path, transform=transf,
                                 batch_size=1, train=True, image_type=img_type, dataset=dataset, frames=32)

    model = CapsuleNet(architecture=model_type, dataset=dataset)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    params = list(model.parameters())
    optimizer = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    criterion = nn.BCELoss()

    for epoch in range(0, nr_epochs + 1):

        print('Epoch: ', epoch)
        train_loss = 0.0
        predictions_vect = []
        targets_vect = []
        losses = []

        model.train()
        data_train.shuffle()

        for i in range(int(len(data_train))):

            t = time.time()
            data, targets = data_train[i]
            # data = data.reshape(data.shape[0], data.shape[2], data.shape[1], data.shape[3],
            #                     data.shape[4])
            data = torch.squeeze(data)
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

            batch_t = time.time() - t
            train_loss += loss.item()
            losses.append(train_loss)
            avg_loss = np.mean(losses)

            if len(torch.unique(torch.cat(targets_vect).flatten())) > 1:
                auc_train = roc_auc_score(torch.cat(targets_vect).flatten(), torch.cat(predictions_vect).flatten())
            else:
                auc_train = '-'

            print('Minibatch: ' + str(i) + '/' + str(len(data_train)) + ' Loss: ' + str(avg_loss) +
                  ' AUC total: ' + str(auc_train), ' Time elapsed:', batch_t * len(data_train) / 3600)
            train_loss = 0.0
