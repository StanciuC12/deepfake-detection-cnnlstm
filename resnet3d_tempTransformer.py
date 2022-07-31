import math
from functools import partial
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score


def get_inplanes():

    return [64, 128, 256, 512]


def conv3x1x1(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=(3, 1, 1),
                     stride=stride,
                     padding=(1, 0, 0),
                     bias=False)


def conv1x1x1(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv3x1x1(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x1x1(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):

    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()

        self.conv1 = conv1x1x1(in_planes, planes)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = conv3x1x1(planes, planes, stride)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = conv1x1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNetTransformerEncoder(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 block_inplanes,
                 n_input_channels=3,
                 no_max_pool=False,
                 shortcut_type='B',
                 widen_factor=1.0,
                 n_classes=2048):
        super().__init__()

        block_inplanes = [int(x * widen_factor) for x in block_inplanes]

        self.in_planes = block_inplanes[0]
        self.no_max_pool = no_max_pool

        self.conv1 = nn.Conv3d(n_input_channels,
                               self.in_planes,
                               kernel_size=(5, 1, 1),
                               stride=(1, 1, 1),
                               padding=(2, 0, 0),
                               bias=False)
        self.bn1 = nn.BatchNorm3d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(1, 5, 5), stride=(1, 4, 4), padding=(0, 1, 1))
        self.maxpool2 = nn.MaxPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0))
        self.maxpool_img = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=(0, 0, 0))
        self.layer1 = self._make_layer(block, block_inplanes[0], layers[0],
                                       shortcut_type)
        self.layer2 = self._make_layer(block,
                                       block_inplanes[1],
                                       layers[1],
                                       shortcut_type,
                                       stride=1)
        self.layer3 = self._make_layer(block,
                                       block_inplanes[2],
                                       layers[2],
                                       shortcut_type,
                                       stride=1)
        self.layer4 = self._make_layer(block,
                                       block_inplanes[3],
                                       layers[3],
                                       shortcut_type,
                                       stride=1)

        self.avgpool = nn.AdaptiveAvgPool3d((None, 1, 1))
        self.fc = nn.Linear(block_inplanes[3] * block.expansion, n_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        #### Transformer encoder part
        self.enc_nn = nn.Linear(2048, 1024)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=1024, nhead=16, dim_feedforward=2048, activation='gelu')
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=6, )

        self.mlp_head = nn.Sequential(nn.Linear(1024, 512), self.relu, nn.Linear(512, 1), nn.Sigmoid())
        # src = torch.rand(10, 32, 512)
        # out = transformer_encoder(src)


    def _downsample_basic_block(self, x, planes, stride):
        out = F.avg_pool3d(x, kernel_size=1, stride=stride)
        zero_pads = torch.zeros(out.size(0), planes - out.size(1), out.size(2),
                                out.size(3), out.size(4))
        if isinstance(out.data, torch.cuda.FloatTensor):
            zero_pads = zero_pads.cuda()

        out = torch.cat([out.data, zero_pads], dim=1)

        return out

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(self._downsample_basic_block,
                                     planes=planes * block.expansion,
                                     stride=stride)
            else:
                downsample = nn.Sequential(
                    conv1x1x1(self.in_planes, planes * block.expansion, stride),
                    nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(
            block(in_planes=self.in_planes,
                  planes=planes,
                  stride=stride,
                  downsample=downsample))
        self.in_planes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):

        lenx = x.shape[0]

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if not self.no_max_pool:
            x = self.maxpool(x)

        x = self.layer1(x)
        x = self.maxpool2(x)
        x = self.maxpool_img(x)
        x = self.layer2(x)
        x = self.maxpool_img(x)
        x = self.layer3(x)
        x = self.maxpool_img(x)
        x = self.layer4(x)

        x = self.avgpool(x)

        positional_encodings = getPositionEncoding(17, 1024)

        class_embedding = torch.ones(lenx, 1, 1024).to("cuda:0")

        x = x.reshape(lenx, 16, 2048)
        x = self.enc_nn(x)
        x = torch.cat([class_embedding, x], dim=1)
        x = x + positional_encodings

        out_transf = self.transformer_encoder(x)[:, 0, :]

        out = self.mlp_head(out_transf)

        return out


def getPositionEncoding(seq_len, d, n=10000):
    P = np.zeros((seq_len, d))
    for k in range(seq_len):
        for i in np.arange(int(d / 2)):
            denominator = np.power(n, 2 * i / d)
            P[k, 2 * i] = np.sin(k / denominator)
            P[k, 2 * i + 1] = np.cos(k / denominator)

    return torch.Tensor(P).to("cuda:0")



def generate_model(model_depth, **kwargs):
    assert model_depth in [10, 18, 34, 50, 101, 152, 200]

    if model_depth == 10:
        model = ResNetTransformerEncoder(BasicBlock, [1, 1, 1, 1], get_inplanes(), **kwargs)
    elif model_depth == 18:
        model = ResNetTransformerEncoder(BasicBlock, [2, 2, 2, 2], get_inplanes(), **kwargs)
    elif model_depth == 34:
        model = ResNetTransformerEncoder(BasicBlock, [3, 4, 6, 3], get_inplanes(), **kwargs)
    elif model_depth == 50:
        model = ResNetTransformerEncoder(Bottleneck, [3, 4, 6, 3], get_inplanes(), **kwargs)
    elif model_depth == 101:
        model = ResNetTransformerEncoder(Bottleneck, [3, 4, 23, 3], get_inplanes(), **kwargs)
    elif model_depth == 152:
        model = ResNetTransformerEncoder(Bottleneck, [3, 8, 36, 3], get_inplanes(), **kwargs)
    elif model_depth == 200:
        model = ResNetTransformerEncoder(Bottleneck, [3, 24, 36, 3], get_inplanes(), **kwargs)

    return model


if __name__ == "__main__":

    from data_loader import DeepFakeDataset
    import time
    from torchvision.transforms import transforms

    dataset_adr = r'F:\ff++\saved_images'  # r'E:\saved_img'
    train_file_path = r'train_test_split.xlsx'
    img_type = 'fullface'

    dataset = 'FF++'
    model_type = 'capsule_features'
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

    model = generate_model(50)
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
            data = data.reshape(data.shape[0], data.shape[2], data.shape[1], data.shape[3],
                                      data.shape[4])
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
                  ' AUC total: ' + str(auc_train), ' Time elapsed:', batch_t * len(data_train)/3600)
            train_loss = 0.0





