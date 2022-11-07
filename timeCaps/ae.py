import os

import torch
from data_loader import DeepFakeDataset
import time
from torchvision.transforms import transforms
""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)

        return logits


if __name__ == "__main__":

    dataset_adr = r'F:\ff++\saved_images'  # r'E:\saved_img'
    train_file_path = r'train_test_split.xlsx'
    img_type = 'fullface'

    dataset = 'FF++'
    model_type = 'AE_unet'
    ######################
    lr = 1e-4
    #####################3
    weight_decay = 0
    nr_epochs = 15
    lr_decay = 0.9
    test_data_frequency = 1
    train_batch_size = 8
    test_batch_size = 8
    gradient_clipping_value = None  # 1
    model_param_adr = r'E:\saved_model\AE_unet_fullface_epoch_4_param_FF++_1310_2315.pkl'    # None if new training

    transf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        # Resnet and VGG19 expects to have data normalized this way (because pretrained)
    ])

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    denorm = transforms.Normalize(
        mean=[-m / s for m, s in zip(mean, std)],
        std=[1.0 / s for s in std],
        # always_apply=True,
        # max_pixel_value=1.0
    )

    data_train = DeepFakeDataset(root_dir=dataset_adr, train_file=train_file_path, transform=transf,
                                 batch_size=1, train=True, image_type=img_type, dataset=dataset, frames=32)

    model = UNet(n_channels=3, n_classes=3)
    model.to('cuda')
    #loss = torch.nn.MSELoss().to('cuda')
    loss = torch.nn.L1Loss().to('cuda')

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    epoch_done = 0
    if model_param_adr:
        model.load_state_dict(torch.load(model_param_adr))

    print('Model loaded')

    # for epoch in range(10):
    #     losses = 0
    #     for i in range(len(data_train)):
    #
    #         t = data_train[i][0][0, 0:4, :, 22:278, 22:278]
    #         t = t.to('cuda')
    #         x = model(t)
    #
    #         l = loss(x, t)
    #
    #         optimizer.zero_grad()
    #         l.backward()
    #         optimizer.step()
    #
    #         losses += l
    #
    #         print('Loss', losses/i)
    #
    #     print(f'\n\n\n\n Epoch {epoch}')
    #
    #     # Saving model
    #     torch.save(model.state_dict(),
    #                os.path.join(r'E:\saved_model', model_type + '_' + img_type + '_epoch_' + str(epoch) + '_param_' + dataset + '_' +
    #                             str(time.gmtime()[2]) + str(time.gmtime()[1]) + '_' + str(time.gmtime()[3]) + str(time.gmtime()[4]) + '.pkl'))
    #
    #
    #
    # print('yo')