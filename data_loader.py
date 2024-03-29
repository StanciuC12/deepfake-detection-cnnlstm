import copy
import os
import time
import pandas as pd
from PIL import Image
import torch.utils.data
import numpy as np


class DeepFakeDataset(torch.utils.data.Dataset):

    def __init__(self, root_dir, train_file, batch_size=1, transform=None,
                 train=True, image_type='fullface', dataset='all', frames=20, skip=4, image_dim=(3, 299, 299)):

        self.root_dir = root_dir
        self.train = train
        self.train_file_path = train_file
        self.transform = transform
        self.batch_size = batch_size
        self.image_type = image_type
        self.dataset = dataset
        self.label_df = pd.read_excel(os.path.join(self.root_dir, self.train_file_path))
        self.frames = frames
        self.skip = skip
        self.image_dim = image_dim
        if self.train:
            self.label_df = self.label_df[self.label_df['TrainTest'] == 0].reset_index(drop=True)
            self.label_df = self.label_df.iloc[0:int(len(self.label_df) / self.batch_size) * self.batch_size]
        else:
            self.label_df = self.label_df[self.label_df['TrainTest'] == 1].reset_index(drop=True)
            self.label_df = self.label_df.iloc[0:int(len(self.label_df) / self.batch_size) * self.batch_size]

        if dataset != 'all':
            self.label_df = self.label_df[self.label_df['Dataset'] == dataset].reset_index(drop=True)
            self.label_df = self.label_df.iloc[0:int(len(self.label_df) / self.batch_size) * self.batch_size]

        self.label_df_original = copy.deepcopy(self.label_df)
        self.classes = list(self.label_df['ClassId'].unique())

    def __getitem__(self, idx):
        """Return (image, target) after resize and preprocessing."""

        folders = [os.path.join(self.root_dir, x) for x in
                               self.label_df.loc[idx * self.batch_size:(idx + 1) * self.batch_size - 1, 'VideoName'].apply(lambda x: x.split('.')[0])]

        data = []
        labels = []
        for folder in folders:

            empty_folder = False
            if os.path.isdir(folder):
                images = [x for x in os.listdir(folder) if self.image_type in x][::self.skip]
            else:
                images = []

            if len(images) == 0:
                empty_folder = True

            if not empty_folder:
                if len(images) > self.frames:
                    images = images[0:self.frames]

                X_time = []
                for img_name in images:

                    img = os.path.join(folder, img_name)

                    X = np.array(Image.open(img), dtype=np.float32)
                    if np.max(X) > 1:
                        X = X / 255

                    if self.transform:
                        X = self.transform(X)

                    X_time.append(X)

                X_time = torch.stack(X_time)
            else:
                X_time = torch.Tensor([])

            if len(X_time) < self.frames:
                to_add_blank_nr = self.frames - len(X_time)
                blank = torch.ones(self.image_dim) * -1
                to_add_list = [blank] * to_add_blank_nr

                X_time = torch.cat((X_time, torch.stack(to_add_list)))

            data.append(X_time)

        labels = self.label_df.loc[idx * self.batch_size:(idx + 1) * self.batch_size - 1, 'ClassId'].values
        labels = torch.Tensor(labels)
        data = torch.stack(data)

        return data, labels

    def __len__(self):
        """Returns the length of the dataset."""
        return int(len(self.label_df) / self.batch_size)

    def shuffle(self):
        self.label_df = self.label_df.sample(frac=1).reset_index(drop=True)

    def augment_dataset(self, less_class=0):

        big_class = 1 if less_class == 0 else 0
        lesser_class = self.label_df_original[self.label_df_original['ClassId'] == less_class]
        bigger_class = self.label_df_original[self.label_df_original['ClassId'] == big_class]

        min_bigger_class = np.min([len(lesser_class) * 7, len(bigger_class)])
        self.label_df = pd.concat([lesser_class, lesser_class, lesser_class, bigger_class.sample(frac=1).iloc[0:min_bigger_class]],
                                  ignore_index=True).sample(frac=1).reset_index(drop=True)


if __name__ == "__main__":

    from torchvision.transforms import transforms

    dataset_adr = r'F:\ff++\saved_images'  # r'E:\saved_img'
    train_file_path = r'train_test_split.xlsx'
    img_type = 'fullface'

    dataset = 'FF++'
    model_type = 'capsule_features'
    ######################
    lr = 1e-3
    #####################
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
                                 batch_size=2, train=True, image_type=img_type, dataset=dataset, frames=300)

    print('hellu')





