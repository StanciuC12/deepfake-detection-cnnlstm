import numpy as np
import pandas as pd
import random
import os

train_percent = 0.8
a = []
l = []

adr1 = r'C:\Users\user\Desktop\ML\AI4Media\Datasets\CelebDF\Celeb-real'
adr2 = r'C:\Users\user\Desktop\ML\AI4Media\Datasets\CelebDF\Celeb-synthesis'
adr3 = r'C:\Users\user\Desktop\ML\AI4Media\Datasets\CelebDF\YouTube-real'

dir1 = os.listdir(adr1)
dir2 = os.listdir(adr2)
dir3 = os.listdir(adr3)
real_videos = list(map(lambda x: x.split('.')[0], dir1 + dir3))
fake_videos = list(map(lambda x: x.split('.')[0], dir2))

f = r'E:\saved_img'
for folder in [x for x in os.listdir(f) if '.' not in x][::-1]:

    a.append(folder)
    l.append(0 if folder in real_videos else 1)

train_test_df = pd.DataFrame()
c = list(zip(a, l))
random.shuffle(c)
a, l = zip(*c)
train_test_df['VideoName'] = [''] * len(a)
train_test_df['VideoName'] = a
train_test_df['VideoName'] = train_test_df['VideoName'].astype(str)
train_test_df['ClassId'] = l
train_test_df['TrainTest'] = 0
train_test_df.loc[int(len(train_test_df) * train_percent):len(train_test_df), 'TrainTest'] = 1
train_test_df.to_excel(r'E:\saved_img\train_test_files.xlsx', index=False)
