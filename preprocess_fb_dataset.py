import os
import pandas as pd
import numpy as np


dfdc_folder = r'E:\DFDC'
test_proportion = 0.8

test_train_dfdc = pd.DataFrame()
for folder in [x for x in os.listdir(dfdc_folder) if '.' not in x]:

    metadata_adr = os.path.join(dfdc_folder, folder, 'metadata.json')
    json = pd.read_json(metadata_adr).T.reset_index(drop=False)
    json.columns = ['VideoName', 'ClassId_name', 'TrainTest', 'original']
    del json['original']
    del json['TrainTest']
    json['ClassId'] = 1 # fake
    json['TrainTest'] = 0  # train
    json.loc[json['ClassId_name'] == 'REAL', 'ClassId'] = 0
    del json['ClassId_name']
    json['folder'] = folder

    test_train_dfdc = pd.concat([test_train_dfdc, json])

test_train_dfdc = test_train_dfdc.reset_index(drop=True)
test_train_dfdc = test_train_dfdc.sample(n=len(test_train_dfdc), random_state=42)
test_train_dfdc = test_train_dfdc.reset_index(drop=True)
test_train_dfdc.loc[int(test_proportion * len(test_train_dfdc)):len(test_train_dfdc), 'TrainTest'] = 1

test_train_dfdc.to_excel(os.path.join(dfdc_folder, 'test_train_dfdc.xlsx'), index=False)


# deleting fake samples so that real samples nr = fake samples nr
delete = True
if delete:
    df = pd.read_excel(r'E:\DFDC\test_train_dfdc.xlsx')
    df['to_delete'] = 0
    nr_of_fake = np.count_nonzero(df['ClassId'] == 1)
    nr_of_real = np.count_nonzero(df['ClassId'] == 0)

    n_to_delete = nr_of_fake - nr_of_real
    sample = df.loc[df['ClassId'] == 1, :].sample(n=n_to_delete).index
    df.loc[sample, 'to_delete'] = 1

    delete_df = df.loc[df['to_delete'] == 1, :]
    len_total = len(delete_df)

    df = df.loc[df['to_delete'] == 0, :]
    df.to_excel(os.path.join(dfdc_folder, 'test_train_dfdc_final.xlsx'), index=False)

    delete_df = delete_df.reset_index()
    for i in range(len(delete_df)):
        try:
            os.unlink(os.path.join(dfdc_folder, delete_df.loc[i, 'folder'], delete_df.loc[i, 'VideoName']))
        except:
            print('sal')
        if i % 100 == 0:
            print(i / len_total)






