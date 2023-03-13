import cv2
import numpy as np
import pandas as pd

NYU_CSVs = {'train':'/Users/varvararaluca/Documents/Datasets/nyu_data/data/nyu2_train.csv', 
            'test':'/Users/varvararaluca/Documents/Datasets/nyu_data/data/nyu2_test.csv'}
validation_ratio = 0.1
mini_train_ratio = 0.3

if __name__ == "__main__":
    df = pd.DataFrame(columns = ['image_path', 'label_path', 'dataset_split', 'mini_train','original_dataset'])
    for split in NYU_CSVs:
        path = NYU_CSVs[split]
        df_split = pd.read_csv(path, delimiter=",",names=['image_path','label_path'])
        if split == 'train':
            print("train")
            #we also want a validation set
            valid_split = df_split.sample(frac = validation_ratio)
            df_split = df_split.drop(valid_split.index)

            #we also want a mini train split in order to test the models on that and then let it train on the whole dataset
            mini_train_split = df_split.sample(frac = mini_train_ratio)
            df_split = df_split.drop(mini_train_split.index)

            #recreate our dataframe
            for index, row in df_split.iterrows():
                df = df.append({'image_path' : row['image_path'], 'label_path' : row['label_path'], 'dataset_split' : "train", 'mini_train': False,'original_dataset': "NYUv2"}, ignore_index = True)
            for index, row in mini_train_split.iterrows():
                df = df.append({'image_path' : row['image_path'], 'label_path' : row['label_path'], 'dataset_split' : "train", 'mini_train': True,'original_dataset': "NYUv2"}, ignore_index = True)
            for index, row in valid_split.iterrows():
                df = df.append({'image_path' : row['image_path'], 'label_path' : row['label_path'], 'dataset_split' : "val", 'mini_train': False,'original_dataset': "NYUv2"}, ignore_index = True)
        else:
            print("test")
            for index, row in df_split.iterrows():
                df = df.append({'image_path' : row['image_path'], 'label_path' : row['label_path'], 'dataset_split' : "test", 'mini_train': False,'original_dataset': "NYUv2"}, ignore_index = True) 

    df.to_csv('/Users/varvararaluca/Documents/Datasets/nyu_data/metadata.csv', sep=',',index=False)




