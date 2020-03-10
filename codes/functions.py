import numpy as np
import pandas as pd
import os
import torch
from tqdm import tqdm
from collections import defaultdict
from sklearn.model_selection import train_test_split
import sklearn.metrics

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

data_folder = "../data"
def load_train_df():

    train_df = pd.read_csv(os.path.join(data_folder, 'train.csv'))

    print("Loading train parquets...")
    train_parquet_files = []
    for i in tqdm(range(4)):
        train_parquet_files.append(pd.read_parquet(os.path.join(data_folder,'train_image_data_{}.parquet'.format(i))))

    train_parquet_all = pd.concat(train_parquet_files, ignore_index=True)

    train_all = train_parquet_all.merge(train_df, on="image_id")

    return train_all

def load_test_df():
    test_df = pd.read_csv(os.path.join(data_folder, 'test.csv'))
    class_map_df = pd.read_csv(os.path.join(data_folder, 'class_map.csv'))

    print("Loading test parquets...")
    test_parquet_files = []
    for i in tqdm(range(4)):
        test_parquet_files.append(pd.read_parquet(os.path.join(data_folder,'test_image_data_{}.parquet'.format(i))))

    test_parquet_all = pd.concat(test_parquet_files, ignore_index=True)

    test_all = test_parquet_all.merge(test_df, on="image_id")

    return test_all

# evaluation metrics
def calc_hierarchical_macro_recall(label1_true, label1_pred,
                                   label2_true, label2_pred,
                                   label3_true, label3_pred,
                                   ):
    # label1:vowel_disacritic
    # label2:grapheme_root
    # label3:consonant_diacritic
    # these are defined in dataset.py

    scores = []

    vowel_recall = sklearn.metrics.recall_score(label1_true, label1_pred, average='macro')
    grapheme_recall = sklearn.metrics.recall_score(label2_true, label2_pred, average='macro')
    consonant_recall = sklearn.metrics.recall_score(label3_true, label3_pred, average='macro')

    scores.append(vowel_recall)
    scores.append(grapheme_recall)
    scores.append(consonant_recall)

    final_score = np.average(scores, weights=[1,2,1])
    return vowel_recall, grapheme_recall, consonant_recall, final_score

def plot_train_history(history, figure_name):
    # plot training log.
    # history : dictionary

    num_plots = len(history)

    num_plots = len(history.keys())
    fig = plt.figure(figsize=(14,4*num_plots))

    for i, (name, log_list) in enumerate(history.items()):

        if isinstance(log_list, list):
            if i > 0:
                ax = fig.add_subplot(num_plots, 1, i+1, sharex=ax)
            else:
                ax = fig.add_subplot(num_plots, 1, i+1)
            x = [i+1 for i in range(len(log_list))]
            ax.plot(x, log_list, label=name)

        elif isinstance(log_list, dict):
            if i > 0:
                ax = fig.add_subplot(num_plots, 1, i+1, sharex=ax)
            else:
                ax = fig.add_subplot(num_plots, 1, i+1)

            for subname, content in log_list.items():
                x = [i+1 for i in range(len(content))]
                ax.plot(x, content, label=subname)

        ax.grid(axis="x")
        if "AUC" in name or "auc" in name:
            ax.set_ylim([0.0, 1.05])
            ax.axhline(y=0.5, xmin=0, xmax=1, color='black')

        ax.set_xlabel("Epoch")
        ax.set_ylabel(name)
        ax.legend(loc='upper right')

    plt.savefig(figure_name)
    plt.close()


def cutmix_aug(img_batch, sl=0.2, sh=0.5, r1=0.3, r2=3.3):
    rand_index = torch.randperm(img_batch.size()[0]).to(device)

    labels1_a = labels1
    labels2_a = labels2
    labels3_a = labels3

    labels1_b = labels1[rand_index]
    labels2_b = labels2[rand_index]
    labels3_b = labels3[rand_index]

    batchsize, channels, height, width = img_batch.size()

    S = height * width


    # sample box center from uniform dist.
    x = np.random.randint(low=0, high=width)
    y = np.random.randint(low=0, high=height)

    # sampling bbox area and bbox aspect ratio from uniform dist.
    Se = np.random.uniform(sl, sh) * S
    # re = np.random.uniform(r1, r2)

    re = (np.random.beta(0.5, 0.5) * (r2 - r1)) + r1


    # box height and width
    box_h = int(np.sqrt(Se * re))
    box_w = int(np.sqrt(Se / re))

    y1 = np.clip(y - box_h // 2, 0, height)
    y2 = np.clip(y1 + box_h, 0, height)
    x1 = np.clip(x - box_w // 2, 0, width)
    x2 = np.clip(x1 + box_w, 0, width)

    img_batch[:, :, bbx1:bbx2, bby1:bby2] = img_batch[rand_index, :, bbx1:bbx2, bby1:bby2]

    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (inputs.size()[-1] * inputs.size()[-2]))


    return img_batch, lam

def random_erasing_aug(img_batch, sl=0.02, sh=0.4, r1=0.3, r2=3.3, mean=0.0818658566, std=0.22140448):
    """yielding random erased imgs

    Args:
        img_batch: torch.tensor
        sl: minimum ratio of area of bbox
        sh: maximum ratio of area of bbox
        r1: minimum value of aspect ratio of bbox.
        r2: maximum value of aspect ratio of bbox.
        mean: mean value used in normalizing images
        value: standard deviation value used in normalizing images

    Targets:
        image : augmented

    Image types:
        uint8, float32

    Reference:
    |  https://arxiv.org/pdf/1708.04896.pdf
    """

    batchsize, channels, height, width = img_batch.size()

    # sample box center from uniform dist.
    x = np.random.randint(low=0, high=width)
    y = np.random.randint(low=0, high=height)

    S = height * width

    # sampling bbox area and bbox aspect ratio from uniform dist.
    Se = np.random.uniform(sl, sh) * S
    re = np.random.uniform(r1, r2)

    # box height and width
    box_h = int(np.sqrt(Se * re))
    box_w = int(np.sqrt(Se / re))

    y1 = np.clip(y - box_h // 2, 0, height)
    y2 = np.clip(y1 + box_h, 0, height)
    x1 = np.clip(x - box_w // 2, 0, width)
    x2 = np.clip(x1 + box_w, 0, width)

    # fill_value = (np.random.randint(255)/255 - mean) / std
    fill_value = (torch.randint(low=0, high=255, size=(y2-y1, x2-x1))/255 - mean).float() / std
    fill_value = fill_value#.to(device)

    img_batch[:, :, y1:y2, x1:x2] = fill_value
    return img_batch



def cutout_aug(img_batch, max_w, max_h, fill_value=0, random_fill=False, mean=0.0818658566, std=0.22140448):
    """CoarseDropout of the square regions in the image.

    Args:
        img_batch: torch.tensor which is normlized by (X-mean)/std
        max_w(int): box width
        max_h(int): box height
        fill_value(int): pixel value to fill the box.
        random_fill(bool): whether to decide the pixel value randomly, 0~255.
        mean: mean value used in normalizing images
        value: standard deviation value used in normalizing images

    Targets:
        image : augmented

    Image types:
        uint8, float32

    Reference:
    |  https://arxiv.org/abs/1708.04552
    |  https://github.com/uoguelph-mlrg/Cutout/blob/master/util/cutout.py
    |  https://github.com/aleju/imgaug/blob/master/imgaug/augmenters/arithmetic.py
    """
    img_batch_ = img_batch.clone()

    batchsize, channels, height, width = img_batch_.size()

    # sample box center from uniform dist.
    x = np.random.randint(low=0, high=width)
    y = np.random.randint(low=0, high=height)

    y1 = np.clip(y - max_h // 2, 0, height)
    y2 = np.clip(y1 + max_h, 0, height)
    x1 = np.clip(x - max_w // 2, 0, width)
    x2 = np.clip(x1 + max_w, 0, width)


    if random_fill:
        fill_value = (torch.randint(low=0, high=255, size=(y2-y1, x2-x1)) / 255 - mean).float() / std


    img_batch_[:, :, y1:y2, x1:x2] = fill_value
    return img_batch_
