import numpy as np
import pandas as pd
import torch
from torch.utils.data.dataset import Dataset
from tqdm import tqdm
import os
from collections import defaultdict

from functions import load_train_df

data_folder = "../data"

class BengalImgDataset(Dataset):
    """
    Construct dataset given images and labels.
    """

    def __init__(self, images, vowel, grapheme, consonant, component, transform=None, test_dataset_flag=False):
        """
        Params:
            images: numpy.array

        """
        self.transform = transform
        self.images = images
        self.label1 = vowel
        self.label2 = grapheme
        self.label3 = consonant
        self.component = component

        self.height = 137
        self.width = 236


    def __len__(self):
        """Returns the number of data points."""
        return len(self.images)

    def __getitem__(self, idx):
        """Returns an example or a sequence of examples."""
        label1 = self.label1[idx]
        label2 = self.label2[idx]
        label3 = self.label3[idx]
        compo = self.component[idx]
        image = self.images[idx]

        compo_of = torch.zeros((62,))
        for i in compo:
            compo_of[int(i)] = 1

        image = np.tile(image, (1,1,3))

        if self.transform is not None:
            image = self.transform(image)

        return (image,label1,label2,label3, compo_of)

def pickle_images():
    # pickle images, labels
    train_df = pd.read_csv(os.path.join(data_folder, 'train.csv'))
    # test_df = pd.read_csv(os.path.join(data_folder, 'test.csv'))

    print("Pickle train parquets...")
    for parquet_idx in range(4):
        grapheme_root_label = []
        vowel_diacritic_label = []
        consonant_diacritic_label = []
        images = []

        train_parquet = pd.read_parquet(os.path.join(data_folder,'train_image_data_{}.parquet'.format(parquet_idx)))
        df = train_parquet.merge(train_df, on="image_id")

        for i in tqdm(range(len(df))):

            grapheme_root_label.append(int(df["grapheme_root"].iloc[i]))
            vowel_diacritic_label.append(int(df["vowel_diacritic"].iloc[i]))
            consonant_diacritic_label.append(int(df["consonant_diacritic"].iloc[i]))
            image = df.iloc[i].drop(['image_id', "grapheme_root", "vowel_diacritic","consonant_diacritic","grapheme"]).values.astype(np.uint8)
            # H * W * C
            image = image.reshape(137, 236, 1)
            images.append(image)

        fn = os.path.join(data_folder, "train_data_{}.pkl".format(parquet_idx))
        pd.to_pickle((images, vowel_diacritic_label, grapheme_root_label, consonant_diacritic_label), fn)

def load_pickle_images():
    # load pickled train images, labels
    all_images = []
    all_vowel = []
    all_grapheme = []
    all_consonant = []

    for parquet_idx in range(4):
        fn = os.path.join(data_folder, "train_data_{}.pkl".format(parquet_idx))
        images, vowel_diacritic_label, grapheme_root_label, consonant_diacritic_label = pd.read_pickle(fn)
        all_images.extend(images)
        all_vowel.extend(vowel_diacritic_label)
        all_grapheme.extend(grapheme_root_label)
        all_consonant.extend(consonant_diacritic_label)

    return (all_images, all_vowel, all_grapheme, all_consonant)

def label_to_components(vowels, roots, consonants):
    """
    return component label included in input images.
    Params:
        vowels: vowel labels
        roots: root labels
        consonants: consonant labels
    """

    assert len(vowels) == len(roots) == len(consonants)

    class_map_df = pd.read_csv(os.path.join(data_folder, 'class_map.csv'))

    # component to label dictionary.
    compo_label = {}
    i = 0
    for idx, component in enumerate(class_map_df["component"].values):
        component_list = list(component)
        for c in component_list:
            if not c in compo_label.keys():
                compo_label[c] = i
                i += 1

    print("There is " + str(len(compo_label.keys())) + " component parts")


    vowel_label_to_compo = defaultdict(list)
    root_label_to_compo =  defaultdict(list)
    consonant_label_to_compo =  defaultdict(list)

    # root
    for i in range(0, 168):
        l = class_map_df["label"].values[i]
        compo = class_map_df["component"].values[i]
        assert class_map_df["component_type"].values[i] == "grapheme_root", "{} at {}".format(class_map_df["component_type"].values[i],i)
        components = list(set(compo))
        for c in components:
            root_label_to_compo[l].append(compo_label[c])
    #vowel
    for i in range(168, 168+11):
        l = class_map_df["label"].values[i]
        compo = class_map_df["component"].values[i]
        assert class_map_df["component_type"].values[i] == "vowel_diacritic", "{} at {}".format(class_map_df["component_type"].values[i],i)

        components = list(set(compo))
        for c in components:
            vowel_label_to_compo[l].append(compo_label[c])

    #conso
    for i in range(168+11, 168+11+7):
        l = class_map_df["label"].values[i]
        compo = class_map_df["component"].values[i]
        assert class_map_df["component_type"].values[i] == "consonant_diacritic", "{} at {}".format(class_map_df["component_type"].values[i],i)

        components = list(set(compo))
        for c in components:
            consonant_label_to_compo[l].append(compo_label[c])

    labels = []
    for root, vowel, consonant in zip(roots, vowels, consonants):
        labels.append(list(set(root_label_to_compo[root] + vowel_label_to_compo[vowel] + consonant_label_to_compo[consonant])))
    return np.asarray(labels)


import cv2
from tqdm import tqdm_notebook as tqdm
import zipfile
import io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
HEIGHT = 137
WIDTH = 236
SIZE = 128

def bbox(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return rmin, rmax, cmin, cmax

def crop_resize(img0, size=SIZE, pad=16):
    #crop a box around pixels large than the threshold
    #some images contain line at the sides
    ymin,ymax,xmin,xmax = bbox(img0[5:-5,5:-5] > 80)
    #cropping may cut too much, so we need to add it back
    xmin = xmin - 13 if (xmin > 13) else 0
    ymin = ymin - 10 if (ymin > 10) else 0
    xmax = xmax + 13 if (xmax < WIDTH - 13) else WIDTH
    ymax = ymax + 10 if (ymax < HEIGHT - 10) else HEIGHT
    img = img0[ymin:ymax,xmin:xmax]
    #remove lo intensity pixels as noise
    img[img < 28] = 0
    lx, ly = xmax-xmin,ymax-ymin
    l = max(lx,ly) + pad
    #make sure that the aspect ratio is kept in rescaling
    img = np.pad(img, [((l-ly)//2,), ((l-lx)//2,)], mode='constant')
    return cv2.resize(img,(size,size))


def save_prerosessed_imgs(imgs):
    HEIGHT = 137
    WIDTH = 236
    SIZE = 128
    # save images
    cropped_imgs = []
    for idx in tqdm(range(len(imgs))):
        img = imgs[idx]
        #somehow the original input is inverted
        img0 = 255 - imgs[idx]
        img0 = img0.reshape([HEIGHT, WIDTH])

        #normalize each image by its max val
        img1 = (img0*(255.0/img0.max())).astype(np.uint8)
        img1 = crop_resize(img1)
        img1 = img1.reshape([SIZE, SIZE, 1])
        cropped_imgs.append(img1)

    pd.to_pickle(cropped_imgs, "../data/cropped_imgs.pkl")
    return

if __name__ == "__main__":
    # save all train images.
    # train_df = load_train_df()
    # _ = BengalDataset(df=train_df, transform=transforms, read_pickle=False, save_images=True)

    # pickle_images()
    load_pickle_images()
