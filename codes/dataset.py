import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from tqdm import tqdm


class BengalDataset(Dataset):

    def __init__(self, df, transform=None, test_dataset_flag=False):
        self.transform = transform
        self.df = df

        self.height = 137
        self.width = 236

        self.images = []
        self.grapheme_root_label = []
        self.vowel_diacritic_label = []
        self.consonant_diacritic_label = []
        print("setting up dataset...")
        for i in tqdm(range(len(df))):

            if test_dataset_flag:
                self.grapheme_root_label.append(0)
                self.vowel_diacritic_label.append(0)
                self.consonant_diacritic_label.append(0)
                image = df.iloc[i].drop(['image_id', "row_id", "component"]).values.astype(np.uint8)

            else:

                self.grapheme_root_label.append(int(df["grapheme_root"].iloc[i]))
                self.vowel_diacritic_label.append(int(df["vowel_diacritic"].iloc[i]))
                self.consonant_diacritic_label.append(int(df["consonant_diacritic"].iloc[i]))

                image = df.iloc[i].drop(['image_id', "grapheme_root", "vowel_diacritic","consonant_diacritic","grapheme"]).values.astype(np.uint8)
            image = image.reshape(self.height, self.width, 1)
            self.images.append(image)

    def __len__(self):
        """Returns the number of data points."""
        return len(self.images)

    def __getitem__(self, idx):
        """Returns an example or a sequence of examples."""
        label1 = self.vowel_diacritic_label[idx]
        label2 = self.grapheme_root_label[idx]
        label3 = self.consonant_diacritic_label[idx]
        image = self.images[idx]

        if self.transform is not None:
            image = self.transform(image)

        return (image,label1,label2,label3)
