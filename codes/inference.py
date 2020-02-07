import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from collections import defaultdict
from sklearn.model_selection import train_test_split

import torch
from torchvision import transforms, utils


from dataset import BengalDataset
from model import se_resnet34
from functions import load_test_df

batchsize = 64
device = "cuda:0" if torch.cuda.is_available() else "cpu"


transforms = transforms.Compose([transforms.ToPILImage(mode=None),
                                transforms.ToTensor()])

# load
test_all = load_test_df()
test_dataset = BengalDataset(df=test_all, transform=transforms, test_dataset_flag=True)

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batchsize, shuffle=False)

model = se_resnet34(num_classes=2).to(device)
loss_fn = torch.nn.CrossEntropyLoss()

# model load
fn = "../models/model.dat"
checkpoint = torch.load(fn)

model.load_state_dict(checkpoint["model"])
model = model.to(device)
print("model loaded.")

model.eval()
pred1, pred2, pred3 = [], [], []
with torch.no_grad():
    print("start inference...")
    for idx, (inputs, _, _, _) in tqdm(enumerate(test_loader), total=len(test_loader)):

        out1, out2, out3 = model(inputs.to(device))
        pred1.extend(out1.cpu().numpy())
        pred2.extend(out2.cpu().numpy())
        pred3.extend(out3.cpu().numpy())

    p1 = np.argmax(pred1, axis=1)
    p2 = np.argmax(pred2, axis=1)
    p3 = np.argmax(pred3, axis=1)
    print('p1', p1.shape, 'p2', p2.shape, 'p3', p3.shape)

row_id = []
target = []

for i in tqdm(range(len(p1))):
    row_id += [f'Test_{i}_grapheme_root', f'Test_{i}_vowel_diacritic',
               f'Test_{i}_consonant_diacritic']
    target += [p1[i], p2[i], p3[i]]
submission_df = pd.DataFrame({'row_id': row_id, 'target': target})
submission_df.to_csv('submission.csv', index=False)
