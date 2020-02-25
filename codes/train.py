import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from collections import defaultdict
from sklearn.model_selection import train_test_split
import joblib

import slackweb

import torch
import torchvision
# from torchvision import transforms, utils

from dataset import BengalImgDataset, load_pickle_images
from model import se_resnet34, se_resnet152, densenet121, se_resnext101_32x8d, se_resnext50_32x4d
from functions import load_train_df, plot_train_history,calc_hierarchical_macro_recall
from config import args

# data split
# !pip install iterative-stratification
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

slack = slackweb.Slack(url="https://hooks.slack.com/services/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")

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

# configs
data_folder = "../data"
model_fn = "../models/{}.dat".format(args.name)
result_hist_fn = "../result/{}_train_history.png".format(args.name)
seed = args.seed
epoch_num = args.epoch
batchsize = args.batchsize
lr = args.lr
device = "cuda:0" if torch.cuda.is_available() else "cpu"
height = 137
width = 236
cutmix_prob = 0.1
mixup_prob = 0.1

print("Running device: ", device)
print("batchsize: ", batchsize)

if args.model == "resnet34":
    # model = se_resnet34(num_classes=2).to(device)
    model = se_resnet34(num_classes=2, multi_output=True).to(device)
elif args.model == "resnet152":
    model = se_resnet152(num_classes=2, multi_output=True).to(device)
elif args.model == "resnext50":
    model = se_resnext50_32x4d(num_classes=2, multi_output=True).to(device)
elif args.model == "resnext101":
    model = se_resnext101_32x8d(num_classes=2, multi_output=True).to(device)
elif args.model == "densenet":
    model = densenet121(if_selayer=True).to(device)
else:
    raise ValueError()


# train_all = load_train_df()
_, vowels, graphemes, consonants = load_pickle_images()
imgs = np.asarray(pd.read_pickle(os.path.join(data_folder, "cropped_imgs.pkl")))
# convert into 3-dim images
imgs = np.tile(imgs, (1,1,1,3))
# imgs = imgs[:,:,:,0]
print("#imgs: ", imgs.shape)

# train_imgs, val_imgs, train_vowels, val_vowels, train_graphemes, val_graphemes, train_consonants, val_consonants = train_test_split(imgs, vowels, graphemes, consonants,
#                                                                                                                                     test_size=0.2,
#                                                                                                                                     random_state=seed,
#                                                                                                                                     shuffle=True)

nfolds = 5
mskf = MultilabelStratifiedKFold(n_splits=nfold, random_state=args.seed)
img_idx_list = [i for i in range(len(imgs))]
for i, (train_idx, val_idx) in enumerate(mskf.split(img_idx_list)):

    train_imgs = imgs[train_idx]
    val_imgs = imgs[val_idx]
    train_vowels = vowels[train_idx]
    val_vowels = vowels[val_idx]
    train_graphemes = graphemes[train_idx]
    val_graphemes = graphemes[val_idx]
    train_consonants = consonants[train_idx]
    val_consonants = consonants[val_idx]


# ----------------------------------------------------------------------------------------------------
# set up dataset, models, optimizer
mean = 0.0818658566
std = 0.22140448
transforms = torchvision.transforms.Compose([torchvision.transforms.ToPILImage(mode=None),
                                             # torchvision.transforms.RandomRotation(degrees=5,),
                                             torchvision.transforms.RandomAffine(degrees=8, translate=(0.01, 0.01), scale=(0.95, 1.05), shear=None, resample=False, fillcolor=0),
                                             torchvision.transforms.ToTensor(),
                                             torchvision.transforms.Normalize([mean,mean,mean],[std,std,std])])
                                             # torchvision.transforms.Normalize(mean,std,)])

val_transforms = torchvision.transforms.Compose([torchvision.transforms.ToPILImage(mode=None),
                                             torchvision.transforms.ToTensor(),
                                             torchvision.transforms.Normalize([mean,mean,mean],[std,std,std])])
                                             # torchvision.transforms.Normalize(mean,std)])

# train_dataset = BengalDataset(df=train_info, transform=transforms)
# val_dataset = BengalDataset(df=val_info, transform=transforms)
train_dataset = BengalImgDataset(images=train_imgs,
                                 vowel=train_vowels,
                                 grapheme=train_graphemes,
                                 consonant=train_consonants,
                                 transform=transforms)

val_dataset = BengalImgDataset(images=val_imgs,
                                vowel=val_vowels,
                                grapheme=val_graphemes,
                                consonant=val_consonants,
                                transform=val_transforms)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batchsize, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batchsize, shuffle=False)


# optimizer = torch.optim.Adam(model.parameters(), lr=lr)
optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True, dampening=0, weight_decay=0.0005)
#scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20], gamma=0.1)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.33, patience=5, verbose=False, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=1e-5, eps=1e-08)

#scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=1e-4, max_lr=0.05)
loss_fn = torch.nn.CrossEntropyLoss()
# ----------------------------------------------------------------------------------------------------

logger = defaultdict(list)
val_best_loss = 1e+10

for epoch_idx in range(1, epoch_num+1, 1):

    epoch_logger = defaultdict(list)

    model.train()
    for idx, (inputs, labels1, labels2, labels3) in tqdm(enumerate(train_loader), total=len(train_loader)):
        inputs = inputs[:, 0, :, :].unsqueeze(1)
        inputs = inputs.to(device)

        labels1 = labels1.to(device)
        labels2 = labels2.to(device)
        labels3 = labels3.to(device)

        r = np.random.rand(1)
        if args.cutmix and r < cutmix_prob:
            #beta = .1

            lam = np.random.beta(args.cutmix_alpha, args.cutmix_alpha)
            rand_index = torch.randperm(inputs.size()[0]).to(device)
            labels1_a = labels1
            labels2_a = labels2
            labels3_a = labels3

            labels1_b = labels1[rand_index]
            labels2_b = labels2[rand_index]
            labels3_b = labels3[rand_index]

            bbx1, bby1, bbx2, bby2 = rand_bbox(inputs.size(), lam)
            inputs[:, :, bbx1:bbx2, bby1:bby2] = inputs[rand_index, :, bbx1:bbx2, bby1:bby2]
            # adjust lambda to exactly match pixel ratio
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (inputs.size()[-1] * inputs.size()[-2]))
            # compute output
            out1, out2, out3 = model(inputs)
            # loss = (loss_fn(out1, labels1_a)+loss_fn(out2, labels2_a)+loss_fn(out3, labels3_a)) * lam + (loss_fn(out1, labels1_b)+loss_fn(out2, labels2_b)+loss_fn(out3, labels3_b)) * (1.0 - lam)
            loss1 = loss_fn(out1, labels1_a) * lam + loss_fn(out1, labels1_b) * (1.0 - lam)
            loss2 = loss_fn(out2, labels2_a) * lam + loss_fn(out2, labels2_b) * (1.0 - lam)
            loss3 = loss_fn(out3, labels3_a) * lam + loss_fn(out3, labels3_b) * (1.0 - lam)


        elif args.mixup and r < mixup_prob:
            lam = np.random.beta(args.mixup_alpha, args.mixup_alpha)
            rand_index = torch.randperm(inputs.size()[0]).to(device)

            inputs = lam * inputs + (1 - lam) * inputs[rand_index, :]

            labels1_a = labels1
            labels2_a = labels2
            labels3_a = labels3

            labels1_b = labels1[rand_index]
            labels2_b = labels2[rand_index]
            labels3_b = labels3[rand_index]

            out1, out2, out3 = model(inputs)
            # loss = (loss_fn(out1, labels1_a)+loss_fn(out2, labels2_a)+loss_fn(out3, labels3_a)) * lam + (loss_fn(out1, labels1_b)+loss_fn(out2, labels2_b)+loss_fn(out3, labels3_b)) * (1.0 - lam)
            loss1 = loss_fn(out1, labels1_a) * lam + loss_fn(out1, labels1_b) * (1.0 - lam)
            loss2 = loss_fn(out2, labels2_a) * lam + loss_fn(out2, labels2_b) * (1.0 - lam)
            loss3 = loss_fn(out3, labels3_a) * lam + loss_fn(out3, labels3_b) * (1.0 - lam)

        elif args.augmix:
            pass

        else:
            out1, out2, out3 = model(inputs)
            # loss = loss_fn(out1, labels1) + loss_fn(out2, labels2) + loss_fn(out3, labels3)
            loss1 = loss_fn(out1, labels1)
            loss2 = loss_fn(out2, labels2)
            loss3 = loss_fn(out3, labels3)

        loss = loss1 + loss2 + loss3

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_logger["train_loss1"].append(loss1.item())
        epoch_logger["train_loss2"].append(loss2.item())
        epoch_logger["train_loss3"].append(loss3.item())

        epoch_logger["train_label1"].extend(labels1.cpu().numpy())
        epoch_logger["train_label2"].extend(labels2.cpu().numpy())
        epoch_logger["train_label3"].extend(labels3.cpu().numpy())

        epoch_logger["train_label1_pred"].extend(out1.detach().cpu().numpy())
        epoch_logger["train_label2_pred"].extend(out2.detach().cpu().numpy())
        epoch_logger["train_label3_pred"].extend(out3.detach().cpu().numpy())


    # Validation phase
    with torch.no_grad():
        model.eval()

        for idx, (inputs, labels1, labels2, labels3) in tqdm(enumerate(val_loader), total=len(val_loader)):

            inputs = inputs[:, 0, :, :].unsqueeze(1)
            inputs = inputs.to(device)

            labels1 = labels1.to(device)
            labels2 = labels2.to(device)
            labels3 = labels3.to(device)

            out1, out2, out3 = model(inputs)
            # loss = loss_fn(out1, labels1) + loss_fn(out2, labels2) + loss_fn(out3, labels3)
            loss1 = loss_fn(out1, labels1)
            loss2 = loss_fn(out2, labels2)
            loss3 = loss_fn(out3, labels3)

            epoch_logger["val_loss1"].append(loss1.item())
            epoch_logger["val_loss2"].append(loss2.item())
            epoch_logger["val_loss3"].append(loss3.item())

            epoch_logger["val_label1"].extend(labels1.cpu().numpy())
            epoch_logger["val_label2"].extend(labels2.cpu().numpy())
            epoch_logger["val_label3"].extend(labels3.cpu().numpy())

            epoch_logger["val_label1_pred"].extend(out1.cpu().numpy())
            epoch_logger["val_label2_pred"].extend(out2.cpu().numpy())
            epoch_logger["val_label3_pred"].extend(out3.cpu().numpy())

    # calc macro-averaged recall on validation dataset
    for k in ["train_label1_pred","train_label2_pred","train_label3_pred"]:
        epoch_logger[k] = np.argmax(epoch_logger[k], axis=1)
    for k in ["val_label1_pred","val_label2_pred","val_label3_pred"]:
        epoch_logger[k] = np.argmax(epoch_logger[k], axis=1)

    train_recall1, train_recall2, train_recall3, train_recall = calc_hierarchical_macro_recall(epoch_logger["train_label1"], epoch_logger["train_label1_pred"],
                                                                                                epoch_logger["train_label2"], epoch_logger["train_label2_pred"],
                                                                                                epoch_logger["train_label3"], epoch_logger["train_label3_pred"],)


    val_recall1, val_recall2, val_recall3, val_recall = calc_hierarchical_macro_recall(epoch_logger["val_label1"], epoch_logger["val_label1_pred"],
                                                                                        epoch_logger["val_label2"], epoch_logger["val_label2_pred"],
                                                                                        epoch_logger["val_label3"], epoch_logger["val_label3_pred"],)


    for loss_idx in range(1,4,1):
        logger["train_loss{}".format(loss_idx)].append(np.mean(epoch_logger["train_loss{}".format(loss_idx)]))
        logger["val_loss{}".format(loss_idx)].append(np.mean(epoch_logger["val_loss{}".format(loss_idx)]))

    # weighted loss
    train_losses = [logger["train_loss1"][-1],logger["train_loss2"][-1],logger["train_loss3"][-1]]
    val_losses = [logger["val_loss1"][-1],logger["val_loss2"][-1],logger["val_loss3"][-1]]
    logger["train_loss"].append(np.average(train_losses, weights=[1,2,1]))
    logger["val_loss"].append(np.average(val_losses, weights=[1,2,1]))

    logger["val_recall"].append(val_recall)
    logger["train_recall"].append(train_recall)

    logger["train_recall_label1"].append(train_recall1)
    logger["train_recall_label2"].append(train_recall2)
    logger["train_recall_label3"].append(train_recall3)
    logger["val_recall_label1"].append(val_recall1)
    logger["val_recall_label2"].append(val_recall2)
    logger["val_recall_label3"].append(val_recall3)

    # last_lr = scheduler.get_last_lr()
    # logger["last_lr"].append(last_lr)

    scheduler.step(logger["val_loss"][-1])


    slack.notify(text="{} Epoch:{} train loss:{} val loss:{} train recall:{} val recall{}".format(args.name, epoch_idx,
                                                                                                round(logger["train_loss"][-1], 3),
                                                                                                round(logger["val_loss"][-1], 3),
                                                                                                round(logger["train_recall"][-1], 3),
                                                                                                round(logger["val_recall"][-1], 3)))

    for k, v in logger.items():
        print(k, v[-1])

    if logger["val_loss"][-1] < val_best_loss:
        val_best_loss = logger["val_loss"][-1]

        checkpoint = {"model":model.state_dict(),
                      "oprimizer":optimizer.state_dict(),
                      "cpoch":epoch_idx,
                      "val_best_loss":val_best_loss}
        torch.save(checkpoint, model_fn)

    if epoch_idx % 1 == 0:
        history = {"loss":{"train":logger["train_loss"],
                           "validation":logger["val_loss"]},
                   "recall":{"train_all":logger["train_recall"],
                            "val_all":logger["val_recall"],
                            "train_label1":logger["train_recall_label1"],
                            "train_label2":logger["train_recall_label2"],
                            "train_label3":logger["train_recall_label3"],
                            "val_label1":logger["val_recall_label1"],
                            "val_label2":logger["val_recall_label2"],
                            "val_label3":logger["val_recall_label3"],
                            },
                   # "lr":{"last_lr":logger["last_lr"]}
                   }
        plot_train_history(history, result_hist_fn)
