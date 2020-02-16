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
from model import se_resnet34, se_resnet152, densenet121
from functions import load_train_df, plot_train_history,calc_hierarchical_macro_recall
from config import args

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

# train models for each label
if args.model == "resnet34":
    # model = se_resnet34(num_classes=2).to(device)
    model1 = se_resnet34(num_classes=11, multi_output=False).to(device)
    model2 = se_resnet34(num_classes=168, multi_output=False).to(device)
    model3 = se_resnet34(num_classes=7, multi_output=False).to(device)
elif args.model == "resnet152":
    model1 = se_resnet152(num_classes=11, multi_output=False).to(device)
    model2 = se_resnet152(num_classes=168, multi_output=False).to(device)
    model3 = se_resnet152(num_classes=7, multi_output=False).to(device)
elif args.model == "densenet":
    model = densenet121(if_selayer=True).to(device)
    mode2 = densenet121(if_selayer=True).to(device)
    mode3 = densenet121(if_selayer=True).to(device)



# train_all = load_train_df()
_, vowels, graphemes, consonants = load_pickle_images()
imgs = np.asarray(pd.read_pickle(os.path.join(data_folder, "cropped_imgs.pkl")))
# convert into 3-dim images
imgs = np.tile(imgs, (1,1,1,3))
# imgs = imgs[:,:,:,0]
print("#imgs: ", imgs.shape)

# train_info, val_info = train_test_split(train_all, test_size=0.3, random_state=seed, shuffle=True)
train_imgs, val_imgs, train_vowels, val_vowels, train_graphemes, val_graphemes, train_consonants, val_consonants = train_test_split(imgs, vowels, graphemes, consonants,
                                                                                                                                    test_size=0.2,
                                                                                                                                    random_state=seed,
                                                                                                                                    shuffle=True)


# ----------------------------------------------------------------------------------------------------
# set up dataset, models, optimizer
mean = 0.0818658566
std = 0.22140448
transforms = torchvision.transforms.Compose([torchvision.transforms.ToPILImage(mode=None),
                                             # torchvision.transforms.RandomRotation(degrees=5,),
                                             torchvision.transforms.RandomAffine(degrees=5, translate=(0.02, 0.02), scale=(0.85, 1.15), shear=None, resample=False, fillcolor=0),
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
optimizer1 = torch.optim.SGD(model1.parameters(), lr=lr, momentum=0.9, nesterov=True, dampening=0, weight_decay=0.0005)
optimizer2 = torch.optim.SGD(model2.parameters(), lr=lr, momentum=0.9, nesterov=True, dampening=0, weight_decay=0.0005)
optimizer3 = torch.optim.SGD(model3.parameters(), lr=lr, momentum=0.9, nesterov=True, dampening=0, weight_decay=0.0005)
#scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20], gamma=0.1)
scheduler1 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer1, mode='min', factor=0.33, patience=5, verbose=False, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=1e-5, eps=1e-08)
scheduler2 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer2, mode='min', factor=0.33, patience=5, verbose=False, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=1e-5, eps=1e-08)
scheduler3 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer3, mode='min', factor=0.33, patience=5, verbose=False, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=1e-5, eps=1e-08)

#scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=1e-4, max_lr=0.05)
loss_fn = torch.nn.CrossEntropyLoss()
# ----------------------------------------------------------------------------------------------------

logger = defaultdict(list)
val_best_loss = 1e+10

for epoch_idx in range(1, epoch_num+1, 1):
    # scheduler.step()

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
            beta = .1

            lam = np.random.beta(beta, beta)
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
            out1 = model1(inputs)
            out2 = model2(inputs)
            out3 = model3(inputs)

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

            # compute output
            out1 = model1(inputs)
            out2 = model2(inputs)
            out3 = model3(inputs)

            loss1 = loss_fn(out1, labels1_a) * lam + loss_fn(out1, labels1_b) * (1.0 - lam)
            loss2 = loss_fn(out2, labels2_a) * lam + loss_fn(out2, labels2_b) * (1.0 - lam)
            loss3 = loss_fn(out3, labels3_a) * lam + loss_fn(out3, labels3_b) * (1.0 - lam)


        else:
            # compute output
            out1 = model1(inputs)
            out2 = model2(inputs)
            out3 = model3(inputs)

            loss1 = loss_fn(out1, labels1)
            loss2 = loss_fn(out2, labels2)
            loss3 = loss_fn(out3, labels3)


        optimizer1.zero_grad()
        loss1.backward()
        optimizer1.step()

        optimizer2.zero_grad()
        loss2.backward()
        optimizer2.step()

        optimizer3.zero_grad()
        loss3.backward()
        optimizer3.step()

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

            # compute output
            out1 = model1(inputs)
            out2 = model2(inputs)
            out3 = model3(inputs)

            loss1 = loss_fn(out1, labels1)
            loss2 = loss_fn(out2, labels1)
            loss3 = loss_fn(out3, labels1)

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

    logger["train_loss"].append(np.mean(epoch_logger["train_loss"]))
    logger["val_loss"].append(np.mean(epoch_logger["val_loss"]))

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


    slack.notify(text="Epoch:{} train loss:{} val loss:{} train recall:{} val recall{}".format(epoch_idx,
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
        history = {"loss1":{"train":logger["train_loss1"],
                           "validation":logger["val_loss1"]},
                   "loss2":{"train":logger["train_loss2"],
                                      "validation":logger["val_loss2"]},
                   "loss3":{"train":logger["train_loss3"],
                                      "validation":logger["val_loss3"]},
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
