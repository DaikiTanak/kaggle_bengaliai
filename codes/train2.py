import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from collections import defaultdict
from sklearn.model_selection import train_test_split
import joblib
import copy
import slackweb
import cv2
import pretrainedmodels
import PIL
import torch
import torchvision
# from torchvision import transforms, utils

from dataset import BengalImgDataset, load_pickle_images, label_to_components

from functions import (
    load_train_df,
    plot_train_history,
    calc_hierarchical_macro_recall,
    cutout_aug,
    random_erasing_aug,
    cutmix_aug
    )
from config import args

# data split
# !pip install iterative-stratification
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

debug = args.debug


slack = slackweb.Slack(url="https://hooks.slack.com/services/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")

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # smaple bbox center point from uniform dist.
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

# configs
data_folder = "../data"
seed = args.seed
epoch_num = args.epoch
batchsize = args.batchsize
lr = args.lr
device = "cuda:{}".format(args.gpu) if torch.cuda.is_available() else "cpu"
height = 137
width = 236
size = args.size
cutmix_prob = 0.5
mixup_prob = 0.1
cutout_prob = 0.5
random_erasing_prob = 0.5

print("Running device: ", device)
print("batchsize: ", batchsize)

efficientnet_name = ""

efficientnets = ["efficientnet-b" + str(i) for i in range(8)]

class PretrainedWrapper(torch.nn.Module):
    """
    wrapping pretrained model
    """


    # wrapped model: https://github.com/Cadene/pretrained-models.pytorch
    def __init__(self, wrapped_model, num_classes=11+168+7):
        super().__init__()
        self.model = wrapped_model
        self.num_classes = num_classes

        dim_feats = self.model.last_linear.in_features
        self.model.last_linear = torch.nn.Linear(dim_feats, num_classes)

    def forward(self, x):
        return self.model(x)

if args.resume:
    fn = "../models/pretrained_resnext50_sgd_plat_cutmix_mixup_6folds_fold1.dat"

    checkpoint = torch.load(fn, map_location=device)

    if args.model == "resnext50":
        base_model = pretrainedmodels.se_resnext50_32x4d(num_classes=1000, pretrained='imagenet')
        model = PretrainedWrapper(wrapped_model=base_model, num_classes=11+168+7)
        # model = pretrainedmodels.se_resnext50_32x4d(num_classes=11+168+7, pretrained=False).to(device)
        model.load_state_dict(checkpoint["model_recall"])

        model = model.to(device)

    else:
        raise ValueError()
else:


    if args.model == "resnext50":
        base_model = pretrainedmodels.se_resnext50_32x4d(num_classes=1000, pretrained='imagenet')
        print("model input:", base_model.input_size)
        model = PretrainedWrapper(wrapped_model=base_model, num_classes=11+168+7).to(device)
        # model = pretrainedmodels.se_resnext50_32x4d(num_classes=11+168+7, pretrained=False).to(device)

    else:
        raise ValueError()


# train_all = load_train_df()
imgs, vowels, graphemes, consonants = load_pickle_images()



mean = 241.52097741088866/255
std = 41.02520206476833/255

print("Use original images.")
imgs = np.asarray(imgs)
resized = []
for idx, img in enumerate(imgs):
    img = img[:,:,0]
    img_ = cv2.resize(img, (size,size), interpolation=cv2.INTER_AREA).reshape(size,size,1)
    resized.append(img_)

imgs = np.asarray(resized)

# calc mean, std
# pixels = []
# for img in imgs[:10000]:
#     pixels.extend(img[:,:,0].flatten())
# print("MEAN:", np.mean(pixels))
# print("STD:", np.std(pixels))
# input()

transforms = torchvision.transforms.Compose([torchvision.transforms.ToPILImage(mode=None),
                                            torchvision.transforms.RandomRotation(degrees=args.affine_rotate, resample=PIL.Image.BILINEAR, expand=False, center=None, fill=255),
                                            torchvision.transforms.ToTensor(),
                                            # torchvision.transforms.RandomAffine(degrees=args.affine_rotate, translate=(args.affine_translate, args.affine_translate), scale=(0.95, 1.05), shear=None, resample=False, fillcolor=0),
                                            torchvision.transforms.Normalize([mean,mean,mean],[std,std,std])])
                                             # torchvision.transforms.Normalize(mean,std,)])

val_transforms = torchvision.transforms.Compose([torchvision.transforms.ToPILImage(mode=None),
                                             torchvision.transforms.ToTensor(),
                                             torchvision.transforms.Normalize([mean,mean,mean],[std,std,std])])


if debug:
    # test on small dataset
    imgs = imgs[:1000]
    vowels = vowels[:1000]
    graphemes = graphemes[:1000]
    consonants = consonants[:1000]


if args.component_labels:
    component_labels = label_to_components(vowels=vowels, roots=graphemes, consonants=consonants)
else:
    # dummy labels
    component_labels = np.asarray([[0]] * len(vowels))

print("#imgs: ", imgs.shape)


# train_imgs, val_imgs, train_vowels, val_vowels, train_graphemes, val_graphemes, train_consonants, val_consonants = train_test_split(imgs, vowels, graphemes, consonants,
#                                                                                                                                     test_size=0.2,
#                                                                                                                                     random_state=seed,
#                                                                                                                                     shuffle=True)

nfold = 6
mskf = MultilabelStratifiedKFold(n_splits=nfold, random_state=args.seed)
img_idx_list = [i for i in range(len(imgs))]
labels = np.hstack((np.reshape(vowels, (len(vowels),1)),
                     np.reshape(graphemes, (len(vowels),1)),
                     np.reshape(consonants, (len(vowels),1))))
print("labels:", labels.shape)

vowels = np.array(vowels)
graphemes = np.array(graphemes)
consonants = np.array(consonants)

for fold_idx, (train_idx, val_idx) in enumerate(mskf.split(img_idx_list, labels)):
    print("Start fold:{}".format(fold_idx+1))

    train_imgs = imgs[train_idx]
    val_imgs = imgs[val_idx]
    train_vowels = vowels[train_idx]
    val_vowels = vowels[val_idx]
    train_graphemes = graphemes[train_idx]
    val_graphemes = graphemes[val_idx]
    train_consonants = consonants[train_idx]
    val_consonants = consonants[val_idx]

    train_component = component_labels[train_idx]
    val_component = component_labels[val_idx]

    if args.resume:
        name = "pretrained_resnext50_sgd_plat_cutmix_mixup_6folds"
        model_fn = "../models/{}_fold{}_restart.dat".format(name, fold_idx+1)
        result_hist_fn = "../result/{}_train_history_fold{}_restart.png".format(name, fold_idx+1)
        logger_fn = "../result/{}_train_log_fold{}_restart.dat".format(name, fold_idx+1)
    else:

        model_fn = "../models/{}_fold{}.dat".format(args.name, fold_idx+1)
        result_hist_fn = "../result/{}_train_history_fold{}.png".format(args.name, fold_idx+1)
        logger_fn = "../result/{}_train_log_fold{}.dat".format(args.name, fold_idx+1)

    # ----------------------------------------------------------------------------------------------------
    # set up dataset, models, optimizer


    # train_dataset = BengalDataset(df=train_info, transform=transforms)
    # val_dataset = BengalDataset(df=val_info, transform=transforms)
    train_dataset = BengalImgDataset(images=train_imgs,
                                     vowel=train_vowels,
                                     grapheme=train_graphemes,
                                     consonant=train_consonants,
                                     component=train_component,
                                     transform=transforms)

    val_dataset = BengalImgDataset(images=val_imgs,
                                    vowel=val_vowels,
                                    grapheme=val_graphemes,
                                    consonant=val_consonants,
                                    component=val_component,
                                    transform=val_transforms)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batchsize, shuffle=True, num_workers=6)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batchsize, shuffle=False, num_workers=6)


    if args.optim == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif args.optim == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True, dampening=0, weight_decay=0.0005)
    #scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 20], gamma=0.1)
    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epoch, eta_min=1e-6)
    elif args.scheduler == "plat":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=args.lr_drop, patience=args.patience, verbose=False, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=1e-8, eps=1e-08)
    else:
        raise ValueError

    loss_fn = torch.nn.CrossEntropyLoss()
    # ----------------------------------------------------------------------------------------------------

    logger = defaultdict(list)
    val_best_loss = 1e+100
    val_best_recall = 0

    for epoch_idx in range(1, epoch_num+1, 1):

        epoch_logger = defaultdict(list)

        model.train()
        for idx, (inputs, labels1, labels2, labels3, component) in tqdm(enumerate(train_loader), total=len(train_loader)):

            # inputs: batchsize * 3 * h * w

            labels1 = labels1.to(device)
            labels2 = labels2.to(device)
            labels3 = labels3.to(device)
            component = component.to(device)

            r = np.random.rand(1)
            if r < 0.5:
                # cutmix or cutout
                lam = np.random.beta(1, 1)
                max_w = int(128*args.cutout_size)
                max_h = int(128*args.cutout_size)

                # augmented_iuputs = cutout_aug(inputs, max_w, max_h, random_fill=args.cutout_random).to(device)
                iuputs, lam, rand_index = cutmix_aug(inputs)

                # followings for cutout
                # out = model(augmented_iuputs)
                #
                # out1 = out[:, :11]
                # out2 = out[:, 11:168+11]
                # out3 = out[:, 11+168:11+168+7]
                # loss1 = loss_fn(out1, labels1)
                # loss2 = loss_fn(out2, labels2)
                # loss3 = loss_fn(out3, labels3)

            else:
                # mixup
                lam = np.random.beta(args.mixup_alpha, args.mixup_alpha)
                rand_index = torch.randperm(inputs.size()[0])

                inputs = lam * inputs + (1 - lam) * inputs[rand_index, :]

            labels1_a = labels1
            labels2_a = labels2
            labels3_a = labels3

            labels1_b = labels1[rand_index]
            labels2_b = labels2[rand_index]
            labels3_b = labels3[rand_index]

            inputs = inputs.to(device)

            out = model(inputs)

            out1 = out[:, :11]
            out2 = out[:, 11:168+11]
            out3 = out[:, 11+168:11+168+7]
            # loss = (loss_fn(out1, labels1_a)+loss_fn(out2, labels2_a)+loss_fn(out3, labels3_a)) * lam + (loss_fn(out1, labels1_b)+loss_fn(out2, labels2_b)+loss_fn(out3, labels3_b)) * (1.0 - lam)
            loss1 = loss_fn(out1, labels1_a) * lam + loss_fn(out1, labels1_b) * (1.0 - lam)
            loss2 = loss_fn(out2, labels2_a) * lam + loss_fn(out2, labels2_b) * (1.0 - lam)
            loss3 = loss_fn(out3, labels3_a) * lam + loss_fn(out3, labels3_b) * (1.0 - lam)


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

            for idx, (inputs, labels1, labels2, labels3, component) in tqdm(enumerate(val_loader), total=len(val_loader)):


                inputs = inputs.to(device)

                out = model(inputs)

                labels1 = labels1.to(device)
                labels2 = labels2.to(device)
                labels3 = labels3.to(device)
                component = component.to(device)

                out1 = out[:, :11]
                out2 = out[:, 11:168+11]
                out3 = out[:, 11+168:11+168+7]

                # loss = loss_fn(out1, labels1) + loss_fn(out2, labels2) + loss_fn(out3, labels3)
                loss1 = loss_fn(out1, labels1)
                loss2 = loss_fn(out2, labels2)
                loss3 = loss_fn(out3, labels3)

                if args.component_labels:
                    loss_component = loss_compo(out_component, component)
                    loss += loss_component
                    epoch_logger["val_loss_compo"].append(loss_component.item())


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

        if args.component_labels:
            logger["train_loss_compo"].append(np.mean(epoch_logger["train_loss_compo"]))
            logger["val_loss_compo"].append(np.mean(epoch_logger["val_loss_compo"]))

        # save weighted loss
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

        if args.scheduler == "cosine":
            scheduler.step()
            print("learning rate:", scheduler.get_lr())
        elif args.scheduler == "plat":
            scheduler.step(logger["val_loss"][-1])


        for k, v in logger.items():
            print(k, v[-1])

        save_flag = False
        if logger["val_recall"][-1] > val_best_recall or epoch_idx == 1:
            val_best_recall = logger["val_recall"][-1]
            model_recall = copy.deepcopy(model).state_dict()
            optim_recall = copy.deepcopy(optimizer).state_dict()

            save_flag = True

        if logger["val_loss"][-1] < val_best_loss or epoch_idx == 1:
            val_best_loss = logger["val_loss"][-1]
            model_loss = copy.deepcopy(model).state_dict()
            optim_loss = copy.deepcopy(optimizer).state_dict()

            save_flag = True

        if save_flag:
            checkpoint = {
                "model":model_loss,
                "model_recall":model_recall,
                "oprimizer":optim_loss,
                "optimizer_recall":optim_recall,
                "epoch":epoch_idx,
                "val_best_loss":val_best_loss,
                "val_best_recall":val_best_recall
                }
            torch.save(checkpoint, model_fn)

        slack.notify(text="{} fold:{} Epoch:{} train loss:{} val loss:{} train recall:{} val recall{} best{}".format(args.name, fold_idx+1, epoch_idx,
                                                                                                            round(logger["train_loss"][-1], 4),
                                                                                                            round(logger["val_loss"][-1], 4),
                                                                                                            round(logger["train_recall"][-1], 4),
                                                                                                            round(logger["val_recall"][-1], 4),
                                                                                                            round(val_best_recall, 4)))



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
            joblib.dump(logger, logger_fn, compress=9)

    if args.full_cv:
        pass
    else:
        # train on single fold.
        break
