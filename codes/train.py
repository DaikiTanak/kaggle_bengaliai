import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from collections import defaultdict
from sklearn.model_selection import train_test_split
import joblib

import slackweb

import torch
from torchvision import transforms, utils


from dataset import BengalDataset
from model import se_resnet34, se_resnet152
from functions import load_train_df, plot_train_history,calc_hierarchical_macro_recall

slack = slackweb.Slack(url="https://hooks.slack.com/services/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")


data_folder = "../data"
seed = 46
epoch_num = 50
batchsize = 128
device = "cuda:0" if torch.cuda.is_available() else "cpu"

print("Running device: ", device)

train_all = load_train_df()
print("train all df shape: ", train_all.shape)

train_info, val_info = train_test_split(train_all, test_size=0.3, random_state=seed, shuffle=True)
print(train_info.shape, val_info.shape)


# ----------------------------------------------------------------------------------------------------
# set up dataset, models, optimizer
transforms = transforms.Compose([transforms.ToPILImage(mode=None),
                                transforms.ToTensor()])

train_dataset = BengalDataset(df=train_info, transform=transforms)
val_dataset = BengalDataset(df=val_info, transform=transforms)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batchsize, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batchsize, shuffle=False)

model = se_resnet34(num_classes=2).to(device)
# model = se_resnet152(num_classes=2).to(device)
optimizer = optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
#scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=1e-4, max_lr=0.05)
loss_fn = torch.nn.CrossEntropyLoss()
# ----------------------------------------------------------------------------------------------------

logger = defaultdict(list)
val_best_loss = 1e+10

for epoch_idx in range(1, epoch_num+1, 1):

    epoch_logger = defaultdict(list)

    model.train()
    for idx, (inputs, labels1, labels2, labels3) in tqdm(enumerate(train_loader), total=len(train_loader)):
        inputs = inputs.to(device)
        labels1 = labels1.to(device)
        labels2 = labels2.to(device)
        labels3 = labels3.to(device)

        out1, out2, out3 = model(inputs)
        loss = loss_fn(out1, labels1) + loss_fn(out2, labels2) + loss_fn(out3, labels3)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_logger["train_loss"].append(loss.item())

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
            inputs = inputs.to(device)
            labels1 = labels1.to(device)
            labels2 = labels2.to(device)
            labels3 = labels3.to(device)

            out1, out2, out3 = model(inputs)
            loss = loss_fn(out1, labels1) + loss_fn(out2, labels2) + loss_fn(out3, labels3)

            epoch_logger["val_loss"].append(loss.item())

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

    train_recall = calc_hierarchical_macro_recall(epoch_logger["train_label1"], epoch_logger["train_label1_pred"],
                                                epoch_logger["train_label2"], epoch_logger["train_label2_pred"],
                                                epoch_logger["train_label3"], epoch_logger["train_label3_pred"],)


    val_recall = calc_hierarchical_macro_recall(epoch_logger["val_label1"], epoch_logger["val_label1_pred"],
                                                epoch_logger["val_label2"], epoch_logger["val_label2_pred"],
                                                epoch_logger["val_label3"], epoch_logger["val_label3_pred"],)

    logger["train_loss"].append(np.mean(epoch_logger["train_loss"]))
    logger["val_loss"].append(np.mean(epoch_logger["val_loss"]))

    logger["val_recall"].append(val_recall)
    logger["train_recall"].append(train_recall)


    slack.notify(text="Epoch:{} train loss:{} val loss:{} train recall:{} val recall{}".format(epoch_idx,
                                                                                                logger["train_loss"][-1],
                                                                                                logger["val_loss"][-1],
                                                                                                logger["train_recall"][-1],
                                                                                                logger["val_recall"][-1]))

    for k, v in logger.items():
        print(k, v[-1])

    if logger["val_loss"][-1] < val_best_loss:
        val_best_loss = logger["val_loss"][-1]

        fn = "../models/model.dat"
        checkpoint = {"model":model.state_dict(),
                      "oprimizer":optimizer.state_dict(),
                      "cpoch":epoch_idx,
                      "val_best_loss":val_best_loss}
        torch.save(checkpoint, fn)

    if epoch_idx % 1 == 0:
        history = {"loss":{"train":logger["train_loss"],
                           "validation":logger["val_loss"]},
                   "recall":{"train":logger["train_recall"],
                            "val":logger["val_recall"]}}
        plot_train_history(history, "../result/train_history.png")
