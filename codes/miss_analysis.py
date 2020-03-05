import torch

from model import (
    se_resnet34,
    densenet121
)

models_path = "../models"

def load_checkpoint(name):

    check = torch.load("../models/"+model, map_location="cuda")
    loss = check["val_best_loss"]

    model = se_resnet34(num_classes=2, multi_output=True).to(device)
    model = densenet121(if_selayer=True).to(device)

    model.load_state_dict(check["model"])
    model = model.to(device)

    model.eval()

    return model


def load_data():
    pass



def model_output(loader):

    for idx, img in loader:
        pass
