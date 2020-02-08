import os
import json
import argparse

def save_config(args, config_savepath):

    with open(config_savepath, "w") as f:
        json.dump(args.__dict__, f, indent=4)
    return

parser = argparse.ArgumentParser()

parser.add_argument('--gpu', default=0, type=int, required=False)
parser.add_argument('--epoch', default=150, type=int, required=False)
parser.add_argument('--batchsize', default=128, type=int, required=False)
parser.add_argument("--lr", default=1e-3, type=float, required=False)
parser.add_argument('--seed', default=2, type=int, required=False)
parser.add_argument("--model", default="resnet", type=str, required=False)

args = parser.parse_args()
