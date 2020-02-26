import os
import json
import argparse

def save_config(args, config_savepath):

    with open(config_savepath, "w") as f:
        json.dump(args.__dict__, f, indent=4)
    return

parser = argparse.ArgumentParser()

parser.add_argument('--gpu', default=0, type=int, required=False)
parser.add_argument('--epoch', default=30, type=int, required=False)
parser.add_argument('--batchsize', default=300, type=int, required=False)
parser.add_argument("--lr", default=0.1, type=float, required=False)
parser.add_argument('--seed', default=46, type=int, required=False)
parser.add_argument("--model", default="resnet34", type=str, required=False)
parser.add_argument("--name", default="test", type=str, required=False)
parser.add_argument('--cutmix', action='store_true', default=False, required=False, help="cut-mix regularization")
parser.add_argument('--mixup', action='store_true', default=False, required=False, help="mix-up regularization")
parser.add_argument('--cutout', action='store_true', default=False, required=False, help="mix-up regularization")
parser.add_argument('--cutout_size', default=0.5, type=float, required=False)
parser.add_argument('--augmix', action='store_true', default=False, required=False, help="aug-mix regularization")
parser.add_argument("--mixup_alpha", default=0.1, type=float, required=False)
parser.add_argument("--cutmix_alpha", default=0.1, type=float, required=False)
parser.add_argument('--weighted_loss', action='store_true', default=False, required=False)


args = parser.parse_args()
