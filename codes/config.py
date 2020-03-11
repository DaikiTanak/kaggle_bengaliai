import os
import json
import argparse

def save_config(args, config_savepath):

    with open(config_savepath, "w") as f:
        json.dump(args.__dict__, f, indent=4)
    return

parser = argparse.ArgumentParser()

parser.add_argument('--gpu', default=0, type=int, required=False)
parser.add_argument('--epoch', default=100, type=int, required=False)
parser.add_argument('--batchsize', default=300, type=int, required=False)
parser.add_argument("--lr", default=0.1, type=float, required=False)
parser.add_argument('--seed', default=46, type=int, required=False)
parser.add_argument("--model", default="resnet34", type=str, required=False)
parser.add_argument("--name", default="test", type=str, required=False)
parser.add_argument('--cutmix', action='store_true', default=False, required=False, help="cut-mix regularization")
parser.add_argument('--mixup', action='store_true', default=False, required=False, help="mix-up regularization")
parser.add_argument('--cutout', action='store_true', default=False, required=False, help="mix-up regularization")
parser.add_argument('--cutout_random', action='store_true', default=False, required=False, help="mix-up regularization")
parser.add_argument('--cutout_size', default=0.5, type=float, required=False)
parser.add_argument('--augmix', action='store_true', default=False, required=False, help="aug-mix regularization")
parser.add_argument("--mixup_alpha", default=0.1, type=float, required=False)
parser.add_argument("--cutmix_alpha", default=0.1, type=float, required=False)
parser.add_argument('--full_cv', action='store_true', default=False, required=False)
parser.add_argument("--affine_translate", default=0.01, type=float, required=False, help="shift parameter in affine transformation")
parser.add_argument('--affine_rotate', default=8, type=int, required=False)
parser.add_argument('--affine_scale', default=0.05, type=float, required=False)
parser.add_argument('--random_erasing', action='store_true', default=False, required=False, help="random-erasing regularization")
parser.add_argument('--original', action='store_true', default=False, required=False, help="using original images as inputs")
parser.add_argument('--component_labels', action='store_true', default=False, required=False, help="using original images as inputs")
parser.add_argument('--debug', action='store_true', default=False, required=False, help="using original images as inputs")
parser.add_argument('--lr_drop', default=0.33, type=float, required=False)

# followings are for random-erasing
parser.add_argument("--sl", default=0.02, type=float, required=False)
parser.add_argument("--sh", default=0.4, type=float, required=False)
parser.add_argument("--r1", default=0.3, type=float, required=False)
parser.add_argument("--r2", default=3.3, type=float, required=False)

parser.add_argument("--crop_scale_min", default=1.0, type=float, required=False)

parser.add_argument('--patience', default=5, type=int, required=False)
parser.add_argument('--size', default=128, type=int, required=False)
parser.add_argument("--optim", default="sgd", type=str, required=False)


args = parser.parse_args()
