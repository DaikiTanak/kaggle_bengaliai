#!/bin/bash


# python train.py --name resnet34_stratified --model resnet34 --epoch 80 --lr 0.1
# python train.py --name resnet34_stratified_cutmix.4 --model resnet34 --cutmix --cutmix_alpha 0.4 --epoch 150 --lr 0.1
# python train.py --name resnet34_stratified_weighted_loss --weighted_loss --model resnet34 --epoch 60 --lr 0.1

python train.py --cutout --cutout_size 0.4 --model resnet34 --name resnet34_stratified_cutout.4 --batchsize 500 --lr 0.1 --cutout --epoch 100
python train.py --cutout --cutout_size 0.6 --model resnet34 --name resnet34_stratified_cutout.6 --batchsize 500 --lr 0.1 --cutout --epoch 100
python train.py --cutout --cutout_size 0.7 --model resnet34 --name resnet34_stratified_cutout.7 --batchsize 500 --lr 0.1 --cutout --epoch 100


#python train.py --name resnet152_augment --model resnet152 --epoch 60 --lr 0.1
#python train.py --name densenet_augment --model densenet --epoch 60 --lr 0.1

# python train_multi.py --name multi_resnet34_base --epoch 60 --batchsize 64
# python train_multi.py --name multi_resnet34_mixup_alpha.3 --epoch 100 --mixup --mixup_alpha 0.3 --batchsize 64
# python train.py --name resnet34_cutmix_alpha.1 --epoch 100 --cutmix --cutmix_alpha 0.1 --batchsize 256
# python train.py --model resnext101 --name resnext101_base --epoch 100 --batchsize 100
# python train.py --model resnext101 --name resnext101_cutmix_alpha.3 --epoch 150 --cutmix --cutmix_alpha 0.3 --batchsize 100
# python train.py --model resnext101 --name resnext101_cutmix_alpha.1 --epoch 150 --cutmix --cutmix_alpha 0.1 --batchsize 100
# python train.py --model resnext101 --name resnext101_mixup_alpha.1 --epoch 150 --mixup --mixup_alpha 0.1 --batchsize 100
# python train.py --model resnext101 --name resnext101_mixup_alpha.3 --epoch 150 --mixup --mixup_alpha 0.3 --batchsize 100
