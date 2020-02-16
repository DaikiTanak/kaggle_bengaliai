#!/bin/bash


#python train.py --name resnet34_augment --model resnet34 --epoch 60 --lr 0.1
#python train.py --name resnet152_augment --model resnet152 --epoch 60 --lr 0.1
#python train.py --name densenet_augment --model densenet --epoch 60 --lr 0.1

python train_multi.py --name multi_resnet34_base --epoch 60 --batchsize 128
python train_multi.py --name multi_resnet34_mixup_alpha.3 --epoch 100 --mixup --mixup_alpha 0.3 --batchsize 128
# python train.py --name resnet34_cutmix_alpha.1 --epoch 100 --cutmix --cutmix_alpha 0.1 --batchsize 256
python train.py --name resnet34_cutup_alpha.3 --epoch 100 --cutmix --cutmix_alpha 0.3 --batchsize 256
