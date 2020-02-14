#!/bin/bash


#python train.py --name resnet34_augment --model resnet34 --epoch 60 --lr 0.1
#python train.py --name resnet152_augment --model resnet152 --epoch 60 --lr 0.1
#python train.py --name densenet_augment --model densenet --epoch 60 --lr 0.1


python train.py --name densenet_mixup_alpha.3 --epoch 100 --mixup --mixup_alpha 0.3 --model densenet
python train.py --name densenet_mixup_alpha.1 --epoch 100 --mixup --mixup_alpha 0.1 --model densenet
python train.py --name densenet_base --epoch 100 --model densenet
