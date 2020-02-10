#!/bin/bash


python train.py --name resnet34_augment --model resnet34 --epoch 60 --lr 0.1
python train.py --name resnet152_augment --model resnet152 --epoch 60 --lr 0.1
python train.py --name densenet_augment --model densenet --epoch 60 --lr 0.1
