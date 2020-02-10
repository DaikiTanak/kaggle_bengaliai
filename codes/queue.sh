#!/bin/bash


python train.py --name  resnet34_augment --model resnet --epoch 50 --lr 0.1
python train.py --name densenet_augment --model densenet --epoch 50 --lr 0.1
