#!/bin/bash


python train.py --name  resnet_augment --model resnet --epoch 100
python train.py --name densenet_augment --model densenet --epoch 100
