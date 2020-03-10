#!/bin/bash


# resnet34: batchsize 500
# size 224: 200
# resnext50: 172
# densenet121: 200
# inception-v3: 100
# efficientnet-b4: 128


python train.py --batchsize 200 --name resnet34_cutout.6_shift.5_224 --cutout_random --cutout --cutout_size 0.6 --affine_translate 0.5 --size 224 --epoch 120 --model resnet34 --patience 10
python train.py --batchsize 400 --name resnet34_cutmix_s.2-.5_r.3-3.3 --epoch 120 --model resnet34 --cutmix --sl 0.2 --sh 0.5 --r1 0.3 --r2 3.3 --patience 10
python train.py --batchsize 400 --name resnet34_cutmix_s.2-.5_r.25-4.0 --epoch 120 --model resnet34 --cutmix --sl 0.2 --sh 0.5 --r1 0.25 --r2 4.0 --patience 10


# python train.py --model bengali_resnext50 --name bengali_resnext50_stratified_shift.5_cutout.6_random --batchsize 172 --epoch 150 --affine_translate 0.5 --cutout --cutout_random --cutout_size 0.6 --affine_rotate 11 --gpu 0
