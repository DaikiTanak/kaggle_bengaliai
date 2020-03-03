#!/bin/bash


# resnet34: batchsize 500
# densenet121: 200


# python train.py --model resnet34 --name resnet34_stratified_shift.5_cutout.6_random --batchsize 500 --lr 0.1 --epoch 150 --affine_translate 0.5 --cutout --cutout_random --cutout_size 0.6 --affine_rotate 11
# python train.py --model densenet --name densenet_stratified_shift.5 --batchsize 200 --lr 0.1 --epoch 150 --affine_translate 0.5 --affine_rotate 11

python train.py --model resnet34 --name resnet34_stratified_original_imgs --original --batchsize 500 --lr 0.1 --epoch 100

python train.py --model resnet34 --name resnet34_stratified_random_erasing_sl.25_sh_.4 --random_erasing --sl 0.25 --sh 0.4 --batchsize 500 --lr 0.1 --epoch 100
python train.py --model resnet34 --name resnet34_stratified_random_erasing_sl.25_sh_.4_r1_.5_r2_2.0 --r1 0.5 --r2 2.0 --random_erasing --sl 0.25 --sh 0.4 --batchsize 500 --lr 0.1 --epoch 100

python train.py --model resnet34 --name resnet34_stratified_scale0.1 --affine_scale 0.1 --batchsize 500 --lr 0.1 --epoch 100
python train.py --model resnet34 --name resnet34_stratified_scale0.2 --affine_scale 0.2 --batchsize 500 --lr 0.1 --epoch 100
