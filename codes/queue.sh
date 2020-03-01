#!/bin/bash


# resnet34: batchsize 500

# python train.py --cutout --cutout_size 0.5 --model resnet34 --name resnet34_stratified_cutout.5 --batchsize 500 --lr 0.1 --epoch 100
# python train.py --cutout --cutout_size 0.8 --model resnet34 --name resnet34_stratified_cutout.8 --batchsize 500 --lr 0.1 --epoch 100
#
# python train.py --random_erasing --model resnet34 --name resnet34_stratified_random_erasing --batchsize 500 --lr 0.1 --epoch 100
#
# python train.py --model resnet34 --name resnet34_stratified_shift.01 --batchsize 500 --lr 0.1 --epoch 100 --affine_translate 0.01
# python train.py --model resnet34 --name resnet34_stratified_shift.1 --batchsize 500 --lr 0.1 --epoch 100 --affine_translate 0.1

# python train.py --cutout --cutout_random --cutout_size 0.6 --model resnet34 --name resnet34_stratified_cutout.6_random --batchsize 500 --lr 0.1 --epoch 100
#
# python train.py --model resnet34 --name resnet34_stratified_shift.2 --batchsize 500 --lr 0.1 --epoch 100 --affine_translate 0.2

python train.py --model resnet34 --name resnet34_stratified_shift.4 --batchsize 500 --lr 0.1 --epoch 100 --affine_translate 0.4
python train.py --model resnet34 --name resnet34_stratified_shift.5 --batchsize 500 --lr 0.1 --epoch 100 --affine_translate 0.5

python train.py --model resnet34 --affine_rotate 5 --name resnet34_stratified_rotate5 --batchsize 500 --lr 0.1 --epoch 100 --affine_translate 0.1
python train.py --model resnet34 --affine_rotate 8 --name resnet34_stratified_rotate8 --batchsize 500 --lr 0.1 --epoch 100 --affine_translate 0.1
python train.py --model resnet34 --affine_rotate 11 --name resnet34_stratified_rotate11 --batchsize 500 --lr 0.1 --epoch 100 --affine_translate 0.1
python train.py --model resnet34 --affine_rotate 15 --name resnet34_stratified_rotate15 --batchsize 500 --lr 0.1 --epoch 100 --affine_translate 0.1

python train.py --model resnet34 --name resnet34_stratified_shift.3_cutout.6_random --batchsize 500 --lr 0.1 --epoch 150 --affine_translate 0.3 --cutout --cutout_random --cutout_size 0.6 --full_cv

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
