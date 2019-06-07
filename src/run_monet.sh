#! /bin/sh

python main.py --dataset room --seed 2 --lr 3e-4 --beta1 0.9 --beta2 0.999 \
    --objective H --model H --batch_size 64 --z_dim 16 --max_iter 120000 \
    --C_stop_iter 1e5 --C_max 20 --gamma 100 --beta 1 --viz_name dsprites_monet_finalM7 \
    --dset_dir /raid1/hbku1/CLEVR_v1.0/