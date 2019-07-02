#! /bin/sh

python main.py --dataset room --seed 2 --lr 1e-4 --beta1 0.9 --beta2 0.999 \
    --objective H --model H --batch_size 8 --z_dim 16 --max_iter 1200000 \
    --C_stop_iter 1e5 --C_max 20 --gamma 100 --beta 0.5 --viz_name monet_gx3 \
    --dset_dir ../data/CLEVR_v1.0/