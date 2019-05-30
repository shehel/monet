#! /bin/sh

python main.py --dataset dsprites --seed 2 --lr 3e-4 --beta1 0.9 --beta2 0.999 \
    --objective H --model H --batch_size 64 --z_dim 10 --max_iter 120000 \
    --C_stop_iter 1e5 --C_max 20 --gamma 100 --beta 4 --viz_name dsprites_B_gamma100_z10_VAEpaperB4
