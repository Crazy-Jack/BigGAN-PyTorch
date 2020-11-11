#!/bin/bash
python test.py \
--data_root /user_data/tianqinl/Dataset \
--dataset Mysmall_128 --shuffle  --num_workers 32 --batch_size 64 \
--num_G_accumulations 8 --num_D_accumulations 8 \
--num_D_steps 1 --G_lr 1e-4 --D_lr 4e-4 --E_lr 1e-4 --D_B2 0.999 --G_B2 0.999 --E_B2 0.999 \
--lambda_vae_kld 1e-4 \
--G_attn 64 --D_attn 64 \
--G_nl inplace_relu --D_nl inplace_relu \
--SN_eps 1e-6 --BN_eps 1e-5 --adam_eps 1e-6 \
--G_ortho 0.0 \
--G_shared \
--G_init ortho --D_init ortho \
--hier --dim_z 120 --shared_dim 128 \
--G_eval_mode \
--G_ch 96 --D_ch 96 \
--ema --use_ema --ema_start 20000 \
--test_every 2000 --save_every 100 --num_best_copies 5 --num_save_copies 1 --seed 11 \
--use_multiepoch_sampler \
--pbar tqdm \
--inference_nosample \
--load_experiment_name VAE18_GAN_selfatt \
--experiment_name test_VAE18_GAN_selfatt_get_activation \
--resume \
--encoder Resnet-18 \
--no_sparsity \


# --load_in_mem
