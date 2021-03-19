#!/bin/bash
python ../../train.py \
--data_root /user_data/tianqinl/Dataset \
--dataset Mysmall_128 --parallel --shuffle  --num_workers 32 --batch_size 36 \
--num_G_accumulations 3 --num_D_accumulations 3 \
--num_D_steps 1 --G_lr 1e-4 --D_lr 4e-4 --E_lr 1e-4 --D_B2 0.999 --G_B2 0.999 --E_B2 0.999 \
--lambda_vae_kld 1e-3 --lambda_vae_recon 1 \
--G_attn 8_16_32_64 --D_attn 64 \
--G_nl inplace_relu --D_nl inplace_relu \
--SN_eps 1e-6 --BN_eps 1e-5 --adam_eps 1e-6 \
--G_ortho 0.0 \
--G_shared \
--G_init ortho --D_init ortho --E_init xavier \
--hier --dim_z 120 --shared_dim 128 \
--G_eval_mode \
--G_ch 64 --D_ch 64 \
--test_every 2000 --save_img_every 20 --save_model_every 100 --num_best_copies 5 --num_save_copies 1 --seed 0 \
--use_multiepoch_sampler \
--pbar tqdm \
--inference_nosample \
--experiment_name hypercolumn_sparse_implicit_recover_vc_sparse_10percent \
--sparsity_resolution 8_16_32_64 --sparsity_ratio 10_10_10_10 \
--save_weights \
--encoder Resnet-18 \
--sparsity_mode implicit_sparse_vc_recover_vcsparse \
--sparse_decay_rate 1e-3 \
--no_adaptive_tau \
# --resume \