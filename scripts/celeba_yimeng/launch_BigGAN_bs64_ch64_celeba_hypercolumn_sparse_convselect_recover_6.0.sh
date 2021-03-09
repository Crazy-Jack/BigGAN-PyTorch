#!/bin/bash
python3.7 ../../train.py \
--data_root /data2/tianqinl/Dataset \
--dataset CelebA --parallel --shuffle  --num_workers 32 --batch_size 36 \
--num_G_accumulations 3 --num_D_accumulations 3 \
--num_D_steps 1 --G_lr 1e-5 --D_lr 4e-5 --E_lr 1e-5 --D_B2 0.999 --G_B2 0.999 --E_B2 0.999 \
--lambda_vae_kld 1e-3 --lambda_vae_recon 10 \
--G_attn 8_16_32_64 --D_attn 64 \
--G_nl inplace_relu --D_nl inplace_relu \
--SN_eps 1e-6 --BN_eps 1e-5 --adam_eps 1e-6 \
--G_ortho 0.0 \
--G_shared \
--G_init ortho --D_init ortho --E_init xavier \
--hier --dim_z 120 --shared_dim 128 \
--G_eval_mode \
--G_ch 96 --D_ch 96 \
--test_every 2000 --save_img_every 20 --save_model_every 100 --num_best_copies 5 --num_save_copies 1 --seed 0 \
--use_multiepoch_sampler \
--pbar tqdm \
--inference_nosample \
--experiment_name hypercolumn_sparse_conv_sparse_vc_recover_mode_6.0 \
--sparsity_resolution 32 --sparsity_ratio 10 \
--save_weights \
--encoder Resnet-18 \
--sparsity_mode conv_sparse_vc_recover_no_sparse_mode_6.0 \
--sparse_decay_rate 1e-3 \
--no_adaptive_tau \
--sparse_vc_interaction_num 2 \
--vc_dict_size 1000 \
--num_epochs 1000 \
--sparse_vc_prob_interaction 2 \
--patchGAN \
--lambda_g_additional 100 \
# --resume \

### LOG ###
# in 6.0 (compare with 5.3):
# plan sparse layer on 32x32 layer
# only use VC not the map information
