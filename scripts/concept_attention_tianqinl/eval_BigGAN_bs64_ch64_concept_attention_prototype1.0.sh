#!/bin/bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PAT:/home/tianqinl/.conda/envs/3D_pytorch/lib/

python ../../prob_intermediate.py \
--data_root /user_data/tianqinl/Dataset \
--dataset CelebA --parallel --shuffle  --num_workers 32 --batch_size 36 \
--num_G_accumulations 3 --num_D_accumulations 3 \
--num_D_steps 1 --G_lr 1e-5 --D_lr 4e-5 --E_lr 1e-5 --D_B2 0.999 --G_B2 0.999 --E_B2 0.999 \
--lambda_vae_kld 1e-3 --lambda_vae_recon 10 \
--G_attn 64 --D_attn 64 \
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
--experiment_name concept_prototype_momem_attention_1.0 \
--encoder Resnet-18 \
--no_adaptive_tau \
--no_sparsity \
--attend_mode concept_proto_attention_1.0 \
--cp_pool_size_per_cluster 100 \
--cp_num_k 20 \
--cp_dim 64 \
--cp_warmup_total_iter 10000 \
--cp_momentum 0.6 \
--resume \

# 1.1: use no momentum pool