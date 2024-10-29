#!/bin/bash

# download model from https://github.com/ag1988/injecting_numeracy/blob/master/pre_training/download.sh

python -m tevatron.driver.train \
  --output_dir model_genbert_nq_dpr \
  --dataset_name Tevatron/wikipedia-nq \
  --model_name_or_path models/genbert \
  --do_train \
  --save_steps 20000 \
  --fp16 \
  --per_device_train_batch_size 64 \
  --train_n_passages 2 \
  --learning_rate 1e-5 \
  --q_max_len 32 \
  --p_max_len 156 \
  --num_train_epochs 40
