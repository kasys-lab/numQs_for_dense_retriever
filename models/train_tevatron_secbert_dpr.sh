#!/bin/bash

python -m tevatron.driver.train \
  --output_dir model_secbert_nq_dpr \
  --dataset_name Tevatron/wikipedia-nq \
  --model_name_or_path nlpaueb/sec-bert-base \
  --do_train \
  --save_steps 20000 \
  --fp16 \
  --per_device_train_batch_size 64 \
  --train_n_passages 2 \
  --learning_rate 1e-5 \
  --q_max_len 32 \
  --p_max_len 156 \
  --num_train_epochs 40 \
