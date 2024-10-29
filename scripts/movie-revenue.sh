#!/bin/bash

python experiments/movie_dataset/main.py \
    encoder.dpr=false encoder.batch_size=512 benchmark.execlude_non_numeric=false \
    benchmark.query_num=6048 \
    benchmark.query_nums_pick_replace=False \
    benchmark.query_target_category="['none']" \
    benchmark.currencies="['dollars']" \
    benchmark.is_exchange_rate_list="[False]" \
    benchmark.movie_dataset_types="['original', 'shuffled']"
