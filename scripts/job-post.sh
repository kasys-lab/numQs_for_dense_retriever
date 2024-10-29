#!/bin/bash

python experiments/job_dataset/job_post/main.py \
    encoder.dpr=false encoder.batch_size=512 benchmark.execlude_non_numeric=false \
    benchmark.currencies="['dollars']" \
    benchmark.query_target_category="['none']" \
    benchmark.job_title_types="['title']" \
    benchmark.is_exchange_rate_list="[False]"
