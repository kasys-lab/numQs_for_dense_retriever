#!/bin/bash

python experiments/synthetic_unit_dataset/main.py \
    encoder.dpr=false encoder.batch_size=512 benchmark.execlude_non_numeric=false benchmark.depth=200 dataset.num_times=10
