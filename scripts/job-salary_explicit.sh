#!/bin/bash

python experiments/sensitive_attribute_synthetic_dataset/synthetic_salary_explicit/main.py \
    encoder.dpr=false encoder.batch_size=512 benchmark.execlude_non_numeric=false benchmark.depth=500
