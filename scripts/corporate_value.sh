#!/bin/bash

python experiments/numerical_expression/artificial_startup_value/main.py \
    encoder.dpr=false encoder.batch_size=512 benchmark.query_num=3000 benchmark.depth=200 benchmark.query_target_category='["primary", "secondary", "primary_secondary"]' benchmark.execlude_non_numeric=false
