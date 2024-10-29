#!/bin/bash

python experiments/job_dataset/company_employee/main.py \
    encoder.dpr=false encoder.batch_size=512 benchmark.query_target_category='["none"]' benchmark.execlude_non_numeric=false
