# Investigating the Performance of Dense Retrievers for Queries with Numerical Conditions


```bash
.
├── README.md
├── experimental_settings <- Contains the base configuration for the experiments
│   ├── __init__.py
│   └── numerical_condition_benchmark_base.py
├── experiments
│   ├── job_dataset
│   │   ├── company_employee
│   │   │   ├── company_employee_dataset_benchmark.py
│   │   │   ├── conf
│   │   │   │   └── config.yaml
│   │   │   ├── main.py
│   │   │   └── make_company_employee_datasets.py
│   │   ├── download_raw_datasets.sh
│   │   └── job_post
│   │       ├── conf
│   │       │   └── config.yaml
│   │       ├── job_post_dataset_benchmark.py
│   │       ├── main.py
│   │       └── make_job_post_datasets.py
│   ├── movie_dataset
│   │   ├── conf
│   │   │   └── config.yaml
│   │   ├── download_raw_datasets.sh
│   │   ├── main.py
│   │   ├── make_movie_datasets.py
│   │   └── movie_revenue_benchmark.py
│   ├── synthetic_corporate_value
│   │   ├── conf
│   │   │   └── config.yaml
│   │   ├── coporate_value_dataset_benchmark.py
│   │   ├── main.py
│   │   └── make_corporate_value_datasets.py
│   └── synthetic_unit_dataset
│       ├── conf
│       │   └── config.yaml
│       ├── main.py
│       ├── make_unit_datasets.py
│       └── unit_dataset_benchmark.py
├── models <- Contains the training scripts for the RQ3 「RQ3: Does the Internal Knowledge of Language Models Influence the Retrieval Effectiveness for NumQs?」
│   ├── train_tevatron_bert_dpr.sh
│   ├── train_tevatron_finbert_dpr.sh
│   ├── train_tevatron_genbert_dpr.sh
│   ├── train_tevatron_secbert_dpr.sh
│   └── train_tevatron_secbert_num_dpr.sh
├── pyproject.toml
├── scripts <- Contains the execution scripts for the experiments
│   ├── company-employee.sh
│   ├── corporate_value.sh
│   ├── job-post.sh
│   ├── movie-revenue.sh
│   └── unit.sh
├── utils
│   ├── __init__.py
│   ├── bm25_sparse_retriever.py
│   ├── colbert_retriever.py
│   ├── dpr_embedding.py
│   ├── e5_embedding.py
│   ├── models
│   │   ├── __init__.py
│   │   ├── embedding_model.py
│   │   └── retriever_model.py
│   ├── openai_embedding.py
│   ├── random_embedding.py
│   └── tevatron_dpr_embedding.py
└── uv.lock
```
