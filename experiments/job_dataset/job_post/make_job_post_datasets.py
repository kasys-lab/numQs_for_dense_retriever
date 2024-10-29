import numpy as np
import pandas as pd


def make_job_dataset(job_posting_df: pd.DataFrame):
    job_dataset = job_posting_df[
        ["company_name", "max_salary", "title", "description", "skill_name", "industry"]
    ]

    job_dataset["title"] = job_dataset["title"].apply(lambda x: x.replace("\n", " "))
    job_dataset["description"] = job_dataset["description"].apply(
        lambda x: x.replace("\n", " ")[:100]
    )

    job_dataset["max_salary_int"] = job_dataset["max_salary"].astype(int)
    job_dataset["max_salary_int_k"] = job_dataset["max_salary_int"].apply(
        lambda x: x / 1000
    )
    job_dataset["max_salary_int_m"] = job_dataset["max_salary_int"].apply(
        lambda x: x / 1000000
    )

    job_dataset["max_salary_int_k"] = job_dataset["max_salary_int_k"].apply(
        lambda x: int(x) if x.is_integer() else x
    )
    job_dataset["max_salary_int_m"] = job_dataset["max_salary_int_m"].apply(
        lambda x: int(x) if x.is_integer() else x
    )

    job_dataset = job_dataset.sort_values("max_salary_int", ascending=False)

    return job_dataset


def create_job_datasets(save_dir: str, job_dataset_dir="linkedin-job-postings"):
    job_postings_path = f"{job_dataset_dir}/postings.csv"
    skill_mapping_path = f"{job_dataset_dir}/mappings/skills.csv"
    jobid_skills_path = f"{job_dataset_dir}/jobs/job_skills.csv"
    company_industry_path = f"{job_dataset_dir}/companies/company_industries.csv"

    job_postings_df = pd.read_csv(job_postings_path)

    job_postings_df = job_postings_df[job_postings_df["pay_period"] == "YEARLY"]

    job_postings_df = job_postings_df.dropna(
        subset=["company_name", "max_salary", "title", "description"]
    )

    jobid_skills_df = pd.read_csv(jobid_skills_path)
    skill_mapping_df = pd.read_csv(skill_mapping_path)

    jobid_skills_df = pd.merge(jobid_skills_df, skill_mapping_df, on="skill_abr")

    jobid_skills_df = (
        jobid_skills_df.groupby("job_id")["skill_name"].apply(list).reset_index()
    )

    merged_job_postings_df = pd.merge(job_postings_df, jobid_skills_df, on="job_id")

    company_industry_df = pd.read_csv(company_industry_path)

    merged_job_postings_df = pd.merge(
        merged_job_postings_df, company_industry_df, on="company_id"
    )

    merged_job_postings_df = merged_job_postings_df.dropna(
        subset=["skill_name", "industry"]
    )

    merged_job_postings_df = merged_job_postings_df.drop_duplicates(
        subset=["description"]
    )

    job_dataset_path = f"{save_dir}/job_dataset.csv"

    job_dataset = make_job_dataset(merged_job_postings_df.copy())

    job_dataset.to_csv(job_dataset_path, index=False)

    shuffled_job_dataset_path = f"{save_dir}/shuffled_job_dataset.csv"

    shuffled_job_postings_df = merged_job_postings_df.copy()

    shuffled_salary = shuffled_job_postings_df["max_salary"].values

    shuffled_salary = np.random.choice(
        shuffled_salary, len(shuffled_salary), replace=False
    )

    shuffled_job_postings_df["max_salary"] = shuffled_salary

    shuffled_job_dataset = make_job_dataset(shuffled_job_postings_df)

    shuffled_job_dataset.to_csv(shuffled_job_dataset_path, index=False)


def main(args):
    create_job_datasets(args.save_dir, args.job_dataset_dir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, default="job_dataset")
    parser.add_argument("--job_dataset_dir", type=str, default="linkedin-job-postings")

    args = parser.parse_args()

    main(args)
