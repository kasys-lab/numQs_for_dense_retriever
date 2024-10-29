import pandas as pd


def make_company_employee_dataset(job_posting_df: pd.DataFrame):
    job_dataset = job_posting_df[
        ["company_size", "city", "name", "description", "employee_count", "industry"]
    ]

    job_dataset["name"] = job_dataset["name"].apply(lambda x: x.replace("\n", " "))

    job_dataset["description"] = job_dataset["description"].apply(
        lambda x: x.replace("\n", " ").replace("\r", " ")[:100]
    )

    job_dataset["employee_count_int"] = job_dataset["employee_count"].astype(int)
    job_dataset["employee_count_int_k"] = job_dataset["employee_count_int"].apply(
        lambda x: x / 1000
    )
    job_dataset["employee_count_int_m"] = job_dataset["employee_count_int"].apply(
        lambda x: x / 1000000
    )

    job_dataset = job_dataset.sort_values("employee_count_int", ascending=False)

    return job_dataset


def create_company_employee_datasets(
    save_dir: str, job_dataset_dir="linkedin-job-postings"
):
    companies_path = f"{job_dataset_dir}/companies/companies.csv"
    company_industry_path = f"{job_dataset_dir}/companies/company_industries.csv"
    company_employee_path = f"{job_dataset_dir}/companies/employee_counts.csv"

    companies_df = pd.read_csv(companies_path)

    company_industry_df = pd.read_csv(company_industry_path)

    company_employee_df = pd.read_csv(company_employee_path)
    company_employee_df = (
        company_employee_df.groupby("company_id")["employee_count"].max().reset_index()
    )

    merged_companies_df = pd.merge(companies_df, company_industry_df, on="company_id")
    merged_companies_df = pd.merge(
        merged_companies_df, company_employee_df, on="company_id"
    )

    merged_companies_df = merged_companies_df.dropna(
        subset=["industry", "employee_count", "city", "name", "description"]
    )

    city_mapping = {}
    merged_companies_df["normalized_city"] = (
        merged_companies_df["city"].str.lower().str.replace(" ", "")
    )

    unique_cities = merged_companies_df["normalized_city"].unique()
    for city in unique_cities:
        city_df = merged_companies_df[merged_companies_df["normalized_city"] == city]
        major_city = city_df.groupby("city").size().idxmax()

        city_mapping[city] = major_city

    merged_companies_df["city"] = merged_companies_df["normalized_city"].apply(
        lambda x: city_mapping[x]
    )

    city_counts = merged_companies_df["city"].value_counts()
    other_cities = city_counts[city_counts < 10].index
    merged_companies_df.loc[merged_companies_df["city"].isin(other_cities), "city"] = (
        "Other"
    )
    merged_companies_df.loc[merged_companies_df["city"] == "0", "city"] = "Other"

    merged_companies_df = merged_companies_df[merged_companies_df["city"] != "Other"]

    merged_companies_df = merged_companies_df.dropna(
        subset=["industry", "employee_count", "city"]
    )

    merged_companies_df = merged_companies_df.drop_duplicates(subset=["description"])

    company_employee_dataset_path = f"{save_dir}/company_employee_dataset.csv"

    company_employee_dataset = make_company_employee_dataset(merged_companies_df)

    company_employee_dataset.to_csv(company_employee_dataset_path, index=False)

    shuffled_company_employee_dataset_path = (
        f"{save_dir}/shuffled_company_employee_dataset.csv"
    )

    shuffled_company_employee_postings_df = merged_companies_df.copy()
    shuffled_company_employee_postings_df["employee_count"] = (
        shuffled_company_employee_postings_df["employee_count"].sample(frac=1).values
    )

    shuffled_company_employee_dataset = make_company_employee_dataset(
        shuffled_company_employee_postings_df
    )

    shuffled_company_employee_dataset.to_csv(
        shuffled_company_employee_dataset_path, index=False
    )


def main(args):
    create_company_employee_datasets(args.save_dir, args.job_dataset_dir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", type=str, default="job_dataset")
    parser.add_argument("--job_dataset_dir", type=str, default="linkedin-job-postings")

    args = parser.parse_args()

    main(args)
