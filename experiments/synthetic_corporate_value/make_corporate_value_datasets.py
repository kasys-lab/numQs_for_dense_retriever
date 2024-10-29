import pandas as pd
import pycountry
from faker import Faker
from tqdm import tqdm


class FakeStartupValueGenerator:
    def __init__(self) -> None:
        pass

    @classmethod
    def generate_csv(
        cls,
        save_dir,
        num_range=(1, 1_000),
        num_step=1,
        job_dataset_dir: str = "linkedin-job-postings",
        seed=42,
    ):
        fake = Faker()
        fake.seed_instance(seed)

        countries_codes = [
            "US",
            "GB",
            "CN",
            "CA",
            "JP",
            "FR",
            "DE",
            "IT",
            "IN",
            "MX",
            "BR",
            "KR",
            "AU",
        ]
        countries = [
            pycountry.countries.get(alpha_2=code).name for code in countries_codes
        ]
        industries = cls._load_industries(job_dataset_dir)

        numeral_patterns = ["normal", "k", "m"]
        need_comma_patterns = [False, True]
        currency_patterns = ["dollar", "USD", "symbol"]

        headers = [
            "country",
            "industry",
            "number",
            "value_str",
            "company_name",
            "numeral_pattern",
            "need_comma_pattern",
            "currency_pattern",
        ]

        rows = []
        for num in tqdm(range(num_range[0], num_range[1], num_step)):
            country = fake.random_element(countries)
            industry = fake.random_element(industries)
            company_name = fake.company()
            value = num

            for numeral_pattern in numeral_patterns:
                for need_comma_pattern in need_comma_patterns:
                    for currency_pattern in currency_patterns:
                        for scale_pattern in numeral_patterns:
                            if scale_pattern == "normal":
                                base_value = value
                            elif scale_pattern == "k":
                                base_value = value * 1000
                            elif scale_pattern == "m":
                                base_value = value * 1_000_000

                            if numeral_pattern == "normal":
                                raw_value = base_value
                            elif numeral_pattern == "k":
                                raw_value = base_value / 1000
                            elif numeral_pattern == "m":
                                raw_value = base_value / 1_000_000

                            if raw_value.is_integer():
                                raw_value = int(raw_value)

                            if need_comma_pattern:
                                raw_value_text = "{:,}".format(raw_value)
                            else:
                                raw_value_text = str(raw_value)

                            if numeral_pattern == "k":
                                raw_value_text = f"{raw_value_text}k"
                            elif numeral_pattern == "m":
                                raw_value_text = f"{raw_value_text}M"

                            if currency_pattern == "dollar":
                                value_str = f"{raw_value_text} dollars"
                            elif currency_pattern == "USD":
                                value_str = f"{raw_value_text} USD"
                            elif currency_pattern == "symbol":
                                value_str = f"${raw_value_text}"

                            rows.append(
                                [
                                    country,
                                    industry,
                                    base_value,
                                    value_str,
                                    company_name,
                                    numeral_pattern,
                                    need_comma_pattern,
                                    currency_pattern,
                                ]
                            )

        df = pd.DataFrame(rows, columns=headers)
        df.to_csv(f"{save_dir}/startup_value.csv", index=False)

    @staticmethod
    def _load_skills(job_dataset_dir: str):
        skills_df = pd.read_csv(f"{job_dataset_dir}/mappings/skills.csv")
        skills_df = skills_df.dropna(subset=["skill_name"])

        unique_skills = skills_df["skill_name"].unique()

        return unique_skills

    @staticmethod
    def _load_industries(job_dataset_dir: str):
        industries_df = pd.read_csv(f"{job_dataset_dir}/mappings/industries.csv")
        industries_df = industries_df.dropna(subset=["industry_name"])

        unique_industries = industries_df["industry_name"].unique()

        return unique_industries
