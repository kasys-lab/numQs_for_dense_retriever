import logging

import numpy as np
import pandas as pd
from faker import Faker
from names_dataset import NameDataset

logger = logging.getLogger(__name__)


class ArtificialScoreTableGenerator:
    @classmethod
    def generate_csv(
        cls,
        csv_filepath,
        num_names=1000,
        num_category_pairs=5000,
        country_codes: list = [
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
        ],
        job_dataset_dir: str = "linkedin-job-postings",
        score_range=(0, 120),
        score_step=5,
        seed=42,
    ):
        logger.info(
            f"Generating artificial score dataset... ({num_category_pairs} skill-industry-score pairs x {len(country_codes)} countries x genders = {num_category_pairs * len(country_codes) * 2})"
        )

        names_data = cls._get_names(
            country_codes=country_codes, num_names=num_names, seed=seed
        )

        skills = cls._load_skills(job_dataset_dir)
        industries = cls._load_industries(job_dataset_dir)
        skill_industry_score_pairs = cls._make_skill_industry_score_pairs(
            skills,
            industries,
            n_pairs=num_category_pairs,
            score_range=score_range,
            score_step=score_step,
            seed=seed,
        )

        headers = ["name", "country_code", "gender", "skill", "industry", "score"]

        rows = []
        for index, pairs in enumerate(skill_industry_score_pairs):
            skill, industry, score = pairs

            for country_code in country_codes:
                for gender in ["M", "F"]:
                    name_data = names_data[country_code][gender][index % num_names]
                    name = name_data["name"]
                    rows.append([name, country_code, gender, skill, industry, score])

        df = pd.DataFrame(rows, columns=headers)
        df.to_csv(csv_filepath, index=False)

    @staticmethod
    def _load_skills(job_dataset_dir: str):
        skills_df = pd.read_csv(f"{job_dataset_dir}/mappings/skills.csv")
        skills_df = skills_df.dropna(subset=["skill_name"])

        unique_skills = skills_df["skill_name"].unique()

        return unique_skills.tolist()

    @staticmethod
    def _load_industries(job_dataset_dir: str):
        industries_df = pd.read_csv(f"{job_dataset_dir}/mappings/industries.csv")
        industries_df = industries_df.dropna(subset=["industry_name"])

        unique_industries = industries_df["industry_name"].unique()

        return unique_industries.tolist()

    @staticmethod
    def _make_skill_industry_score_pairs(
        skills: list[str],
        industries: list[str],
        n_pairs: int = 100,
        score_range=(700000000, 2000000000),
        score_step=10000000,
        seed: int = 42,
    ):
        faker = Faker()

        faker.seed_instance(seed)

        # all skill, industry pairs
        skill_industry_pairs = []
        for skill in skills:
            for industry in industries:
                skill_industry_pairs.append((skill, industry))

        selected_skill_industry_pairs = faker.random_choices(
            skill_industry_pairs, length=n_pairs
        )

        result = []
        for skill, industry in selected_skill_industry_pairs:
            score = faker.random_int(
                min=score_range[0], max=score_range[1], step=score_step
            )
            result.append((skill, industry, score))

        return result

    @staticmethod
    def _get_names(
        country_codes: list = [
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
        ],
        num_names: int = 1000,
        seed: int = 42,
    ):
        nd = NameDataset()
        np.random.seed(seed)

        def _is_all_english(s):
            return all(c.isascii() and c.isalpha() for c in s)

        def _fill_names(names, num_names):
            if len(names) < num_names:
                return names + np.random.choice(names, num_names - len(names)).tolist()
            else:
                return names[:num_names]

        names_data = {}
        for country_code in country_codes:
            name_data = nd.get_top_names(
                n=int(num_names * 1.5), country_alpha2=country_code
            )
            male_names = name_data[country_code]["M"]
            female_names = name_data[country_code]["F"]

            male_names = list(set(male_names))
            female_names = list(set(female_names))

            male_names = list(filter(_is_all_english, male_names))
            female_names = list(filter(_is_all_english, female_names))

            male_names = _fill_names(male_names, num_names)
            female_names = _fill_names(female_names, num_names)

            male_name_data = list(
                map(
                    lambda x: {"name": x, "country_code": country_code, "gender": "M"},
                    male_names,
                )
            )
            female_name_data = list(
                map(
                    lambda x: {"name": x, "country_code": country_code, "gender": "F"},
                    female_names,
                )
            )

            # shuffle
            male_name_data = np.random.permutation(male_name_data).tolist()  # type: ignore
            female_name_data = np.random.permutation(female_name_data).tolist()  # type: ignore

            names_data[country_code] = {"M": male_name_data, "F": female_name_data}

        return names_data
