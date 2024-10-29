import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../../.."))

import numpy as np

from experimental_settings.numerical_condition_benchmark_base import (
    NumericalConditionDatasetBuilderBase,
)


class JobPostDataset(NumericalConditionDatasetBuilderBase):
    def __init__(
        self,
        csv_data_filepath,
        entity_category_primary_column,
        entity_category_secondary_column,
        number_column,
        max_corpus=-1,
        query_num=5000,
        only_save_relevance=True,
        is_primary_category_list=True,
        is_secondary_category_list=False,
        number_mode="none",
        currency="dollars",
        is_exchange_rate=False,
        job_title_mode="company_and_title",
        seed=42,
        query_nums_pick_replace=True,
    ):
        super().__init__(
            csv_data_filepath,
            entity_category_primary_column,
            entity_category_secondary_column,
            number_column,
            max_corpus,
            query_num,
            only_save_relevance,
            is_primary_category_list,
            is_secondary_category_list,
            seed=seed,
            query_nums_pick_replace=query_nums_pick_replace,
        )

        np.random.seed(seed)

        self.number_mode = number_mode
        self.currency = currency
        self.is_exchange_rate = is_exchange_rate
        self.job_title_mode = job_title_mode

    def _exchange_number(self, number):
        if self.is_exchange_rate:
            number = self._exchange_rate(number)

        if self.number_mode == "K":
            number = number / 1000

            if number.is_integer():
                number = int(number)

            return number
        elif self.number_mode == "M":
            number = number / 1000000

            if number.is_integer():
                number = int(number)

            return number
        elif self.number_mode == "none":
            return number
        else:
            raise ValueError(f"Invalid number_mode: {self.number_mode}")

    def _exchange_rate(self, number):
        if self.currency == "dollars":
            return number
        elif self.currency == "yen":
            return int(number * 150)
        elif self.currency == "euro":
            return int(number * 0.9)
        else:
            raise ValueError(f"Invalid currency: {self.currency}")

    def _format_number(self, number):
        number = self._exchange_number(number)

        if self.number_mode == "K":
            return f"{number}k {self.currency}"
        elif self.number_mode == "M":
            return f"{number}M {self.currency}"
        elif self.number_mode == "none":
            return f"{number} {self.currency}"
        else:
            raise ValueError(f"Invalid number_mode: {self.number_mode}")

    def _format_between_number(self, lower_bound, upper_bound):
        lower_bound = self._exchange_number(lower_bound)
        upper_bound = self._exchange_number(upper_bound)

        if self.number_mode == "K":
            return f"{lower_bound}k to {upper_bound}k {self.currency}"
        elif self.number_mode == "M":
            return f"{lower_bound}k to {upper_bound}M {self.currency}"
        elif self.number_mode == "none":
            return f"{lower_bound} to {upper_bound} {self.currency}"
        else:
            raise ValueError(f"Invalid number_mode: {self.number_mode}")

    def _format_job_title(self, company_name, title):
        if self.job_title_mode == "company_and_title":
            return f"{company_name} is looking for {title}"
        elif self.job_title_mode == "title":
            return f"{title} is available"
        elif self.job_title_mode == "company":
            return f"{company_name} is looking for a new employee"
        else:
            raise ValueError(f"Invalid job_title_mode: {self.job_title_mode}")

    def generate_corpus_text(
        self, primary_category: str, secondary_category: str, number, metadata=None
    ):
        company_name = metadata["company_name"]
        title = metadata["title"]

        job_title_text = self._format_job_title(company_name, title)

        skills = primary_category
        if len(skills) > 2:
            skills = np.random.choice(skills, 2, replace=False).tolist()

        skills_text = ", ".join(skills)

        industry_text = secondary_category

        number_text = self._format_number(number)

        corpus_text = f"{job_title_text} with {skills_text} skills in {industry_text} industry. The salary is {number_text}."

        return corpus_text, skills, industry_text

    def generate_corpus(
        self, primary_category, secondary_category, number, metadata=None
    ):
        text, skills, industry = self.generate_corpus_text(
            primary_category, secondary_category, number, metadata
        )

        entities = {
            self.PRIMARY_CATEGORY_DICT_NAME: skills,
            self.SECONDARY_CATEGORY_DICT_NAME: industry,
            "number": int(number),
            "metadata": metadata,
        }

        data = {
            "text": text,
            "entities": entities,
            "number": int(number),
        }

        return data

    def generate_primary_query_target(self, primary_category):
        return f"{primary_category} job"

    def generate_secondary_query_target(self, secondary_category):
        return f"{secondary_category} industry job"

    def generate_primary_secondary_query_target(
        self, primary_category, secondary_category
    ):
        return f"{secondary_category} industry {primary_category} job"

    def generate_none_query_target(self):
        return "job"

    def generate_equal_query_text(self, query_target, query_number):
        number_text = self._format_number(query_number)

        return f"What are the {query_target} that has a salary equal to {number_text}?"

    def generate_less_query_text(self, query_target, query_number):
        number_text = self._format_number(query_number)

        return f"What are the {query_target} that has a salary less than {number_text}?"

    def generate_more_query_text(self, query_target, query_number):
        number_text = self._format_number(query_number)

        return f"What are the {query_target} that has a salary more than {number_text}?"

    def generate_between_query_text(self, query_target, query_number):
        lower_bound, upper_bound = sorted(query_number)

        number_text = self._format_between_number(lower_bound, upper_bound)

        return f"What are the {query_target} that has a salary between {number_text}?"

    def generate_around_query_text(self, query_target, query_number):
        number_text = self._format_number(query_number)

        return f"What are the {query_target} that has a salary around {number_text}?"

    def generate_none_query_text(self, query_target):
        return f"What are all the {query_target}?"
