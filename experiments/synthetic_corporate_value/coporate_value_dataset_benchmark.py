import os
import random
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../.."))

import numpy as np

from experimental_settings.numerical_condition_benchmark_base import (
    NumericalConditionDatasetBuilderBase,
)


class StartupValueDataset(NumericalConditionDatasetBuilderBase):
    def __init__(
        self,
        csv_data_filepath,
        entity_category_primary_column,
        entity_category_secondary_column,
        number_column,
        numeral_pattern,
        need_comma_pattern,
        currency_pattern,
        max_corpus=-1,
        query_num=5000,
        only_save_relevance=True,
        is_primary_category_list=False,
        is_secondary_category_list=False,
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
        random.seed(seed)

        assert numeral_pattern in ["normal", "k", "m"]
        assert need_comma_pattern in [False, True]
        assert currency_pattern in ["dollar", "USD", "symbol"]

        self.numeral_pattern = numeral_pattern  # ["normal", "k", "m"]
        self.need_comma_pattern = need_comma_pattern  # [False, True]
        self.currency_pattern = currency_pattern  # ["dollar", "USD", "symbol"]

    def _format_number(self, number):
        return self.generate_query_number_text(
            number, self.numeral_pattern, self.need_comma_pattern, self.currency_pattern
        )

    def _format_between_number(self, lower_bound, upper_bound):
        lower_bound = self.generate_query_number_text(
            lower_bound,
            self.numeral_pattern,
            self.need_comma_pattern,
            self.currency_pattern,
        )
        upper_bound = self.generate_query_number_text(
            upper_bound,
            self.numeral_pattern,
            self.need_comma_pattern,
            self.currency_pattern,
        )

        return f"{lower_bound} to {upper_bound}"

    def generate_corpus_text(  # type: ignore
        self, primary_category: str, secondary_category: str, number, metadata
    ):
        company_name = metadata["company_name"]
        value_str = metadata["value_str"]

        country = primary_category
        industry_text = secondary_category
        # english
        corpus_text = f"{company_name} is a {industry_text} company in {country}. The company's value is {value_str}."

        return corpus_text, country, industry_text

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

    def generate_query_number_text(
        self, query_number, numeral_pattern, need_comma_pattern, currency_pattern
    ):
        raw_salary = query_number

        if numeral_pattern == "k":
            raw_salary = raw_salary / 1000
        elif numeral_pattern == "m":
            raw_salary = raw_salary / 1000000

        if need_comma_pattern:
            raw_salary_text = "{:,}".format(raw_salary)
        else:
            raw_salary_text = str(raw_salary)

        if numeral_pattern == "k":
            raw_salary_text = f"{raw_salary_text}k"
        elif numeral_pattern == "m":
            raw_salary_text = f"{raw_salary_text}M"

        if currency_pattern == "dollar":
            salary_str = f"{raw_salary_text} dollars"
        elif currency_pattern == "USD":
            salary_str = f"{raw_salary_text} USD"
        elif currency_pattern == "symbol":
            salary_str = f"${raw_salary_text}"
        else:
            raise ValueError(f"Invalid currency pattern: {currency_pattern}")

        return salary_str

    def generate_primary_query_target(self, primary_category):
        return f"{primary_category} company"

    def generate_secondary_query_target(self, secondary_category):
        return f"{secondary_category} industry company"

    def generate_primary_secondary_query_target(
        self, primary_category, secondary_category
    ):
        return f"{secondary_category} industry {primary_category} company"

    def generate_none_query_target(self):
        return "company"

    def generate_equal_query_text(self, query_target, query_number):
        number_text = self._format_number(query_number)

        return f"What are the {query_target} that has a value equal to {number_text}?"

    def generate_less_query_text(self, query_target, query_number):
        number_text = self._format_number(query_number)

        return f"What are the {query_target} that has a value less than {number_text}?"

    def generate_more_query_text(self, query_target, query_number):
        number_text = self._format_number(query_number)

        return f"What are the {query_target} that has a value more than {number_text}?"

    def generate_between_query_text(self, query_target, query_number):
        lower_bound, upper_bound = query_number

        number_text = self._format_between_number(lower_bound, upper_bound)

        return f"What are the {query_target} that has a value between {number_text}?"

    def generate_around_query_text(self, query_target, query_number):
        number_text = self._format_number(query_number)

        return f"What are the {query_target} that has a value around {number_text}?"

    def generate_none_query_text(self, query_target):
        return f"What are all the {query_target}?"
