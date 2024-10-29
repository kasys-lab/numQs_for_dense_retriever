import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../../../.."))

import numpy as np

from experimental_settings.numerical_condition_benchmark_base import (
    NumericalConditionDatasetBuilderBase,
)


class ArtificialTOEFLScoreDataset(NumericalConditionDatasetBuilderBase):
    def __init__(
        self,
        csv_data_filepath,
        entity_category_primary_column,
        entity_category_secondary_column,
        number_column,
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

    def _format_number(self, number):
        return f"{number} points"

    def _format_between_number(self, lower_bound, upper_bound):
        return f"{lower_bound} to {upper_bound} points"

    def generate_corpus_text(
        self, primary_category: str, secondary_category: str, number, metadata=None
    ):
        name = metadata["name"]

        skill_text = primary_category
        industry_text = secondary_category

        number_text = self._format_number(number)

        corpus_text = f"{name} works in a {industry_text} company with a {skill_text} job, and the TOEFL score is {number_text}."

        return corpus_text

    def generate_primary_query_target(self, primary_category):
        return f"{primary_category} job"

    def generate_secondary_query_target(self, secondary_category):
        return f"{secondary_category} companies"

    def generate_primary_secondary_query_target(
        self, primary_category, secondary_category
    ):
        return f"{primary_category} jobs in {secondary_category} companies"

    def generate_none_query_target(self):
        return "jobs"

    def generate_equal_query_text(self, query_target, query_number):
        number_text = self._format_number(query_number)

        return f"Who has a TOEFL score of {number_text} among those who work in {query_target}?"

    def generate_less_query_text(self, query_target, query_number):
        number_text = self._format_number(query_number)

        return f"Who has a TOEFL score of less than {number_text} among those who work in {query_target}?"

    def generate_more_query_text(self, query_target, query_number):
        number_text = self._format_number(query_number)

        return f"Who has a TOEFL score of more than {number_text} among those who work in {query_target}?"

    def generate_between_query_text(self, query_target, query_number):
        lower_bound, upper_bound = sorted(query_number)

        number_text = self._format_between_number(lower_bound, upper_bound)

        return f"Who has a TOEFL score between {number_text} among those who work in {query_target}?"

    def generate_around_query_text(self, query_target, query_number):
        number_text = self._format_number(query_number)

        return f"Who has a TOEFL score around {number_text} among those who work in {query_target}?"

    def generate_none_query_text(self, query_target):
        return f"What are all the {query_target}?"
