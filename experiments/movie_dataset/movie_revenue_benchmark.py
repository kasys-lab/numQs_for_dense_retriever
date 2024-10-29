import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

import numpy as np

from experimental_settings.numerical_condition_benchmark_base import (
    NumericalConditionDatasetBuilderBase,
)


class MovieRevenueDataset(NumericalConditionDatasetBuilderBase):
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
        is_secondary_category_list=True,
        number_mode="none",
        currency="dollars",
        is_exchange_rate=False,
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

        self.number_mode = number_mode  # "none" or "K" or "M"
        self.currency = currency  # "dollars" or "yen" or "euro"
        self.is_exchange_rate = is_exchange_rate

    def _exchange_rate(self, number):  # dollar -> yen or euro
        if self.currency == "dollars":
            return number
        elif self.currency == "yen":
            return int(number * 150)
        elif self.currency == "euro":
            return int(number * 0.9)
        else:
            raise ValueError(f"Invalid currency: {self.currency}")

    def _format_number(self, number):
        if self.is_exchange_rate:
            number = self._exchange_rate(number)

        if self.number_mode == "K":
            return f"{number} thousand {self.currency}"
        elif self.number_mode == "M":
            return f"{number} million {self.currency}"
        elif self.number_mode == "none":
            return f"{number} {self.currency}"
        else:
            raise ValueError(f"Invalid number_mode: {self.number_mode}")

    def _format_between_number(self, lower_bound, upper_bound):
        if self.number_mode == "K":
            return f"{lower_bound} to {upper_bound} thousand {self.currency}"
        elif self.number_mode == "M":
            return f"{lower_bound} to {upper_bound} million {self.currency}"
        elif self.number_mode == "none":
            return f"{lower_bound} to {upper_bound} {self.currency}"
        else:
            raise ValueError(f"Invalid number_mode: {self.number_mode}")

    def generate_corpus_text(
        self, primary_category: str, secondary_category: str, number, metadata=None
    ):
        movie_title = metadata["title"]

        genres = primary_category
        if len(genres) > 2:
            genres = np.random.choice(genres, 2, replace=False).tolist()

        genres_text = ", ".join(genres)

        production_companies = secondary_category
        if len(production_companies) > 2:
            production_companies = np.random.choice(
                production_companies, 2, replace=False
            ).tolist()

        production_companies_text = ", ".join(production_companies)

        number_text = self._format_number(number)
        # english
        return (
            f"{movie_title} is a {genres_text} movie produced by {production_companies_text}. It has earned {number_text} in revenue.",
            genres,
            production_companies,
        )

    def generate_corpus(
        self, primary_category, secondary_category, number, metadata=None
    ):
        text, genres, production_companies = self.generate_corpus_text(
            primary_category, secondary_category, number, metadata
        )

        entities = {
            self.PRIMARY_CATEGORY_DICT_NAME: genres,
            self.SECONDARY_CATEGORY_DICT_NAME: production_companies,
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
        return f"{primary_category} movies"

    def generate_secondary_query_target(self, secondary_category):
        return f"{secondary_category} produced movies"

    def generate_primary_secondary_query_target(
        self, primary_category, secondary_category
    ):
        return f"{secondary_category} produced {primary_category} movies"

    def generate_none_query_target(self):
        return "movies"

    def generate_equal_query_text(self, query_target, query_number):
        number_text = self._format_number(query_number)

        return f"What are the {query_target} that earned equal to {number_text} in revenue?"

    def generate_less_query_text(self, query_target, query_number):
        number_text = self._format_number(query_number)

        return f"What are the {query_target} that earned less than {number_text} in revenue?"

    def generate_more_query_text(self, query_target, query_number):
        number_text = self._format_number(query_number)

        return f"What are the {query_target} that earned more than {number_text} in revenue?"

    def generate_between_query_text(self, query_target, query_number):
        lower_bound, upper_bound = sorted(query_number)

        number_text = self._format_between_number(lower_bound, upper_bound)

        return f"What are the {query_target} that earned revenue between {number_text} in revenue?"

    def generate_around_query_text(self, query_target, query_number):
        number_text = self._format_number(query_number)

        return f"What are the {query_target} that earned revenue around {number_text} in revenue?"

    def generate_none_query_text(self, query_target):
        return f"What are the {query_target}?"
