import json

import pandas as pd
from faker import Faker


class UnitDatasetGenerator:
    """通貨、時間、重量、距離、面積、体積、温度、速度、電力を同じレンジでデータを生成するクラス"""

    def __init__(self) -> None:
        pass

    @classmethod
    def generate_csv(
        cls, save_dir, num_range=(1, 1000), num_step=1, num_times=5, seed=42
    ):
        faker = Faker()
        faker.seed_instance(seed)
        numbers = [i for i in range(num_range[0], num_range[1], num_step)]
        numbers = numbers * num_times

        names = faker.words(nb=len(numbers))
        names = [name.upper() for name in names]

        generators = [
            cls.generate_currency_dataset,
            cls.generate_minutes_dataset,
            cls.generate_hours_dataset,
            cls.generate_years_dataset,
            cls.generate_tons_dataset,
            cls.generate_kilograms_dataset,
            cls.generate_grams_dataset,
            cls.generate_ponds_dataset,
            cls.generate_miles_dataset,
            cls.generate_kilometers_dataset,
            cls.generate_meters_dataset,
            cls.generate_centimeters_dataset,
            cls.generate_square_meters_dataset,
            cls.generate_square_kilometers_dataset,
            cls.generate_square_feet_dataset,
            cls.generate_square_inches_dataset,
            cls.generate_cubic_centimeters_dataset,
            cls.generate_cubic_meters_dataset,
            cls.generate_cubic_kilometers_dataset,
            cls.generate_celsius_dataset,
            cls.generate_fahrenheit_dataset,
            cls.generate_kelvin_dataset,
            cls.generate_miles_per_hour_dataset,
            cls.generate_meters_per_second_dataset,
            cls.generate_kilometers_per_hour_dataset,
            cls.generate_watts_dataset,
            cls.generate_kilowatts_dataset,
        ]

        generator_names = list(
            map(lambda x: x.__name__.replace("generate_", ""), generators)
        )

        data_filepaths = []
        query_patterns_filepaths = []
        for generator, generator_name in zip(generators, generator_names):
            texts, query_patterns = generator(names, numbers)

            df = pd.DataFrame({"name": names, "number": numbers, "text": texts})

            data_filepath = f"{save_dir}/{generator_name}.csv"
            df.to_csv(data_filepath, index=False)

            query_patterns_filepath = f"{save_dir}/{generator_name}_query_patterns.json"
            with open(query_patterns_filepath, "w") as f:
                json.dump(query_patterns, f, indent=4)

            data_filepaths.append(data_filepath)
            query_patterns_filepaths.append(query_patterns_filepath)

        return generator_names, data_filepaths, query_patterns_filepaths

    @staticmethod
    def generate_currency_dataset(names: list[str], numbers: list[int]):
        assert len(names) == len(
            numbers
        ), "The length of names and numbers must be the same."

        texts = []
        for name, number in zip(names, numbers):
            text = f"{name} are sold for {number} dollars."

            texts.append(text)

        query_patterns = {
            "equal": "What is sold for {number} dollars?",
            "less": "What is sold for less than {number} dollars?",
            "more": "What is sold for more than {number} dollars?",
            "around": "What is sold for around {number} dollars?",
            "between": "What is sold for between {number_min} and {number_max} dollars?",
            "none": "What is sold",
        }

        return texts, query_patterns

    @staticmethod
    def generate_minutes_dataset(names: list[str], numbers: list[int]):
        assert len(names) == len(
            numbers
        ), "The length of names and numbers must be the same."

        texts = []
        for name, number in zip(names, numbers):
            text = f"{name} operate for {number} minutes."

            texts.append(text)

        query_patterns = {
            "equal": "What operates for {number} minutes?",
            "less": "What operates for less than {number} minutes?",
            "more": "What operates for more than {number} minutes?",
            "around": "What operates for around {number} minutes?",
            "between": "What operates for between {number_min} and {number_max} minutes?",
            "none": "What operates",
        }

        return texts, query_patterns

    @staticmethod
    def generate_hours_dataset(names: list[str], numbers: list[int]):
        assert len(names) == len(
            numbers
        ), "The length of names and numbers must be the same."

        texts = []
        for name, number in zip(names, numbers):
            text = f"{name} operate for {number} hours."

            texts.append(text)

        query_patterns = {
            "equal": "What operates for {number} hours?",
            "less": "What operates for less than {number} hours?",
            "more": "What operates for more than {number} hours?",
            "around": "What operates for around {number} hours?",
            "between": "What operates for between {number_min} and {number_max} hours?",
            "none": "What operates",
        }

        return texts, query_patterns

    @staticmethod
    def generate_years_dataset(names: list[str], numbers: list[int]):
        assert len(names) == len(
            numbers
        ), "The length of names and numbers must be the same."

        texts = []
        for name, number in zip(names, numbers):
            text = f"{name} operate for {number} years."

            texts.append(text)

        query_patterns = {
            "equal": "What operates for {number} years?",
            "less": "What operates for less than {number} years?",
            "more": "What operates for more than {number} years?",
            "around": "What operates for around {number} years?",
            "between": "What operates for between {number_min} and {number_max} years?",
            "none": "What operates",
        }

        return texts, query_patterns

    @staticmethod
    def generate_tons_dataset(names: list[str], numbers: list[int]):
        assert len(names) == len(
            numbers
        ), "The length of names and numbers must be the same."

        texts = []
        for name, number in zip(names, numbers):
            text = f"{name} weigh {number} tons."

            texts.append(text)

        query_patterns = {
            "equal": "What weighs {number} tons?",
            "less": "What weighs less than {number} tons?",
            "more": "What weighs more than {number} tons?",
            "around": "What weighs around {number} tons?",
            "between": "What weighs between {number_min} and {number_max} tons?",
            "none": "What weighs",
        }

        return texts, query_patterns

    @staticmethod
    def generate_kilograms_dataset(names: list[str], numbers: list[int]):
        assert len(names) == len(
            numbers
        ), "The length of names and numbers must be the same."

        texts = []
        for name, number in zip(names, numbers):
            text = f"{name} weigh {number} kilograms."

            texts.append(text)

        query_patterns = {
            "equal": "What weighs {number} kilograms?",
            "less": "What weighs less than {number} kilograms?",
            "more": "What weighs more than {number} kilograms?",
            "around": "What weighs around {number} kilograms?",
            "between": "What weighs between {number_min} and {number_max} kilograms?",
            "none": "What weighs",
        }

        return texts, query_patterns

    @staticmethod
    def generate_grams_dataset(names: list[str], numbers: list[int]):
        assert len(names) == len(
            numbers
        ), "The length of names and numbers must be the same."

        texts = []
        for name, number in zip(names, numbers):
            text = f"{name} weigh {number} grams."

            texts.append(text)

        query_patterns = {
            "equal": "What weighs {number} grams?",
            "less": "What weighs less than {number} grams?",
            "more": "What weighs more than {number} grams?",
            "around": "What weighs around {number} grams?",
            "between": "What weighs between {number_min} and {number_max} grams?",
            "none": "What weighs",
        }

        return texts, query_patterns

    @staticmethod
    def generate_ponds_dataset(names: list[str], numbers: list[int]):
        assert len(names) == len(
            numbers
        ), "The length of names and numbers must be the same."

        texts = []
        for name, number in zip(names, numbers):
            text = f"{name} weigh {number} ponds."

            texts.append(text)

        query_patterns = {
            "equal": "What weighs {number} ponds?",
            "less": "What weighs less than {number} ponds?",
            "more": "What weighs more than {number} ponds?",
            "around": "What weighs around {number} ponds?",
            "between": "What weighs between {number_min} and {number_max} ponds?",
            "none": "What weighs",
        }

        return texts, query_patterns

    @staticmethod
    def generate_miles_dataset(names: list[str], numbers: list[int]):
        assert len(names) == len(
            numbers
        ), "The length of names and numbers must be the same."

        texts = []
        for name, number in zip(names, numbers):
            text = f"{name} ran {number} miles."

            texts.append(text)

        query_patterns = {
            "equal": "What ran {number} miles?",
            "less": "What ran less than {number} miles?",
            "more": "What ran more than {number} miles?",
            "around": "What ran around {number} miles?",
            "between": "What ran between {number_min} and {number_max} miles?",
            "none": "What ran",
        }

        return texts, query_patterns

    @staticmethod
    def generate_kilometers_dataset(names: list[str], numbers: list[int]):
        assert len(names) == len(
            numbers
        ), "The length of names and numbers must be the same."

        texts = []
        for name, number in zip(names, numbers):
            text = f"{name} ran {number} kilometers."

            texts.append(text)

        query_patterns = {
            "equal": "What ran {number} kilometers?",
            "less": "What ran less than {number} kilometers?",
            "more": "What ran more than {number} kilometers?",
            "around": "What ran around {number} kilometers?",
            "between": "What ran between {number_min} and {number_max} kilometers?",
            "none": "What ran",
        }

        return texts, query_patterns

    @staticmethod
    def generate_meters_dataset(names: list[str], numbers: list[int]):
        assert len(names) == len(
            numbers
        ), "The length of names and numbers must be the same."

        texts = []
        for name, number in zip(names, numbers):
            text = f"{name} ran {number} meters."

            texts.append(text)

        query_patterns = {
            "equal": "What ran {number} meters?",
            "less": "What ran less than {number} meters?",
            "more": "What ran more than {number} meters?",
            "around": "What ran around {number} meters?",
            "between": "What ran between {number_min} and {number_max} meters?",
            "none": "What ran",
        }

        return texts, query_patterns

    @staticmethod
    def generate_centimeters_dataset(names: list[str], numbers: list[int]):
        assert len(names) == len(
            numbers
        ), "The length of names and numbers must be the same."

        texts = []
        for name, number in zip(names, numbers):
            text = f"{name} ran {number} centimeters."

            texts.append(text)

        query_patterns = {
            "equal": "What ran {number} centimeters?",
            "less": "What ran less than {number} centimeters?",
            "more": "What ran more than {number} centimeters?",
            "around": "What ran around {number} centimeters?",
            "between": "What ran between {number_min} and {number_max} centimeters?",
            "none": "What ran",
        }

        return texts, query_patterns

    @staticmethod
    def generate_square_meters_dataset(names: list[str], numbers: list[int]):
        assert len(names) == len(
            numbers
        ), "The length of names and numbers must be the same."

        texts = []
        for name, number in zip(names, numbers):
            text = f"{name} has an area of {number} square meters."

            texts.append(text)

        query_patterns = {
            "equal": "What has an area of {number} square meters?",
            "less": "What has an area of less than {number} square meters?",
            "more": "What has an area of more than {number} square meters?",
            "around": "What has an area of around {number} square meters?",
            "between": "What has an area of between {number_min} and {number_max} square meters?",
            "none": "What has an area",
        }

        return texts, query_patterns

    @staticmethod
    def generate_square_kilometers_dataset(names: list[str], numbers: list[int]):
        assert len(names) == len(
            numbers
        ), "The length of names and numbers must be the same."

        texts = []
        for name, number in zip(names, numbers):
            text = f"{name} has an area of {number} square kilometers."

            texts.append(text)

        query_patterns = {
            "equal": "What has an area of {number} square kilometers?",
            "less": "What has an area of less than {number} square kilometers?",
            "more": "What has an area of more than {number} square kilometers?",
            "around": "What has an area of around {number} square kilometers?",
            "between": "What has an area of between {number_min} and {number_max} square kilometers?",
            "none": "What has an area",
        }

        return texts, query_patterns

    @staticmethod
    def generate_square_feet_dataset(names: list[str], numbers: list[int]):
        assert len(names) == len(
            numbers
        ), "The length of names and numbers must be the same."

        texts = []
        for name, number in zip(names, numbers):
            text = f"{name} has an area of {number} square feet."

            texts.append(text)

        query_patterns = {
            "equal": "What has an area of {number} square feet?",
            "less": "What has an area of less than {number} square feet?",
            "more": "What has an area of more than {number} square feet?",
            "around": "What has an area of around {number} square feet?",
            "between": "What has an area of between {number_min} and {number_max} square feet?",
            "none": "What has an area",
        }

        return texts, query_patterns

    @staticmethod
    def generate_square_inches_dataset(names: list[str], numbers: list[int]):
        assert len(names) == len(
            numbers
        ), "The length of names and numbers must be the same."

        texts = []
        for name, number in zip(names, numbers):
            text = f"{name} has an area of {number} square inches."

            texts.append(text)

        query_patterns = {
            "equal": "What has an area of {number} square inches?",
            "less": "What has an area of less than {number} square inches?",
            "more": "What has an area of more than {number} square inches?",
            "around": "What has an area of around {number} square inches?",
            "between": "What has an area of between {number_min} and {number_max} square inches?",
            "none": "What has an area",
        }

        return texts, query_patterns

    @staticmethod
    def generate_cubic_centimeters_dataset(names: list[str], numbers: list[int]):
        assert len(names) == len(
            numbers
        ), "The length of names and numbers must be the same."

        texts = []
        for name, number in zip(names, numbers):
            text = f"{name} has a volume of {number} cubic centimeters."

            texts.append(text)

        query_patterns = {
            "equal": "What has a volume of {number} cubic centimeters?",
            "less": "What has a volume of less than {number} cubic centimeters?",
            "more": "What has a volume of more than {number} cubic centimeters?",
            "around": "What has a volume of around {number} cubic centimeters?",
            "between": "What has a volume of between {number_min} and {number_max} cubic centimeters?",
            "none": "What has a volume",
        }

        return texts, query_patterns

    @staticmethod
    def generate_cubic_meters_dataset(names: list[str], numbers: list[int]):
        assert len(names) == len(
            numbers
        ), "The length of names and numbers must be the same."

        texts = []
        for name, number in zip(names, numbers):
            text = f"{name} has a volume of {number} cubic meters."

            texts.append(text)

        query_patterns = {
            "equal": "What has a volume of {number} cubic meters?",
            "less": "What has a volume of less than {number} cubic meters?",
            "more": "What has a volume of more than {number} cubic meters?",
            "around": "What has a volume of around {number} cubic meters?",
            "between": "What has a volume of between {number_min} and {number_max} cubic meters?",
            "none": "What has a volume",
        }

        return texts, query_patterns

    @staticmethod
    def generate_cubic_kilometers_dataset(names: list[str], numbers: list[int]):
        assert len(names) == len(
            numbers
        ), "The length of names and numbers must be the same."

        texts = []
        for name, number in zip(names, numbers):
            text = f"{name} has a volume of {number} cubic kilometers."

            texts.append(text)

        query_patterns = {
            "equal": "What has a volume of {number} cubic kilometers?",
            "less": "What has a volume of less than {number} cubic kilometers?",
            "more": "What has a volume of more than {number} cubic kilometers?",
            "around": "What has a volume of around {number} cubic kilometers?",
            "between": "What has a volume of between {number_min} and {number_max} cubic kilometers?",
            "none": "What has a volume",
        }

        return texts, query_patterns

    @staticmethod
    def generate_celsius_dataset(names: list[str], numbers: list[int]):
        assert len(names) == len(
            numbers
        ), "The length of names and numbers must be the same."

        texts = []
        for name, number in zip(names, numbers):
            text = f"{name} operates normally at {number} degrees Celsius."

            texts.append(text)

        query_patterns = {
            "equal": "What operates normally at {number} degrees Celsius?",
            "less": "What operates normally at less than {number} degrees Celsius?",
            "more": "What operates normally at more than {number} degrees Celsius?",
            "around": "What operates normally at around {number} degrees Celsius?",
            "between": "What operates normally at between {number_min} and {number_max} degrees Celsius?",
            "none": "What operates normally",
        }

        return texts, query_patterns

    @staticmethod
    def generate_fahrenheit_dataset(names: list[str], numbers: list[int]):
        assert len(names) == len(
            numbers
        ), "The length of names and numbers must be the same."

        texts = []
        for name, number in zip(names, numbers):
            text = f"{name} operates normally at {number} degrees Fahrenheit."

            texts.append(text)

        query_patterns = {
            "equal": "What operates normally at {number} degrees Fahrenheit?",
            "less": "What operates normally at less than {number} degrees Fahrenheit?",
            "more": "What operates normally at more than {number} degrees Fahrenheit?",
            "around": "What operates normally at around {number} degrees Fahrenheit?",
            "between": "What operates normally at between {number_min} and {number_max} degrees Fahrenheit?",
            "none": "What operates normally",
        }

        return texts, query_patterns

    @staticmethod
    def generate_kelvin_dataset(names: list[str], numbers: list[int]):
        assert len(names) == len(
            numbers
        ), "The length of names and numbers must be the same."

        texts = []
        for name, number in zip(names, numbers):
            text = f"{name} operates normally at {number} Kelvin."

            texts.append(text)

        query_patterns = {
            "equal": "What operates normally at {number} Kelvin?",
            "less": "What operates normally at less than {number} Kelvin?",
            "more": "What operates normally at more than {number} Kelvin?",
            "around": "What operates normally at around {number} Kelvin?",
            "between": "What operates normally at between {number_min} and {number_max} Kelvin?",
            "none": "What operates normally",
        }

        return texts, query_patterns

    @staticmethod
    def generate_miles_per_hour_dataset(names: list[str], numbers: list[int]):
        assert len(names) == len(
            numbers
        ), "The length of names and numbers must be the same."

        texts = []
        for name, number in zip(names, numbers):
            text = f"{name} runs at {number} miles per hour."

            texts.append(text)

        query_patterns = {
            "equal": "What runs at {number} miles per hour?",
            "less": "What runs at less than {number} miles per hour?",
            "more": "What runs at more than {number} miles per hour?",
            "around": "What runs at around {number} miles per hour?",
            "between": "What runs at between {number_min} and {number_max} miles per hour?",
            "none": "What runs",
        }

        return texts, query_patterns

    @staticmethod
    def generate_meters_per_second_dataset(names: list[str], numbers: list[int]):
        assert len(names) == len(
            numbers
        ), "The length of names and numbers must be the same."

        texts = []
        for name, number in zip(names, numbers):
            text = f"{name} runs at {number} meters per second."

            texts.append(text)

        query_patterns = {
            "equal": "What runs at {number} meters per second?",
            "less": "What runs at less than {number} meters per second?",
            "more": "What runs at more than {number} meters per second?",
            "around": "What runs at around {number} meters per second?",
            "between": "What runs at between {number_min} and {number_max} meters per second?",
            "none": "What runs",
        }

        return texts, query_patterns

    @staticmethod
    def generate_kilometers_per_hour_dataset(names: list[str], numbers: list[int]):
        assert len(names) == len(
            numbers
        ), "The length of names and numbers must be the same."

        texts = []
        for name, number in zip(names, numbers):
            text = f"{name} runs at {number} kilometers per hour."

            texts.append(text)

        query_patterns = {
            "equal": "What runs at {number} kilometers per hour?",
            "less": "What runs at less than {number} kilometers per hour?",
            "more": "What runs at more than {number} kilometers per hour?",
            "around": "What runs at around {number} kilometers per hour?",
            "between": "What runs at between {number_min} and {number_max} kilometers per hour?",
            "none": "What runs",
        }

        return texts, query_patterns

    @staticmethod
    def generate_watts_dataset(names: list[str], numbers: list[int]):
        assert len(names) == len(
            numbers
        ), "The length of names and numbers must be the same."

        texts = []
        for name, number in zip(names, numbers):
            text = f"{name} consumes {number} watts of power."

            texts.append(text)

        query_patterns = {
            "equal": "What consumes {number} watts of power?",
            "less": "What consumes less than {number} watts of power?",
            "more": "What consumes more than {number} watts of power?",
            "around": "What consumes around {number} watts of power?",
            "between": "What consumes between {number_min} and {number_max} watts of power?",
            "none": "What consumes",
        }

        return texts, query_patterns

    @staticmethod
    def generate_kilowatts_dataset(names: list[str], numbers: list[int]):
        assert len(names) == len(
            numbers
        ), "The length of names and numbers must be the same."

        texts = []
        for name, number in zip(names, numbers):
            text = f"{name} consumes {number} kilowatts of power."

            texts.append(text)

        query_patterns = {
            "equal": "What consumes {number} kilowatts of power?",
            "less": "What consumes less than {number} kilowatts of power?",
            "more": "What consumes more than {number} kilowatts of power?",
            "around": "What consumes around {number} kilowatts of power?",
            "between": "What consumes between {number_min} and {number_max} kilowatts of power?",
            "none": "What consumes",
        }

        return texts, query_patterns
