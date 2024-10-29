import ast
import json

import pandas as pd


def make_movie_dataset(movies_metadata: pd.DataFrame):
    def extract_genre(genres):
        genres = json.loads(genres.replace("'", '"'))
        return [genre["name"] for genre in genres]

    movies_metadata["genres"] = movies_metadata["genres"].apply(extract_genre)

    def extract_production_companies(production_companies):
        production_companies = ast.literal_eval(production_companies)
        return [
            production_company["name"] for production_company in production_companies
        ]

    movies_metadata["production_companies"] = movies_metadata[
        "production_companies"
    ].apply(extract_production_companies)

    filtered_movies_metadata = movies_metadata[
        ["title", "overview", "genres", "production_companies", "revenue"]
    ]

    filtered_movies_metadata["revenue_int"] = filtered_movies_metadata[
        "revenue"
    ].astype(int)

    filtered_movies_metadata["short_overview"] = filtered_movies_metadata[
        "overview"
    ].apply(lambda x: f"{x[:100]}...")

    filtered_movies_metadata["revenue_int_k"] = filtered_movies_metadata[
        "revenue_int"
    ].apply(lambda x: x / 1000)
    filtered_movies_metadata["revenue_int_m"] = filtered_movies_metadata[
        "revenue_int"
    ].apply(lambda x: x / 1000000)

    return filtered_movies_metadata


def create_movie_datasets(
    save_dir: str, movies_metadata_path: str = "the-movies-dataset/movies_metadata.csv"
):
    movies_metadata = pd.read_csv(movies_metadata_path)

    movies_metadata = movies_metadata.dropna(subset=["overview", "revenue"])

    movies_metadata = movies_metadata[movies_metadata["revenue"] > 0]

    movies_metadata = movies_metadata[movies_metadata["original_language"] == "en"]

    movies_metadata = movies_metadata[movies_metadata["production_companies"] != "[]"]

    movies_metadata = movies_metadata[movies_metadata["genres"] != "[]"]

    movies_dataset = make_movie_dataset(movies_metadata.copy())

    movie_dataset_path = f"{save_dir}/movie_dataset.csv"

    movies_dataset.to_csv(movie_dataset_path, index=False)

    shuffled_movie_dataset_path = f"{save_dir}/shuffled_movie_dataset.csv"

    shuffled_filtered_movies_metadata = movies_metadata.copy()
    shuffled_filtered_movies_metadata["revenue"] = (
        shuffled_filtered_movies_metadata["revenue"].sample(frac=1).values
    )

    shuffled_movie_dataset = make_movie_dataset(shuffled_filtered_movies_metadata)

    shuffled_movie_dataset.to_csv(shuffled_movie_dataset_path, index=False)


def main(args):
    create_movie_datasets(args.save_dir, args.movies_metadata_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save_dir", type=str, default="numerical_datasets/movie_datasets"
    )
    parser.add_argument(
        "--movies_metadata_path",
        type=str,
        default="the-movies-dataset/movies_metadata.csv",
    )
    args = parser.parse_args()

    main(args)
