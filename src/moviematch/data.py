from pathlib import Path

import pandas as pd

from moviematch.config import MIN_LIKED_RATING, MOVIES_PATH, RATINGS_PATH


def load_dataset(
    movies_path: Path = MOVIES_PATH,
    ratings_path: Path = RATINGS_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and validate the movie catalogue and user ratings."""
    for path in (movies_path, ratings_path):
        if not path.is_file():
            raise FileNotFoundError(f"Dataset file not found: {path}")

    movies = pd.read_csv(
        movies_path,
        usecols=["movieId", "title", "genres"],
    )
    ratings = pd.read_csv(
        ratings_path,
        usecols=["userId", "movieId", "rating", "timestamp"],
    )

    if movies.empty or ratings.empty:
        raise ValueError("The dataset files must not be empty.")

    if movies.isna().any().any() or ratings.isna().any().any():
        raise ValueError("The dataset contains missing values.")

    if movies["movieId"].duplicated().any():
        raise ValueError("The movie catalogue contains duplicate movie IDs.")

    if ratings.duplicated(["userId", "movieId"]).any():
        raise ValueError("The dataset contains duplicate user–movie ratings.")

    if not ratings["rating"].between(0.5, 5.0).all():
        raise ValueError("Ratings must be between 0.5 and 5.0.")

    if not ratings["movieId"].isin(movies["movieId"]).all():
        raise ValueError("Some ratings reference unknown movie IDs.")

    return movies, ratings

def build_preference_baskets(
    ratings: pd.DataFrame,
    min_rating: float = MIN_LIKED_RATING,
) -> pd.Series:
    """Return one basket of unique liked movie IDs per user."""
    if not 0.5 <= min_rating <= 5.0:
        raise ValueError("min_rating must be between 0.5 and 5.0.")

    user_ids = pd.Index(
        sorted(ratings["userId"].unique()),
        name="userId",
    )

    liked_ratings = ratings.loc[
        ratings["rating"] >= min_rating,
        ["userId", "movieId"],
    ]

    baskets = (
        liked_ratings.groupby("userId")["movieId"]
        .agg(lambda movie_ids: sorted(set(movie_ids)))
        .reindex(user_ids)
    )

    # Keep users with no liked movies as empty baskets.
    baskets = baskets.map(
        lambda basket: basket if isinstance(basket, list) else []
    )
    baskets.name = "liked_movie_ids"

    return baskets

def main() -> None:
    movies, ratings = load_dataset()
    baskets = build_preference_baskets(ratings)
    basket_sizes = baskets.map(len)

    print(f"Movies: {len(movies):,}")
    print(f"Ratings: {len(ratings):,}")
    print(f"Preference baskets: {len(baskets):,}")
    print(f"Empty baskets: {basket_sizes.eq(0).sum():,}")
    print(f"Average liked movies per user: {basket_sizes.mean():.1f}")

    print("\nFirst user's basket:")
    first_user_id = baskets.index[0]
    first_basket = baskets.iloc[0]

    print(f"User ID: {first_user_id}")
    print(f"Liked movie IDs (first 10): {first_basket[:10]}")

    movie_titles = movies.set_index("movieId")["title"]

    print("\nCorresponding titles:")
    for movie_id in first_basket[:10]:
        print(f"- {movie_titles.loc[movie_id]}")

if __name__ == "__main__":
    main()