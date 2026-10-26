import pandas as pd
import pytest

from moviematch.data import build_preference_baskets


def test_build_preference_baskets():
    ratings = pd.DataFrame(
        {
            "userId": [1, 1, 1, 2, 3],
            "movieId": [10, 20, 10, 30, 40],
            "rating": [4.0, 3.5, 5.0, 2.0, 4.5],
        }
    )

    baskets = build_preference_baskets(ratings)

    assert baskets.to_dict() == {
        1: [10],  # Includes the threshold and removes duplicates.
        2: [],    # Preserves users without liked movies.
        3: [40],
    }
    assert baskets.index.name == "userId"
    assert baskets.name == "liked_movie_ids"


@pytest.mark.parametrize("min_rating", [0.0, 5.5])
def test_rejects_invalid_rating_threshold(min_rating):
    ratings = pd.DataFrame(columns=["userId", "movieId", "rating"])

    with pytest.raises(ValueError, match="min_rating"):
        build_preference_baskets(ratings, min_rating=min_rating)