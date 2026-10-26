import pandas as pd

from moviematch.recommender import recommend_movies


def test_recommendations_match_subsets_and_exclude_selected():
    movies = pd.DataFrame(
        {
            "movieId": [1, 2, 3, 4, 5],
            "title": ["Movie A", "Movie B", "Movie C", "Movie D", "Movie E"],
        }
    )

    rules = pd.DataFrame(
        [
            {
                "antecedents": frozenset({1}),
                "consequents": frozenset({4}),
                "support": 0.10,
                "confidence": 0.60,
                "lift": 1.50,
            },
            {
                "antecedents": frozenset({1, 2}),
                "consequents": frozenset({4}),
                "support": 0.08,
                "confidence": 0.80,
                "lift": 2.00,
            },
            {
                "antecedents": frozenset({1}),
                "consequents": frozenset({2}),
                "support": 0.12,
                "confidence": 0.90,
                "lift": 2.10,
            },
            {
                "antecedents": frozenset({5}),
                "consequents": frozenset({3}),
                "support": 0.05,
                "confidence": 0.70,
                "lift": 1.80,
            },
        ]
    )

    result = recommend_movies([1, 2, 3], movies, rules)

    assert result["movieId"].tolist() == [4]
    assert result.iloc[0]["confidence"] == 0.80
    assert "Movie A" in result.iloc[0]["explanation"]
    assert "Movie B" in result.iloc[0]["explanation"]


def test_no_matching_rules_returns_empty_result():
    movies = pd.DataFrame(
        {
            "movieId": [1, 2, 3],
            "title": ["Movie A", "Movie B", "Movie C"],
        }
    )

    rules = pd.DataFrame(
        [
            {
                "antecedents": frozenset({2}),
                "consequents": frozenset({3}),
                "support": 0.10,
                "confidence": 0.60,
                "lift": 1.50,
            },
        ]
    )

    result = recommend_movies([1], movies, rules)

    assert result.empty
    assert "explanation" in result.columns