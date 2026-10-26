from collections.abc import Iterable

import pandas as pd


RECOMMENDATION_COLUMNS = [
    "movieId",
    "title",
    "confidence",
    "lift",
    "support",
    "explanation",
]


def recommend_movies(
    liked_movie_ids: Iterable[int],
    movies: pd.DataFrame,
    rules: pd.DataFrame,
    top_n: int = 5,
) -> pd.DataFrame:
    """Recommend unseen movies using matching association rules."""
    if (
        isinstance(top_n, bool)
        or not isinstance(top_n, int)
        or top_n < 1
    ):
        raise ValueError("top_n must be a positive integer.")

    selected = frozenset(liked_movie_ids)

    if not selected:
        raise ValueError("Select at least one liked movie.")

    titles = movies.set_index("movieId")["title"].to_dict()

    unknown_ids = selected.difference(titles)

    if unknown_ids:
        raise ValueError(
            f"Unknown movie IDs: {sorted(unknown_ids)}"
        )

    if rules.empty:
        return pd.DataFrame(columns=RECOMMENDATION_COLUMNS)

    # A rule matches when all its antecedents are selected.
    matching = rules.loc[
        rules["antecedents"].map(
            lambda antecedents: bool(antecedents)
            and antecedents.issubset(selected)
        )
        & rules["consequents"].map(
            lambda consequents: len(consequents) == 1
            and consequents.isdisjoint(selected)
        )
    ].copy()

    if matching.empty:
        return pd.DataFrame(columns=RECOMMENDATION_COLUMNS)

    matching["movieId"] = matching["consequents"].map(
        lambda consequents: next(iter(consequents))
    )

    # Only recommend movies present in the catalogue.
    matching = matching.loc[
        matching["movieId"].isin(titles)
    ].copy()

    # Rank rules consistently with mining.py.
    # The movie ID provides a deterministic final tie-breaker.
    matching = matching.sort_values(
        by=["confidence", "lift", "support", "movieId"],
        ascending=[False, False, False, True],
        kind="stable",
    )

    # Several rules may recommend the same movie.
    # Keep its strongest matching rule.
    matching = (
        matching.drop_duplicates(subset="movieId")
        .head(top_n)
        .copy()
    )

    matching["title"] = matching["movieId"].map(titles)

    def explain(antecedents: frozenset) -> str:
        liked_titles = [
            str(titles[movie_id])
            for movie_id in sorted(antecedents)
        ]

        return (
            "Recommended from a preference pattern involving: "
            + " + ".join(liked_titles)
        )

    matching["explanation"] = matching["antecedents"].map(explain)

    return matching.loc[
        :, RECOMMENDATION_COLUMNS
    ].reset_index(drop=True)