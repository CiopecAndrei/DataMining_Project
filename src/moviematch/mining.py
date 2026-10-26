import math

import pandas as pd
from mlxtend.frequent_patterns import association_rules, fpgrowth
from mlxtend.preprocessing import TransactionEncoder

from moviematch.config import (
    MAX_ITEMSET_SIZE,
    MIN_CONFIDENCE,
    MIN_LIFT,
    MIN_SUPPORT,
)
from moviematch.data import build_preference_baskets, load_dataset


RULE_COLUMNS = [
    "antecedents",
    "consequents",
    "support",
    "confidence",
    "lift",
]


def encode_baskets(baskets: pd.Series) -> pd.DataFrame:
    """Convert preference baskets into a sparse user-movie matrix."""
    if baskets.empty:
        raise ValueError("At least one preference basket is required.")

    transactions = []

    for basket in baskets:
        if not isinstance(basket, (list, tuple, set, frozenset)):
            raise ValueError(
                "Each preference basket must be a list, tuple, set, "
                "or frozenset of movie IDs."
            )

        transactions.append(list(basket))

    if not any(transactions):
        raise ValueError("At least one non-empty basket is required.")

    encoder = TransactionEncoder()
    encoded = encoder.fit(transactions).transform(
        transactions,
        sparse=True,
    )

    return pd.DataFrame.sparse.from_spmatrix(
        encoded,
        index=baskets.index,
        columns=encoder.columns_,
    )


def mine_patterns(
    baskets: pd.Series,
    min_support: float = MIN_SUPPORT,
    min_confidence: float = MIN_CONFIDENCE,
    min_lift: float = MIN_LIFT,
    max_len: int = MAX_ITEMSET_SIZE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Discover frequent itemsets and single-movie recommendation rules."""
    if not math.isfinite(min_support) or not 0 < min_support <= 1:
        raise ValueError(
            "min_support must be finite and greater than 0, "
            "and no greater than 1."
        )

    if not math.isfinite(min_confidence) or not 0 <= min_confidence <= 1:
        raise ValueError(
            "min_confidence must be finite and between 0 and 1."
        )

    if not math.isfinite(min_lift) or min_lift < 0:
        raise ValueError("min_lift must be finite and nonnegative.")

    if (
        isinstance(max_len, bool)
        or not isinstance(max_len, int)
        or max_len < 2
    ):
        raise ValueError("max_len must be an integer of at least 2.")

    encoded = encode_baskets(baskets)

    # Use string column labels for sparse mining input.
    # Preserve the original movie IDs for downstream recommendations.
    original_ids = {
        str(movie_id): movie_id
        for movie_id in encoded.columns
    }

    if len(original_ids) != len(encoded.columns):
        raise ValueError(
            "Movie IDs must remain unique when converted to strings."
        )

    mining_input = encoded.rename(columns=str)

    itemsets = fpgrowth(
        mining_input,
        min_support=min_support,
        use_colnames=True,
        max_len=max_len,
    )

    if itemsets.empty:
        return itemsets, pd.DataFrame(columns=RULE_COLUMNS)

    # Restore original IDs rather than exposing string IDs downstream.
    itemsets["itemsets"] = itemsets["itemsets"].map(
        lambda items: frozenset(original_ids[item] for item in items)
    )

    # Singleton itemsets cannot produce association rules.
    if not itemsets["itemsets"].map(len).ge(2).any():
        return itemsets, pd.DataFrame(columns=RULE_COLUMNS)

    rules = association_rules(
        itemsets,
        num_itemsets=len(encoded),
        metric="confidence",
        min_threshold=min_confidence,
    )

    if rules.empty:
        return itemsets, pd.DataFrame(columns=RULE_COLUMNS)

    # Each rule recommends exactly one movie.
    # Strictly greater than MIN_LIFT preserves your original behavior:
    # with MIN_LIFT = 1.0, rules must have positive association.
    mask = (
        rules["consequents"].map(len).eq(1)
        & rules["lift"].gt(min_lift)
    )

    rules = rules.loc[mask, RULE_COLUMNS].copy()

    rules = rules.sort_values(
        by=["confidence", "lift", "support"],
        ascending=[False, False, False],
        kind="stable",
    ).reset_index(drop=True)

    return itemsets, rules


def main() -> None:
    """Load preferences, mine patterns, and display a rule preview."""
    movies, ratings = load_dataset()
    baskets = build_preference_baskets(ratings)

    itemsets, rules = mine_patterns(baskets)

    print(f"Preference baskets: {len(baskets):,}")
    print(f"Frequent itemsets: {len(itemsets):,}")
    print(f"Recommendation rules: {len(rules):,}")

    if itemsets.empty:
        print(
            "\nNo frequent itemsets found. "
            "Try lowering MIN_SUPPORT."
        )
        return

    if not itemsets["itemsets"].map(len).ge(2).any():
        print(
            "\nOnly single-movie itemsets were found. "
            "Try lowering MIN_SUPPORT to discover combinations."
        )
        return

    if rules.empty:
        print(
            "\nNo single-movie rules passed the confidence and lift "
            "thresholds. Consider lowering MIN_CONFIDENCE or MIN_LIFT."
        )
        return

    titles = movies.set_index("movieId")["title"].to_dict()

    def format_movies(movie_ids: frozenset) -> str:
        return " + ".join(
            str(titles.get(movie_id, f"Unknown movie ({movie_id})"))
            for movie_id in sorted(movie_ids)
        )

    preview = rules.head(10).copy()
    preview["antecedents"] = preview["antecedents"].map(format_movies)
    preview["consequents"] = preview["consequents"].map(format_movies)

    print("\nTop 10 rules:")
    print(
        preview.to_string(
            index=False,
            float_format=lambda value: f"{value:.3f}",
        )
    )


if __name__ == "__main__":
    main()