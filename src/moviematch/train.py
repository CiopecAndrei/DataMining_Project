import json
from pathlib import Path

from moviematch.data import build_preference_baskets, load_dataset
from moviematch.mining import mine_patterns


ARTIFACTS_DIR = Path(__file__).resolve().parents[2] / "artifacts"


def main() -> None:
    movies, ratings = load_dataset()
    baskets = build_preference_baskets(ratings)

    print("Mining preference patterns...")
    itemsets, rules = mine_patterns(baskets)

    if rules.empty:
        raise ValueError(
            "No recommendation rules were generated. "
            "Review the mining thresholds before training again."
        )

    serializable_rules = rules.copy()

    for column in ("antecedents", "consequents"):
        serializable_rules[column] = serializable_rules[column].map(
            lambda values: sorted(int(value) for value in values)
        )

    # Keep catalogue and rules together as one consistent model artifact.
    model = {
        "movies": json.loads(movies.to_json(orient="records")),
        "rules": json.loads(
            serializable_rules.to_json(orient="records")
        ),
    }

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = ARTIFACTS_DIR / "model.json"
    temporary_path = ARTIFACTS_DIR / "model.json.tmp"

    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(
            model,
            file,
            ensure_ascii=False,
            allow_nan=False,
        )

    temporary_path.replace(model_path)

    print(f"Frequent itemsets: {len(itemsets):,}")
    print(f"Saved rules: {len(rules):,}")
    print(f"Model saved to: {model_path}")


if __name__ == "__main__":
    main()