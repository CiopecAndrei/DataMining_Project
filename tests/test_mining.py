import pandas as pd
import pytest

from moviematch.mining import encode_baskets, mine_patterns


@pytest.fixture
def baskets():
    # Small artificial baskets for testing, not project results.
    return pd.Series(
        [[10, 20], [10, 20], [10], []],
        index=pd.Index([1, 2, 3, 4], name="userId"),
    )


def test_encoding_preserves_empty_baskets(baskets):
    encoded = encode_baskets(baskets)

    assert encoded.shape == (4, 2)
    assert not encoded.loc[4].any()
    assert encoded.loc[1, 10]
    assert encoded.loc[1, 20]


def test_mining_returns_expected_rule(baskets):
    _, rules = mine_patterns(
        baskets,
        min_support=0.5,
        min_confidence=0.5,
        min_lift=1.0,
        max_len=2,
    )

    rule = rules.loc[
        rules["antecedents"].eq(frozenset({10}))
        & rules["consequents"].eq(frozenset({20}))
    ].iloc[0]

    assert rule["support"] == pytest.approx(0.5)
    assert rule["confidence"] == pytest.approx(2 / 3)
    assert rule["lift"] == pytest.approx(4 / 3)


def test_encoding_rejects_only_empty_baskets():
    with pytest.raises(ValueError, match="non-empty basket"):
        encode_baskets(pd.Series([[], []]))