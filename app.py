import json
from pathlib import Path
import re

import pandas as pd
import streamlit as st

from moviematch.recommender import recommend_movies


MODEL_PATH = (
    Path(__file__).resolve().parent / "artifacts" / "model.json"
)


@st.cache_data
def load_model(
    model_path: str,
    modified_at: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cache the saved model; reload when its modification time changes."""
    with Path(model_path).open(encoding="utf-8") as file:
        model = json.load(file)

    movies = pd.DataFrame(model["movies"])
    rules = pd.DataFrame(model["rules"])

    for column in ("antecedents", "consequents"):
        rules[column] = rules[column].map(frozenset)

    return movies, rules


def main() -> None:
    st.set_page_config(
        page_title="MovieMatch",
        page_icon="🎬",
        layout="centered",
    )

    st.title("MovieMatch")
    st.write(
        "Select 3–5 movies you watched and liked. "
        "Find your next movie through shared preference patterns."
    )

    if not MODEL_PATH.is_file():
        st.error(
            "No trained model found. Run "
            "`python -m moviematch.train` first."
        )
        st.stop()

    movies, rules = load_model(
        str(MODEL_PATH),
        MODEL_PATH.stat().st_mtime_ns,
    )

    # Keep IDs internally, but display only movie titles.
    titles = movies.set_index("movieId")["title"].to_dict()

    def title_sort_key(movie_id: int) -> tuple[str, int]:
        title = str(titles[movie_id]).strip()

        # Ignore leading punctuation when sorting, not when displaying.
        normalized_title = re.sub(r"^[^\w]+", "", title).casefold()

        return normalized_title, movie_id

    movie_ids = sorted(titles, key=title_sort_key)

    with st.form("recommendation_form"):
        selected_ids = st.multiselect(
            "Movies you liked",
            options=movie_ids,
            format_func=lambda movie_id: (
                f"{titles[movie_id]} · ID {movie_id}"
            ),
            max_selections=5,
            help="Type a movie title to search the catalogue.",
        )

        submitted = st.form_submit_button("Recommend movies")

    if not submitted:
        return

    if not 3 <= len(selected_ids) <= 5:
        st.warning("Please select between 3 and 5 movies.")
        return

    recommendations = recommend_movies(
        liked_movie_ids=selected_ids,
        movies=movies,
        rules=rules,
        top_n=5,
    )

    if recommendations.empty:
        st.info(
            "No matching preference rules were found for these movies. "
            "Try a different combination."
        )
        return

    st.subheader("Your recommendations")

    for position, row in enumerate(
        recommendations.itertuples(index=False),
        start=1,
    ):
        st.markdown(f"### {position}. {row.title}")
        st.write(row.explanation)

        with st.expander("Why this recommendation?"):
            st.write(
                f"**Confidence:** {row.confidence:.1%}\n\n"
                f"**Lift:** {row.lift:.2f}\n\n"
                f"**Support:** {row.support:.1%}"
            )
            st.caption(
                "These metrics describe patterns in the training data, "
                "not a guarantee that you will enjoy the movie."
            )

        st.divider()

    st.caption(
        "Recommendations use FP-Growth association rules. "
        "Selected movies are excluded from the results."
    )


if __name__ == "__main__":
    main()