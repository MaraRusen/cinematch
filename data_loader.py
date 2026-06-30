"""Load and split MovieLens data."""

from pathlib import Path
import pandas as pd

# Use absolute paths so this module works regardless of working directory
# (important when imported from a Streamlit app or Jupyter notebook)
_HERE = Path(__file__).parent

RATINGS_PATH  = str(_HERE / "ml-latest-small" / "ratings.csv")
MOVIES_PATH   = str(_HERE / "ml-latest-small" / "movies.csv")
TAGS_PATH     = str(_HERE / "ml-latest-small" / "tags.csv")

USER_COL      = "userId"
ITEM_COL      = "movieId"
RATING_COL    = "rating"
TIMESTAMP_COL = "timestamp"
TITLE_COL     = "title"
GENRES_COL    = "genres"
TAG_COL       = "tag"

RANDOM_STATE = 42
TOP_K        = 10


def load_ratings():
    df = pd.read_csv(RATINGS_PATH)
    assert all(c in df.columns for c in [USER_COL, ITEM_COL, RATING_COL, TIMESTAMP_COL])
    return df


def load_movies():
    df = pd.read_csv(MOVIES_PATH)
    assert all(c in df.columns for c in [ITEM_COL, TITLE_COL, GENRES_COL])
    return df


def load_tags():
    """Load user-generated tags (userId, movieId, tag, timestamp)."""
    df = pd.read_csv(TAGS_PATH)
    assert all(c in df.columns for c in [USER_COL, ITEM_COL, TAG_COL, TIMESTAMP_COL])
    return df


def split(ratings, test_frac=0.2):
    """Temporal train/test split (per-user).

    For each user, ratings are sorted by timestamp and the most recent
    `test_frac` fraction is held out as the test set.  This mirrors real
    deployment: models are trained on past interactions and evaluated on
    future ones, avoiding temporal leakage that a random shuffle would cause.

    Every user is guaranteed at least one training rating because we hold
    out floor(n * test_frac) items (minimum 1).
    """
    train_parts, test_parts = [], []
    for _, group in ratings.groupby(USER_COL):
        group_sorted = group.sort_values(TIMESTAMP_COL)
        n_test = max(1, int(len(group_sorted) * test_frac))
        train_parts.append(group_sorted.iloc[:-n_test])
        test_parts.append(group_sorted.iloc[-n_test:])

    train = pd.concat(train_parts).reset_index(drop=True)
    test  = pd.concat(test_parts).reset_index(drop=True)
    return train, test


def seen_items(ratings, user_id):
    return set(ratings.loc[ratings[USER_COL] == user_id, ITEM_COL])


def describe(ratings, movies):
    n_users   = ratings[USER_COL].nunique()
    n_items   = ratings[ITEM_COL].nunique()
    n_ratings = len(ratings)
    sparsity  = 1 - n_ratings / (n_users * n_items)

    print("=" * 45)
    print(f"Users:    {n_users:,}")
    print(f"Items:    {n_items:,}")
    print(f"Ratings:  {n_ratings:,}")
    print(f"Sparsity: {sparsity:.2%}")
    print(f"Avg rating: {ratings[RATING_COL].mean():.2f}")

    top = (ratings.groupby(ITEM_COL).size().nlargest(5)
           .reset_index(name="count")
           .merge(movies[[ITEM_COL, TITLE_COL]], on=ITEM_COL))
    print("\nTop 5 most rated movies:")
    for _, r in top.iterrows():
        print(f"  {r[TITLE_COL]} ({r['count']} ratings)")
    print("=" * 45)


if __name__ == "__main__":
    ratings = load_ratings()
    movies  = load_movies()
    describe(ratings, movies)
    train, test = split(ratings)
    print(f"Train: {len(train):,}  |  Test: {len(test):,}")
