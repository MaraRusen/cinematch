"""Non-personalised recommenders: most popular, highest rated, random."""

import numpy as np
from data_loader import USER_COL, ITEM_COL, RATING_COL, seen_items


class MostPopular:
    """Recommend the most frequently rated items."""

    def fit(self, ratings):
        counts = ratings.groupby(ITEM_COL).size()
        self.ranking_ = counts.sort_values(ascending=False).index.tolist()
        return self

    def recommend(self, user_id, ratings_train, n=10, exclude_seen=True):
        seen = seen_items(ratings_train, user_id) if exclude_seen else set()
        return [m for m in self.ranking_ if m not in seen][:n]


class HighestRated:
    """Recommend items with the highest average rating (minimum vote threshold)."""

    def __init__(self, min_ratings=20):
        self.min_ratings = min_ratings

    def fit(self, ratings):
        agg = ratings.groupby(ITEM_COL)[RATING_COL].agg(avg="mean", n="count")
        agg = agg[agg["n"] >= self.min_ratings]
        self.ranking_ = agg.sort_values("avg", ascending=False).index.tolist()
        return self

    def recommend(self, user_id, ratings_train, n=10, exclude_seen=True):
        seen = seen_items(ratings_train, user_id) if exclude_seen else set()
        return [m for m in self.ranking_ if m not in seen][:n]


class Random:
    """Recommend random unseen items (lower-bound baseline)."""

    def __init__(self, random_state=42):
        self.random_state = random_state

    def fit(self, ratings):
        self.items_ = ratings[ITEM_COL].unique().tolist()
        return self

    def recommend(self, user_id, ratings_train, n=10, exclude_seen=True):
        rng = np.random.default_rng(self.random_state)
        seen = seen_items(ratings_train, user_id) if exclude_seen else set()
        candidates = [m for m in self.items_ if m not in seen]
        n = min(n, len(candidates))
        return rng.choice(candidates, size=n, replace=False).tolist()


if __name__ == "__main__":
    from data_loader import load_ratings, load_movies, split, describe
    ratings = load_ratings()
    movies  = load_movies()
    train, test = split(ratings)
    title = movies.set_index("movieId")["title"].to_dict()

    for ModelClass, kwargs in [(MostPopular, {}), (HighestRated, {"min_ratings": 20}), (Random, {})]:
        model = ModelClass(**kwargs).fit(train)
        recs = model.recommend(1, train, n=5)
        names = [title.get(r, str(r)) for r in recs]
        print(f"\n{ModelClass.__name__}: {names}")
