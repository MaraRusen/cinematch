"""Content-based recommender using TF-IDF on MovieLens genres."""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from data_loader import USER_COL, ITEM_COL, RATING_COL, GENRES_COL, seen_items


class ContentBased:
    """Build item vectors from genres (TF-IDF), build a user profile from
    their centred ratings, and score unseen items by cosine similarity.

    User profile formula:
        profile(u) = Σ_i (r(u,i) - mean_r(u)) * vector(i)

    Items rated above the user's average pull the profile toward their genres;
    items rated below push away — so the profile captures genuine taste.
    """

    def fit(self, ratings, movies, tags=None):
        """
        Parameters
        ----------
        ratings : DataFrame  – training ratings (used to align items only)
        movies  : DataFrame  – must contain movieId and genres columns
        tags    : DataFrame or None – optional (userId, movieId, tag).
                  When provided, per-movie tags are concatenated with genres
                  so TF-IDF captures richer semantics beyond coarse genre labels.
        """
        # Drop movies with no genre info — zero vectors add noise
        m = movies[movies[GENRES_COL] != "(no genres listed)"].copy()

        # Genres are pipe-separated: "Action|Comedy" → "Action Comedy"
        m["genre_text"] = m[GENRES_COL].str.replace("|", " ", regex=False)

        if tags is not None and len(tags) > 0:
            # Aggregate all unique tags per movie into a single string
            tag_agg = (
                tags.groupby(ITEM_COL)["tag"]
                .apply(lambda x: " ".join(x.astype(str).str.lower().unique()))
                .reset_index(name="tag_text")
            )
            m = m.merge(tag_agg, on=ITEM_COL, how="left")
            m["tag_text"]  = m["tag_text"].fillna("")
            m["item_text"] = m["genre_text"] + " " + m["tag_text"]
        else:
            m["item_text"] = m["genre_text"]

        self.item_ids_  = m["movieId"].tolist()
        self.m2i_       = {mid: i for i, mid in enumerate(self.item_ids_)}

        # max_features caps vocabulary; min_df=1 keeps rare tags
        self.vectorizer_ = TfidfVectorizer(min_df=1, max_features=5000)
        self.item_vecs_  = self.vectorizer_.fit_transform(m["item_text"])
        return self

    def _user_profile(self, user_id, ratings_train):
        user_rows = ratings_train[ratings_train[USER_COL] == user_id]
        if user_rows.empty:
            return None
        mean_r = user_rows[RATING_COL].mean()

        # Keep only items present in the TF-IDF vocabulary
        in_vocab  = user_rows[ITEM_COL].isin(self.m2i_)
        user_rows = user_rows[in_vocab]
        if user_rows.empty:
            return None

        item_indices     = user_rows[ITEM_COL].map(self.m2i_).values
        centered_ratings = (user_rows[RATING_COL].values - mean_r).astype(np.float32)

        # Vectorised weighted sum — one sparse matrix multiply replaces the loop
        # item_vecs_[indices] → sparse (n, n_features)
        # centered_ratings @ sparse → dense (n_features,)
        vecs    = self.item_vecs_[item_indices]
        profile = np.asarray(centered_ratings @ vecs).flatten()
        return profile

    def recommend(self, user_id, ratings_train, n=10, exclude_seen=True):
        profile = self._user_profile(user_id, ratings_train)
        if profile is None or np.linalg.norm(profile) == 0:
            return []

        scores = cosine_similarity(profile.reshape(1, -1), self.item_vecs_).flatten()
        seen   = seen_items(ratings_train, user_id) if exclude_seen else set()

        order = np.argsort(scores)[::-1]
        recs  = []
        for idx in order:
            item = self.item_ids_[idx]
            if item not in seen:
                recs.append(item)
            if len(recs) == n:
                break
        return recs


if __name__ == "__main__":
    from data_loader import load_ratings, load_movies, load_tags, split
    ratings = load_ratings()
    movies  = load_movies()
    tags    = load_tags()
    train, _ = split(ratings)
    title = movies.set_index("movieId")["title"].to_dict()

    print("Fitting ContentBased (genres only)...")
    cb_basic = ContentBased().fit(train, movies)
    recs = cb_basic.recommend(1, train, n=5)
    print("  Genres only:    ", [title.get(r) for r in recs])

    print("Fitting ContentBased (genres + tags)...")
    cb_rich = ContentBased().fit(train, movies, tags)
    recs = cb_rich.recommend(1, train, n=5)
    print("  Genres + tags:  ", [title.get(r) for r in recs])
