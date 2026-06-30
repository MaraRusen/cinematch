"""Collaborative filtering: item-item and user-user."""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from data_loader import USER_COL, ITEM_COL, RATING_COL, seen_items


def _build_matrix(ratings):
    """Build a user x item matrix (missing = 0) and return it with ID mappings."""
    matrix = ratings.pivot_table(
        index=USER_COL, columns=ITEM_COL, values=RATING_COL, fill_value=0
    )
    user_ids = matrix.index.tolist()
    item_ids = matrix.columns.tolist()
    u2i = {u: i for i, u in enumerate(user_ids)}
    m2i = {m: i for i, m in enumerate(item_ids)}
    return matrix.values.astype(np.float32), user_ids, item_ids, u2i, m2i


class ItemItemCF:
    """Item-item collaborative filtering.

    Computes cosine similarity between all pairs of items (based on who rated
    them), then scores unseen items by a weighted average of their neighbours'
    ratings from the target user.

    score(u, i) = Σ sim(i,j)*r(u,j) / Σ |sim(i,j)|   (j = rated neighbours)

    min_support filters out items with fewer than min_support training raters.
    Without this, items rated by a single user get cosine similarity = 1.0 with
    every other item that same user rated (both vectors point in the same
    direction in user space), causing them to dominate recommendations with
    artificially inflated scores equal to the user's highest rating.
    """

    def __init__(self, k=20, min_support=5):
        self.k          = k
        self.min_support = min_support

    def fit(self, ratings):
        mat, self.user_ids_, self.item_ids_, self.u2i_, self.m2i_ = _build_matrix(ratings)
        self.user_item_ = mat                              # (n_users, n_items)

        # Items as rows → similarity between items
        self.item_sim_ = np.nan_to_num(
            cosine_similarity(mat.T), nan=0.0
        ).astype(np.float32)  # (n_items, n_items)

        # Zero out similarities involving items with too few raters.
        # Items with < min_support raters have unreliable similarity estimates:
        # a single shared rater gives cosine sim = 1.0 regardless of ratings.
        item_counts  = (mat > 0).sum(axis=0)               # (n_items,)
        popular_mask = item_counts >= self.min_support      # bool array
        self.item_sim_[~popular_mask, :] = 0.0
        self.item_sim_[:, ~popular_mask] = 0.0

        return self

    def recommend(self, user_id, ratings_train, n=10, exclude_seen=True):
        if user_id not in self.u2i_:
            return []

        seen = seen_items(ratings_train, user_id) if exclude_seen else set()
        u_idx = self.u2i_[user_id]
        user_vec  = self.user_item_[u_idx]          # (n_items,)
        rated_mask = (user_vec != 0).astype(np.float32)

        # Vectorised: numerator[i] = Σ_j sim[i,j]*r[j]  (rated j only)
        numerators   = self.item_sim_ @ (user_vec * rated_mask)
        denominators = np.abs(self.item_sim_) @ rated_mask
        # Use safe division to avoid 0/0 RuntimeWarning
        scores = np.divide(numerators, denominators,
                           out=np.zeros_like(numerators),
                           where=denominators > 0)

        order = np.argsort(scores)[::-1]
        recs = []
        for idx in order:
            item = self.item_ids_[idx]
            if item not in seen:
                recs.append(item)
            if len(recs) == n:
                break
        return recs


class UserUserCF:
    """User-user collaborative filtering (extension method).

    Finds users most similar to the target user and aggregates their ratings
    to predict scores for unseen items.
    """

    def __init__(self, k=20):
        self.k = k

    def fit(self, ratings):
        mat, self.user_ids_, self.item_ids_, self.u2i_, self.m2i_ = _build_matrix(ratings)
        self.user_item_ = mat
        self.user_sim_  = np.nan_to_num(
            cosine_similarity(mat), nan=0.0
        ).astype(np.float32)  # (n_users, n_users)
        return self

    def recommend(self, user_id, ratings_train, n=10, exclude_seen=True):
        if user_id not in self.u2i_:
            return []

        seen  = seen_items(ratings_train, user_id) if exclude_seen else set()
        u_idx = self.u2i_[user_id]

        sims = self.user_sim_[u_idx].copy()
        sims[u_idx] = -1  # exclude self

        k = min(self.k, len(self.user_ids_) - 1)
        top_k = np.argpartition(sims, -k)[-k:]    # top-k neighbour indices

        neighbour_sims    = sims[top_k]                         # (k,)
        neighbour_ratings = self.user_item_[top_k]              # (k, n_items)
        rated_mask        = (neighbour_ratings != 0).astype(np.float32)

        numerators   = neighbour_sims @ (neighbour_ratings * rated_mask)
        denominators = np.abs(neighbour_sims) @ rated_mask
        scores = np.divide(numerators, denominators,
                           out=np.zeros_like(numerators),
                           where=denominators > 0)

        order = np.argsort(scores)[::-1]
        recs = []
        for idx in order:
            item = self.item_ids_[idx]
            if item not in seen:
                recs.append(item)
            if len(recs) == n:
                break
        return recs


if __name__ == "__main__":
    from data_loader import load_ratings, load_movies, split
    ratings = load_ratings()
    movies  = load_movies()
    train, test = split(ratings)
    title = movies.set_index("movieId")["title"].to_dict()

    print("Fitting ItemItemCF...")
    ii = ItemItemCF(k=20).fit(train)
    recs = ii.recommend(1, train, n=5)
    print("ItemItem recs:", [title.get(r) for r in recs])

    print("Fitting UserUserCF...")
    uu = UserUserCF(k=20).fit(train)
    recs = uu.recommend(1, train, n=5)
    print("UserUser recs:", [title.get(r) for r in recs])
