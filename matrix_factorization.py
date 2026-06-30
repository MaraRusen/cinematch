"""Matrix factorization via TruncatedSVD (scikit-learn, no extra installs).

Note on TruncatedSVD vs. proper Matrix Factorization
-----------------------------------------------------
This implementation uses sklearn's TruncatedSVD, which decomposes the *dense*
user–item matrix (missing entries filled with 0) via a randomised SVD.
It differs from proper latent-factor MF (e.g. ALS, BPR, SGD-MF) in three ways:

1. Missing = zero: "not seen" is conflated with "rated 0 stars", biasing the
   decomposition towards items users never watched.
2. No bias terms: global mean and per-user / per-item offsets are not modelled
   separately, so predicted scores are not interpretable as ratings.
3. No regularisation: no L2 penalty is applied, which can overfit dense regions.

A production MF approach (e.g. NMF, implicit ALS, or the surprise SGD-MF) would
address all three limitations.  TruncatedSVD is used here for reproducibility
without additional dependencies."""

import numpy as np
from sklearn.decomposition import TruncatedSVD
from data_loader import USER_COL, ITEM_COL, RATING_COL, seen_items


class MatrixFactorization:
    """SVD-based matrix factorization using sklearn's TruncatedSVD.

    Decomposes the user-item rating matrix R ≈ U · Σ · V^T using a
    randomised SVD (fast, memory-efficient). The reconstructed matrix
    R_hat = U · Σ · V^T gives a predicted score for every (user, item) pair.

    For a given user we take their row in R_hat, rank all unseen items
    by predicted score, and return the top-n.
    """

    def __init__(self, n_factors=50, random_state=42):
        self.n_factors    = n_factors
        self.random_state = random_state

    def fit(self, ratings):
        matrix = ratings.pivot_table(
            index=USER_COL, columns=ITEM_COL, values=RATING_COL, fill_value=0
        )
        self.user_ids_ = matrix.index.tolist()
        self.item_ids_ = matrix.columns.tolist()
        self.u2i_ = {u: i for i, u in enumerate(self.user_ids_)}

        R = matrix.values.astype(np.float32)

        svd = TruncatedSVD(n_components=self.n_factors, n_iter=5,
                           random_state=self.random_state)
        U_sigma = svd.fit_transform(R)          # (n_users, n_factors)
        Vt      = svd.components_               # (n_factors, n_items)

        # Store the full reconstructed matrix — one row lookup per user at inference
        self.predicted_ = (U_sigma @ Vt).astype(np.float32)  # (n_users, n_items)
        return self

    def predict_score(self, user_id, item_id):
        """Return the SVD-reconstructed score for a (user, item) pair."""
        if user_id not in self.u2i_:
            return None
        try:
            item_idx = self.item_ids_.index(item_id)
        except ValueError:
            return None
        return float(self.predicted_[self.u2i_[user_id], item_idx])

    def recommend(self, user_id, ratings_train, n=10, exclude_seen=True):
        if user_id not in self.u2i_:
            return []

        seen   = seen_items(ratings_train, user_id) if exclude_seen else set()
        u_idx  = self.u2i_[user_id]
        scores = self.predicted_[u_idx]         # (n_items,)

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
    from data_loader import load_ratings, load_movies, split
    ratings = load_ratings()
    movies  = load_movies()
    train, _ = split(ratings)
    title = movies.set_index("movieId")["title"].to_dict()

    print("Fitting MatrixFactorization (SVD)...")
    mf = MatrixFactorization(n_factors=50).fit(train)
    recs = mf.recommend(1, train, n=5)
    print("SVD recs:", [title.get(r) for r in recs])
