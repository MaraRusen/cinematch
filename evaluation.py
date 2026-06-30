"""Evaluation metrics: Precision@K, Recall@K, NDCG@K, MRR, Hit Rate,
Catalog Coverage, Novelty, Average Popularity (popularity bias),
and Intra-List Diversity (ILD)."""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from data_loader import USER_COL, ITEM_COL, RATING_COL


def precision_at_k(recs, relevant, k=10):
    """|hits in top-k| / k"""
    hits = len(set(recs[:k]) & set(relevant))
    return hits / k if k > 0 else 0.0


def recall_at_k(recs, relevant, k=10):
    """|hits in top-k| / |all relevant|"""
    if not relevant:
        return 0.0
    hits = len(set(recs[:k]) & set(relevant))
    return hits / len(relevant)


def hit_rate_at_k(recs, relevant, k=10):
    """1 if any hit in top-k, else 0."""
    return 1.0 if set(recs[:k]) & set(relevant) else 0.0


def dcg_at_k(relevance, k=10):
    """DCG@K = Σ rel_i / log2(i+1), rank i starts at 1."""
    return sum(r / np.log2(i + 2) for i, r in enumerate(relevance[:k]))


def ndcg_at_k(recs, relevant, k=10):
    """NDCG@K = DCG / ideal DCG (binary relevance)."""
    rel_set   = set(relevant)
    relevance = [1 if r in rel_set else 0 for r in recs[:k]]
    dcg       = dcg_at_k(relevance, k)
    ideal     = [1] * min(k, len(relevant)) + [0] * (k - min(k, len(relevant)))
    idcg      = dcg_at_k(ideal, k)
    return dcg / idcg if idcg > 0 else 0.0


def mrr_at_k(recs, relevant, k=10):
    """Reciprocal rank of the first hit in top-k."""
    rel_set = set(relevant)
    for rank, item in enumerate(recs[:k], start=1):
        if item in rel_set:
            return 1.0 / rank
    return 0.0


def catalog_coverage(all_recs, all_items):
    """Fraction of catalogue recommended at least once across all users."""
    recommended = set(item for recs in all_recs for item in recs)
    return len(recommended) / len(all_items) if all_items else 0.0


def novelty_at_k(recs, item_popularity, k=10):
    """Mean self-information of recommended items.

    novelty(i) = -log2( pop(i) / Σ_j pop(j) )

    Higher score means the model recommends more niche (less popular) items.
    Based on: Zhou et al. (2010) "Solving the apparent diversity-accuracy dilemma".
    """
    if not recs or not item_popularity:
        return 0.0
    total = sum(item_popularity.values())
    scores = [-np.log2(item_popularity.get(item, 1) / total + 1e-10)
              for item in recs[:k]]
    return float(np.mean(scores))


def avg_popularity_at_k(recs, item_popularity, k=10):
    """Average rating count of recommended items.

    Lower value = model is recommending more niche items (less popularity bias).
    Useful for diagnosing whether a method is just surfacing blockbusters.
    """
    if not recs:
        return 0.0
    return float(np.mean([item_popularity.get(item, 0) for item in recs[:k]]))


def intra_list_diversity(recs, item_vectors, k=10):
    """Intra-List Diversity (ILD) — average pairwise dissimilarity in top-k.

    ILD = (2 / (k*(k-1))) * Σ_{i<j} (1 - cosine_sim(v_i, v_j))

    Higher ILD means the list covers a broader range of content.
    item_vectors : dict {item_id: sparse or dense vector} or (item_ids, matrix)
                   If a tuple (item_ids, matrix) is passed, matrix rows map to item_ids.
    Based on: Ziegler et al. (2005) "Improving recommendation lists through topic
    diversification."
    """
    if not recs or item_vectors is None:
        return 0.0

    rec_list = recs[:k]

    # Support both dict and (item_ids, matrix) forms
    if isinstance(item_vectors, tuple):
        item_ids, matrix = item_vectors
        id2row = {iid: i for i, iid in enumerate(item_ids)}
        rows = [id2row[iid] for iid in rec_list if iid in id2row]
        if len(rows) < 2:
            return 0.0
        vecs = matrix[rows]
    else:
        vecs_list = [item_vectors[iid] for iid in rec_list if iid in item_vectors]
        if len(vecs_list) < 2:
            return 0.0
        import scipy.sparse as sp
        if sp.issparse(vecs_list[0]):
            import scipy.sparse as sp2
            vecs = sp2.vstack(vecs_list)
        else:
            vecs = np.vstack(vecs_list)

    sim_matrix = cosine_similarity(vecs)
    n = sim_matrix.shape[0]
    # Sum upper triangle (pairwise similarities), divide by number of pairs
    upper = sim_matrix[np.triu_indices(n, k=1)]
    avg_sim = float(np.mean(upper)) if len(upper) > 0 else 1.0
    return 1.0 - avg_sim  # dissimilarity


def evaluate(model, train, test, users, k=10, model_name="",
             item_popularity=None, item_vectors=None, rel_threshold=3.5):
    """Evaluate a model over a list of users and return a results dict.

    Parameters
    ----------
    model           : fitted recommender with .recommend(user, train, n, exclude_seen)
    train, test     : DataFrames from data_loader.split()
    users           : list of user IDs to evaluate
    k               : cut-off rank
    model_name      : if set, prints results to stdout
    item_popularity : dict {movieId: n_ratings} – required for Novelty + AvgPopularity
    item_vectors    : (item_ids, matrix) tuple or dict – required for ILD
    rel_threshold   : minimum rating to count as "relevant" (default 3.5).
                      Using all test items regardless of rating inflates recall
                      because a 0.5-star item is treated as equally relevant as
                      a 5-star one.  3.5 is the standard threshold in RS literature
                      (e.g. Herlocker et al. 2004).
    """
    p_list, r_list, ndcg_list, mrr_list, hr_list = [], [], [], [], []
    novelty_list, pop_list, ild_list = [], [], []
    all_recs = []

    for user_id in users:
        user_test = test[test[USER_COL] == user_id]
        relevant  = set(user_test.loc[user_test[RATING_COL] >= rel_threshold, ITEM_COL])
        if not relevant:
            continue
        try:
            recs = model.recommend(user_id, train, n=k, exclude_seen=True)
        except Exception:
            continue
        if not recs:
            continue

        all_recs.append(recs)
        p_list.append(precision_at_k(recs, relevant, k))
        r_list.append(recall_at_k(recs, relevant, k))
        ndcg_list.append(ndcg_at_k(recs, relevant, k))
        mrr_list.append(mrr_at_k(recs, relevant, k))
        hr_list.append(hit_rate_at_k(recs, relevant, k))

        if item_popularity:
            novelty_list.append(novelty_at_k(recs, item_popularity, k))
            pop_list.append(avg_popularity_at_k(recs, item_popularity, k))
        if item_vectors is not None:
            ild_list.append(intra_list_diversity(recs, item_vectors, k))

    all_items = set(train[ITEM_COL]) | set(test[ITEM_COL])
    coverage  = catalog_coverage(all_recs, all_items)

    results = {
        f"P@{k}":          np.mean(p_list)       if p_list       else 0.0,
        f"R@{k}":          np.mean(r_list)        if r_list       else 0.0,
        f"NDCG@{k}":       np.mean(ndcg_list)     if ndcg_list    else 0.0,
        f"MRR@{k}":        np.mean(mrr_list)      if mrr_list     else 0.0,
        f"HR@{k}":         np.mean(hr_list)       if hr_list      else 0.0,
        "Coverage":        coverage,
        "Novelty":         np.mean(novelty_list)  if novelty_list else 0.0,
        "AvgPopularity":   np.mean(pop_list)      if pop_list     else 0.0,
        "ILD":             np.mean(ild_list)      if ild_list     else 0.0,
        "Users_eval":      len(p_list),
    }

    if model_name:
        print(f"\n{model_name}")
        for key, val in results.items():
            print(f"  {key}: {val:.4f}" if isinstance(val, float) else f"  {key}: {val}")

    return results
