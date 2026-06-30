"""Run all recommenders, evaluate, and print a comparison table."""

import pandas as pd
from data_loader          import (load_ratings, load_movies, load_tags,
                                   split, describe, TOP_K, USER_COL, ITEM_COL)
from baselines            import MostPopular, HighestRated, Random
from collaborative        import ItemItemCF, UserUserCF
from content_based        import ContentBased
from matrix_factorization import MatrixFactorization
from evaluation           import evaluate

# ── 1. Data ───────────────────────────────────────────────────────────────────
ratings = load_ratings()
movies  = load_movies()
tags    = load_tags()
describe(ratings, movies)

train, test = split(ratings)
print(f"\nTrain: {len(train):,}  |  Test: {len(test):,}")

# All users present in both train and test (no [:50] shortcut)
eval_users = list(
    set(test[USER_COL].unique()) & set(train[USER_COL].unique())
)
print(f"Evaluating on {len(eval_users)} users (all eligible)\n")

# Item popularity dict for novelty / popularity-bias metrics
item_popularity = train.groupby(ITEM_COL).size().to_dict()

# Item vectors for ILD — extracted after ContentBased is fitted below
item_vectors = None  # will be set after cb.fit()

# ── 2. Models ─────────────────────────────────────────────────────────────────
title = movies.set_index("movieId")["title"].to_dict()

print("Fitting models...")

pop   = MostPopular().fit(train)
rated = HighestRated(min_ratings=20).fit(train)
rand  = Random().fit(train)

print("  Fitting ItemItemCF  (may take ~10 sec)...")
ii_cf = ItemItemCF(k=20).fit(train)

print("  Fitting UserUserCF  (may take ~10 sec)...")
uu_cf = UserUserCF(k=20).fit(train)

print("  Fitting ContentBased (genres + tags)...")
cb    = ContentBased().fit(train, movies, tags)
# (item_ids_, item_vecs_) tuple used for ILD across all models
item_vectors = (cb.item_ids_, cb.item_vecs_)

print("  Fitting MatrixFactorization (SVD)...")
mf    = MatrixFactorization(n_factors=50).fit(train)

# ── 3. Evaluate ───────────────────────────────────────────────────────────────
models = {
    "Random":       rand,
    "MostPopular":  pop,
    "HighestRated": rated,
    "ContentBased": cb,
    "ItemItemCF":   ii_cf,
    "UserUserCF":   uu_cf,
    "MatrixFact":   mf,
}

print("\nEvaluating at K = 5, 10, 20 ...")
all_k_results = {}
for k_val in [5, 10, 20]:
    k_results = {}
    for name, model in models.items():
        k_results[name] = evaluate(
            model, train, test, eval_users,
            k=k_val, model_name=(name if k_val == 10 else ""),
            item_popularity=item_popularity,
            item_vectors=item_vectors,
        )
    all_k_results[k_val] = k_results

# ── 4. Comparison table ───────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("COMPARISON TABLE — K=10 (primary), all metrics averaged over all eligible users")
df_k10 = pd.DataFrame(all_k_results[10]).T
print(df_k10.to_string())

# Build a wide results DataFrame with columns prefixed by K value
frames = []
for k_val, k_results in all_k_results.items():
    df_k = pd.DataFrame(k_results).T
    # Rename metric columns with K suffix where appropriate; keep beyond-accuracy shared
    rename = {}
    for col in df_k.columns:
        if col in ("Coverage", "Novelty", "AvgPopularity", "ILD", "Users_eval"):
            rename[col] = col  # already model-level, keep once
        else:
            rename[col] = col  # already contains K in name (e.g. P@10)
    frames.append(df_k)

# Save primary (K=10) results
df_k10.to_csv("results.csv")
print("\nSaved to results.csv (K=10)")

# Save multi-K results — accuracy metrics at K=5,10,20 + beyond-accuracy once
BEYOND_ACC = {"Coverage", "Novelty", "AvgPopularity", "ILD", "Users_eval"}
acc_frames = []
for k_val in [5, 10, 20]:
    df_k = pd.DataFrame(all_k_results[k_val]).T
    acc_cols = [c for c in df_k.columns if c not in BEYOND_ACC]
    acc_frames.append(df_k[acc_cols])

df_multi = pd.concat(acc_frames, axis=1)
# Append beyond-accuracy columns from K=10 (K-independent)
df_beyond = pd.DataFrame(all_k_results[10]).T[list(BEYOND_ACC)]
df_multi = pd.concat([df_multi, df_beyond], axis=1)
df_multi.index.name = "Model"
df_multi.to_csv("results_multi_k.csv")
print("Saved to results_multi_k.csv (K=5,10,20)")

print("\nK=5  vs K=10  vs K=20  — P@K comparison:")
for name in models:
    p5  = all_k_results[5][name].get("P@5",  0)
    p10 = all_k_results[10][name].get("P@10", 0)
    p20 = all_k_results[20][name].get("P@20", 0)
    print(f"  {name:20s}  P@5={p5:.4f}  P@10={p10:.4f}  P@20={p20:.4f}")

# ── 5. Example recs for 3 sample users ───────────────────────────────────────
print("\n" + "=" * 80)
print("EXAMPLE RECOMMENDATIONS — 3 sample users, top 5 per model")
print("=" * 80)
for sample_user in eval_users[:3]:
    print(f"\nUser {sample_user}:")
    for name, model in models.items():
        recs  = model.recommend(sample_user, train, n=5)
        names = [title.get(r, str(r)) for r in recs]
        print(f"  {name:15s}: {', '.join(names)}")
