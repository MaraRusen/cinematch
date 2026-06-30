"""CineMatch — Interactive Movie Recommender System
Streamlit UI for the ESADE Recommender Systems Individual Assignment.

Run with:
    cd "C:/Users/marar/10_ESADE/3. Term/Recommender Systems/Individual Assignment"
    streamlit run app.py
"""

import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

# Ensure module imports work regardless of working directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_loader import (
    load_ratings, load_movies, load_tags, split,
    USER_COL, ITEM_COL, RATING_COL, TIMESTAMP_COL,
)
from baselines            import MostPopular, HighestRated, Random
from collaborative        import ItemItemCF, UserUserCF
from content_based        import ContentBased
from matrix_factorization import MatrixFactorization
from evaluation           import evaluate, intra_list_diversity

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CineMatch",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Load & train everything once (cached for the session) ─────────────────────
@st.cache_resource(show_spinner="Training all models… first load takes ~60 seconds")
def load_everything():
    ratings = load_ratings()
    movies  = load_movies()
    tags    = load_tags()
    train, test = split(ratings)
    item_popularity = train.groupby(ITEM_COL).size().to_dict()

    cb_model = ContentBased().fit(train, movies, tags)
    item_vectors = (cb_model.item_ids_, cb_model.item_vecs_)

    models = {
        "Random":               Random().fit(train),
        "Most Popular":         MostPopular().fit(train),
        "Highest Rated":        HighestRated(min_ratings=20).fit(train),
        "Content-Based":        cb_model,
        "Item-Item CF":         ItemItemCF(k=20).fit(train),
        "User-User CF":         UserUserCF(k=20).fit(train),
        "Matrix Factorization": MatrixFactorization(n_factors=50).fit(train),
    }

    eval_users = list(set(test[USER_COL].unique()) & set(train[USER_COL].unique()))
    eval_results = {}
    for name, model in models.items():
        eval_results[name] = evaluate(
            model, train, test, eval_users,
            k=10, item_popularity=item_popularity,
            item_vectors=item_vectors,
            rel_threshold=3.5,
        )
    eval_df = pd.DataFrame(eval_results).T

    return ratings, movies, train, test, models, item_popularity, eval_df


ratings, movies, train, test, models, item_popularity, eval_df = load_everything()
title_map = movies.set_index(ITEM_COL)["title"].to_dict()
genre_map = movies.set_index(ITEM_COL)["genres"].to_dict()
all_users = sorted(ratings[USER_COL].unique().tolist())

# ── Sidebar navigation ────────────────────────────────────────────────────────
st.sidebar.title("🎬 CineMatch")
st.sidebar.caption("Recommender Systems · ESADE 2025/26")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Go to",
    ["📊 Dataset Insights", "🎯 Get Recommendations", "📈 Compare Models"],
)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Dataset Insights (EDA)
# ═══════════════════════════════════════════════════════════════════════════════
if page == "📊 Dataset Insights":
    st.title("Dataset Insights")
    st.markdown(
        "Exploratory analysis of the **MovieLens Small** dataset — "
        "100k ratings from 610 users on 9,742 movies."
    )

    # KPI cards
    n_users   = ratings[USER_COL].nunique()
    n_items   = ratings[ITEM_COL].nunique()
    n_ratings = len(ratings)
    sparsity  = 1 - n_ratings / (n_users * n_items)
    avg_r     = ratings[RATING_COL].mean()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Users",      f"{n_users:,}")
    c2.metric("Movies",     f"{n_items:,}")
    c3.metric("Ratings",    f"{n_ratings:,}")
    c4.metric("Sparsity",   f"{sparsity:.1%}")
    c5.metric("Avg Rating", f"{avg_r:.2f} ★")

    st.markdown("---")

    # Row 1: Rating distribution + Ratings per user
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Rating Distribution")
        counts = ratings[RATING_COL].value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.bar(counts.index, counts.values, width=0.4,
               color="#1f77b4", edgecolor="white")
        for x, y in zip(counts.index, counts.values):
            ax.text(x, y + 150, f"{y/1000:.1f}k", ha="center", fontsize=8)
        ax.set_xlabel("Rating (0.5–5)")
        ax.set_ylabel("Count")
        ax.set_title(f"Ratings skew positive — mean {avg_r:.2f}")
        plt.tight_layout()
        st.pyplot(fig); plt.close(fig)
        st.caption(
            "Users rate movies they chose to watch, so ratings skew positive. "
            "This selection bias affects all models."
        )

    with col2:
        st.subheader("Ratings per User")
        user_counts = ratings.groupby(USER_COL).size()
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.hist(user_counts, bins=40, color="#ff7f0e", edgecolor="white")
        ax.axvline(user_counts.median(), color="red", linestyle="--",
                   label=f"Median = {user_counts.median():.0f}")
        ax.axvline(user_counts.mean(), color="navy", linestyle=":",
                   label=f"Mean = {user_counts.mean():.0f}")
        ax.set_xlabel("Number of ratings")
        ax.set_ylabel("Number of users")
        ax.legend(fontsize=9)
        ax.set_title("Long tail: a few power users rate hundreds of movies")
        plt.tight_layout()
        st.pyplot(fig); plt.close(fig)
        st.caption(
            "Most users rated between 20–100 movies. "
            "Users with very few ratings are hard to serve (cold-start problem)."
        )

    st.markdown("---")

    # Row 2: Top movies + Genre distribution
    col3, col4 = st.columns(2)

    with col3:
        st.subheader("Top 15 Most Rated Movies")
        top_movies = (
            ratings.groupby(ITEM_COL).size()
            .nlargest(15).reset_index(name="count")
            .merge(movies[[ITEM_COL, "title"]], on=ITEM_COL)
        )
        top_movies["short"] = top_movies["title"].str.slice(0, 35)
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.barh(top_movies["short"][::-1], top_movies["count"][::-1],
                color="#2ca02c")
        ax.set_xlabel("Number of ratings")
        ax.set_title("Top movies receive 300–350 ratings each")
        plt.tight_layout()
        st.pyplot(fig); plt.close(fig)

    with col4:
        st.subheader("Genre Distribution")
        all_genres = movies["genres"].str.split("|").explode()
        genre_counts = (
            all_genres[all_genres != "(no genres listed)"]
            .value_counts().head(15)
        )
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.barh(genre_counts.index[::-1], genre_counts.values[::-1],
                color="#9467bd")
        ax.set_xlabel("Number of movies")
        ax.set_title("Drama and Comedy dominate the catalogue")
        plt.tight_layout()
        st.pyplot(fig); plt.close(fig)

    # Long-tail chart
    st.markdown("---")
    st.subheader("Long-Tail Popularity Distribution")
    item_counts = (
        ratings.groupby(ITEM_COL).size()
        .sort_values(ascending=False)
        .reset_index(name="count")
    )
    top10_threshold = int(len(item_counts) * 0.1)
    top10_share = item_counts["count"].iloc[:top10_threshold].sum() / item_counts["count"].sum()

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.fill_between(range(len(item_counts)), item_counts["count"].values,
                    alpha=0.4, color="#1f77b4")
    ax.plot(range(len(item_counts)), item_counts["count"].values,
            color="#1f77b4", linewidth=0.8)
    ax.axvline(top10_threshold, color="red", linestyle="--",
               label=f"Top 10% of movies → {top10_share:.0%} of ratings")
    ax.set_yscale("log")
    ax.set_xlabel("Movie rank (sorted by popularity)")
    ax.set_ylabel("Ratings (log scale)")
    ax.set_title("Classic long tail: most movies are rarely rated")
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig); plt.close(fig)

    st.info(
        f"**Sparsity: {sparsity:.1%}** — only {n_ratings:,} of "
        f"{n_users:,} × {n_items:,} = {n_users * n_items:,} possible ratings exist. "
        "This extreme sparsity is the core challenge; it motivates dimensionality "
        "reduction (Matrix Factorization) and content signals (Content-Based)."
    )


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Get Recommendations
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🎯 Get Recommendations":
    st.title("Get Recommendations")

    st.sidebar.markdown("---")
    st.sidebar.subheader("Settings")
    user_id = st.sidebar.selectbox("User ID", all_users, index=0)
    algo    = st.sidebar.selectbox(
        "Algorithm", list(models.keys()), index=4,
        help="Item-Item CF is selected by default"
    )
    n_recs  = st.sidebar.slider("Recommendations to show", 5, 20, 10)

    col_left, col_right = st.columns([1, 2])

    # ── Left: user profile ──────────────────────────────────────────────────
    with col_left:
        st.subheader(f"User {user_id} — Profile")
        user_train = (
            train[train[USER_COL] == user_id]
            .merge(movies[[ITEM_COL, "title", "genres"]], on=ITEM_COL, how="left")
            .sort_values(RATING_COL, ascending=False)
        )

        n_rated = len(user_train)
        m1, m2 = st.columns(2)
        m1.metric("Ratings given", n_rated)
        m2.metric("Avg rating",    f"{user_train[RATING_COL].mean():.2f}")

        if n_rated < 20:
            st.info(
                f"Cold-start user: only {n_rated} training ratings. "
                "Collaborative methods may have limited personalisation — "
                "Content-Based or Most Popular may give better results here."
            )

        st.markdown("**Top-rated movies:**")
        for _, row in user_train.head(8).iterrows():
            t = row["title"]
            t = t[:38] + "…" if len(t) > 38 else t
            st.markdown(f"- {t} &nbsp; `{row[RATING_COL]:.1f}`")

        genres_flat = user_train["genres"].str.split("|").explode()
        top_genres  = (
            genres_flat[genres_flat != "(no genres listed)"]
            .value_counts().head(5)
        )
        if len(top_genres):
            st.markdown("**Favourite genres:**")
            st.markdown(
                "  ".join(f"`{g}` ×{c}" for g, c in top_genres.items())
            )

    # ── Right: recommendations ──────────────────────────────────────────────
    with col_right:
        st.subheader(f"Top {n_recs} — {algo}")
        recs = models[algo].recommend(user_id, train, n=n_recs, exclude_seen=True)

        # For Matrix Factorization, we can show a predicted score
        mf_model = models.get("Matrix Factorization")
        show_pred_score = (algo == "Matrix Factorization" and mf_model is not None
                           and hasattr(mf_model, "predict_score"))

        if not recs:
            st.warning("No recommendations could be generated for this user.")
        else:
            for rank, item_id in enumerate(recs, start=1):
                t_str  = title_map.get(item_id, f"Movie {item_id}")
                g_str  = genre_map.get(item_id, "").replace("|", " · ")
                pop    = item_popularity.get(item_id, 0)

                r1, r2, r3, r4 = st.columns([0.4, 3.2, 1.0, 1.1])
                r1.markdown(f"**#{rank}**")
                r2.markdown(f"**{t_str}**  \n<small>{g_str}</small>",
                            unsafe_allow_html=True)
                if show_pred_score:
                    score = mf_model.predict_score(user_id, item_id)
                    r3.markdown(f"`pred: {score:.2f}`" if score is not None else "")
                else:
                    r3.markdown("")
                r4.markdown(f"`{pop} ratings`")
                st.divider()


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — Compare Models
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Compare Models":
    st.title("Model Comparison")
    st.markdown(
        "All 7 models evaluated on the **20 % holdout test set** "
        f"({int(eval_df['Users_eval'].iloc[0])} users · K = 10)."
    )

    # Metrics table
    display_df = eval_df.drop(columns=["Users_eval"], errors="ignore").round(4)
    st.dataframe(
        display_df.style
            .highlight_max(axis=0, color="#d4edda")
            .highlight_min(axis=0, color="#f8d7da"),
        use_container_width=True,
    )
    st.caption("Green = best per column · Red = worst per column")

    st.markdown("---")

    # Bar charts — accuracy metrics
    st.subheader("Accuracy Metrics")
    acc_metrics = [c for c in ["P@10", "R@10", "NDCG@10", "MRR@10", "HR@10"]
                   if c in display_df.columns]
    cols = st.columns(len(acc_metrics))
    for i, metric in enumerate(acc_metrics):
        with cols[i]:
            vals   = display_df[metric].astype(float).sort_values(ascending=True)
            colors = ["#2ca02c" if v == vals.max() else "#aec7e8" for v in vals]
            fig, ax = plt.subplots(figsize=(3.5, 3))
            ax.barh(vals.index, vals.values, color=colors)
            ax.set_title(metric, fontsize=11)
            ax.tick_params(axis="y", labelsize=7)
            plt.tight_layout()
            st.pyplot(fig); plt.close(fig)

    st.markdown("---")

    # Beyond-accuracy: Coverage + Novelty + Popularity bias + ILD
    st.subheader("Beyond-Accuracy Metrics")
    col_a, col_b, col_c, col_d = st.columns(4)

    with col_a:
        st.markdown("**Catalog Coverage**")
        st.caption("Fraction of all movies recommended at least once. Higher = more diverse.")
        vals = display_df["Coverage"].astype(float).sort_values(ascending=True)
        fig, ax = plt.subplots(figsize=(3, 3))
        ax.barh(vals.index, vals.values,
                color=["#2ca02c" if v == vals.max() else "#aec7e8" for v in vals])
        ax.set_xlim(0, 1)
        plt.tight_layout()
        st.pyplot(fig); plt.close(fig)

    with col_b:
        st.markdown("**Novelty**")
        st.caption("Mean self-information of recommended items. Higher = more niche picks.")
        vals = display_df["Novelty"].astype(float).sort_values(ascending=True)
        fig, ax = plt.subplots(figsize=(3, 3))
        ax.barh(vals.index, vals.values,
                color=["#2ca02c" if v == vals.max() else "#aec7e8" for v in vals])
        plt.tight_layout()
        st.pyplot(fig); plt.close(fig)

    with col_c:
        st.markdown("**Avg Popularity (Bias)**")
        st.caption("Avg rating count of recommended items. Lower = less blockbuster bias.")
        vals = display_df["AvgPopularity"].astype(float).sort_values(ascending=False)
        colors = ["#2ca02c" if v == vals.min() else "#f4a582" for v in vals]
        fig, ax = plt.subplots(figsize=(3, 3))
        ax.barh(vals.index, vals.values, color=colors)
        plt.tight_layout()
        st.pyplot(fig); plt.close(fig)

    with col_d:
        st.markdown("**Intra-List Diversity (ILD)**")
        st.caption("Avg pairwise dissimilarity in top-10. Higher = recommendations span more genres.")
        if "ILD" in display_df.columns:
            vals = display_df["ILD"].astype(float).sort_values(ascending=True)
            fig, ax = plt.subplots(figsize=(3, 3))
            ax.barh(vals.index, vals.values,
                    color=["#2ca02c" if v == vals.max() else "#aec7e8" for v in vals])
            ax.set_xlim(0, 1)
            plt.tight_layout()
            st.pyplot(fig); plt.close(fig)
        else:
            st.info("ILD not yet computed — re-run evaluation.")

    st.markdown("---")
    st.markdown("### Key Takeaways")
    st.markdown(
        "- **Matrix Factorization** dominates all accuracy metrics (P@10, NDCG@10, HR@10) "
        "by learning compact latent representations that capture taste patterns.\n"
        "- **Random** and **Content-Based** score highest on **Novelty** and **Coverage** — "
        "they surface long-tail items the popularity baselines ignore.\n"
        "- **Most Popular / Highest Rated** show the strongest **popularity bias** "
        "(high AvgPopularity) — they only recommend well-known movies.\n"
        "- **Content-Based** has the lowest **ILD (0.20)** — it clusters recommendations "
        "tightly by genre. **Most Popular** has the highest ILD (0.95) because popular "
        "movies span many genres.\n"
        "- There is an inherent **accuracy–novelty trade-off**: the most accurate models "
        "tend to recommend popular items, which limits discovery."
    )

    # Download button
    st.markdown("---")
    st.download_button(
        "⬇ Download results.csv",
        eval_df.to_csv(index=True),
        "results.csv",
        "text/csv",
    )
