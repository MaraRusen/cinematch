# CineMatch — Movie Recommender System

ESADE Recommender Systems · Individual Assignment · 2025/26

7 recommendation algorithms evaluated on MovieLens Small (100,836 ratings · 610 users · 9,742 movies).

---

## Algorithms

| Method | File |
|---|---|
| Random, MostPopular, HighestRated | `baselines.py` |
| Item-Item CF, User-User CF | `collaborative.py` |
| Content-Based (TF-IDF genres + tags) | `content_based.py` |
| Matrix Factorization (TruncatedSVD) | `matrix_factorization.py` |

---

## Setup

```bash
pip install -r requirements.txt
```

The dataset (`ml-latest-small/`) must be in the same folder. Download from [grouplens.org/datasets/movielens/latest](https://grouplens.org/datasets/movielens/latest/).

---

## Run

**Full evaluation pipeline** (trains all models, evaluates at K=5/10/20, saves `results.csv`):
```bash
python main.py
```

**Interactive Streamlit app** (Dataset Insights · Get Recommendations · Compare Models):
```bash
streamlit run app.py
```
Then open [http://localhost:8502](http://localhost:8502) in your browser.

**Exploratory Data Analysis notebook:**
```bash
jupyter notebook eda.ipynb
```

---

## Project Structure

```
├── data_loader.py          # Load data + temporal train/test split
├── baselines.py            # Non-personalised recommenders
├── collaborative.py        # Item-Item CF + User-User CF
├── content_based.py        # TF-IDF content-based filtering
├── matrix_factorization.py # SVD matrix factorization
├── evaluation.py           # 9 metrics: P/R/NDCG/MRR/HR/Coverage/Novelty/AvgPop/ILD
├── main.py                 # Full pipeline
├── app.py                  # Streamlit UI
├── eda.ipynb               # 9-section EDA notebook (with outputs)
├── results.csv             # K=10 results
├── results_multi_k.csv     # Results at K=5, 10, 20
└── report.html             # Full technical report (open in browser → print to PDF)
```

---

## Evaluation Design

- **Split:** Per-user temporal holdout — each user's most recent 20% of ratings as test set
- **Relevance threshold:** Rating ≥ 3.5 stars counts as "relevant"
- **Users evaluated:** 593 (those with at least one liked test item)
- **K values:** 5, 10, 20

## Key Results (K=10)

| Model | P@10 | NDCG@10 | HR@10 | Coverage | ILD |
|---|---|---|---|---|---|
| Random | 0.003 | 0.004 | 0.032 | 0.119 | 0.839 |
| MostPopular | 0.064 | 0.082 | 0.337 | 0.010 | 0.952 |
| HighestRated | 0.021 | 0.030 | 0.164 | 0.004 | 0.973 |
| ContentBased | 0.008 | 0.009 | 0.066 | 0.321 | 0.201 |
| ItemItemCF | 0.004 | 0.004 | 0.037 | 0.059 | 0.815 |
| UserUserCF | 0.010 | 0.014 | 0.088 | 0.156 | 0.883 |
| **MatrixFact** | **0.090** | **0.112** | **0.498** | 0.068 | 0.926 |
