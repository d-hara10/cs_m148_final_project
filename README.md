# CS M148 Final Project Report - Genre Classification from Spotify Audio Features

**Team:** Zachary Joseph, Daniel Hara, Ryan Persico, Zach Smith, Davis Frolich  
**Course:** CS M148  
**Github:** https://github.com/d-hara10/cs_m148_final_project/blob/main/README.md

## 1) Summary

We developed a supervised learning pipeline to **classify a song’s genre** from Spotify audio features (e.g., danceability, energy, acousticness). Because the raw dataset contains **114 genres**, we scoped the final task to a **balanced 10‑genre classification problem** to keep the label space learnable and the evaluation interpretable.

Our best-performing approach in this project was a **Random Forest classifier** trained on standardized cleaned audio features, which achieved **~0.74 test accuracy** on the 10‑genre task.

---

## 2) Dataset

We used the **Spotify Tracks Dataset** hosted on Hugging Face:
- Link: `https://huggingface.co/datasets/maharshipandya/spotify-tracks-dataset`
- Source: `maharshipandya/spotify-tracks-dataset`
- Rows: ~114k tracks
- Features: popularity, duration, danceability, energy, loudness, speechiness, acousticness, instrumentalness, liveness, valence, tempo, key, mode, time_signature, etc.
- Target label: `track_genre`

In code, the dataset is loaded directly via:

```python
df = pd.read_csv("hf://datasets/maharshipandya/spotify-tracks-dataset/dataset.csv", index_col=0)
```

---

## 3) Problem Statement

**Goal:** Predict `track_genre` from numeric audio features.

### Why we used a 10‑genre subset
The full dataset has **114 unique genres**, which makes the classification problem high-cardinality and harder to evaluate cleanly (confusion matrices become noisy; many genres are semantically overlapping). For the final model we restricted to 10 genres:

`pop, rock, hip-hop, electronic, classical, jazz, country, metal, reggae, comedy`

This subset is **balanced** in the provided dataset (each genre has the same number of samples), so overall accuracy is a meaningful metric.

---

## 4) EDA Highlights

We performed exploratory analysis to understand distributions, outliers, and relationships between features and the target genre:

- **Missingness / invalid values:** We found rare missing values and some invalid `duration_ms = 0` rows.
- **Feature distributions:** Many audio features are bounded (0–1) and show different spreads across genres.
- **Correlation structure:** Several features are correlated (e.g., loudness ↔ energy), which informed later modeling and interpretation.

Plots produced in the notebook include:
- Boxplots of predictors vs. genre
- Correlation heatmap of numeric features

---

## 5) Data Preprocessing & Feature Engineering

Cleaning decisions (implemented in the notebook):

1. **Drop missing values** (rows with `NaN`)
2. **Remove invalid duration rows** (`duration_ms == 0`)
3. **Drop non-numeric / identifier columns** not used for modeling:
   `track_id, artists, album_name, track_name`
4. Convert boolean columns to numeric (e.g., `explicit → 0/1`)

For the neural network experiment, we also applied:
- **Standardization** (`StandardScaler`) on input features

---

## 6) Key Methodology (Main Model)

### Random Forest for 10‑genre classification

We chose a Random Forest because it:
- Handles **nonlinear** relationships well
- Is robust to monotonic transformations and mixed feature scales
- Provides **feature importance** estimates for interpretability
- Works well as a strong baseline on tabular data

**Training setup:**
- Data: 10-genre subset of cleaned dataset (`df_10`)
- Split: 80/20 train/test (`train_test_split`, `random_state=1`)
- Model: `RandomForestClassifier(n_estimators=100, random_state=1)`

**Evaluation metrics:**
- Primary: accuracy
- Secondary: confusion matrix (visual error analysis)

---

## 7) Results

### Holdout test performance (from code notebook outputs)

| Model | Task | Split | Notes | Test Accuracy |
|---|---|---:|---|---:|
| Random Forest (all features) | 10‑genre | 80/20 | Baseline RF with full feature set | **0.743** |
| Random Forest (reduced features) | 10‑genre | 80/20 | Dropped low-importance features | 0.723 |
| Neural Network (2 hidden layers + dropout) | 10‑genre | 85/15 | Standardized inputs + early stopping | 0.657 |

### Error analysis
We used a confusion matrix() to identify which genres are most often confused. The most frequent confusions appear between genres with similar audio profiles (e.g., some overlap between pop/rock/electronic-style features), while more distinct genres (e.g., classical) tend to be easier.
<img width="928" height="855" alt="image" src="https://github.com/user-attachments/assets/2e1b5c05-5a2f-49f7-91a5-dc2737926efc" />

---

## 8) Cross-Validation

**5‑fold CV accuracy (mean ± std):** 
- We ran **5-fold Stratified cross-validation** on the training split to estimate generalization performance
`5-fold CV (train) accuracy: 0.720 ± 0.008
Fold accuracies: [0.718 0.736 0.714 0.714 0.717]`
- We then trained the final model further on the full test split and reported the hold out accuracy


---

## 9) Limitations

- **Label noise / semantics:** “Genre” is not a clean ground truth; many tracks blend genres, and labeling conventions vary.
- **Scope reduction:** Restricting to 10 genres improves tractability but reduces coverage of the full dataset.
- **Feature set limitations:** We used tabular audio features; we did not use raw audio or embeddings, which may capture richer structure.
- **Hyperparameter search depth:** We performed limited tuning; more systematic search could improve performance.

---

## 10) How to Run / Use the Code

1. Open the main notebook: `cs_m148_final_project_MD.ipynb`
2. Run cells top-to-bottom.
3. The notebook will:
   - Load the dataset
   - Clean it
   - Run EDA
   - Train/evaluate models
   - Generate plots (EDA, feature importances, confusion matrices)

**Dependencies:** `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `tensorflow`  
(If loading via `hf://`, your environment must support Hugging Face dataset fetching.)

---

# Appendix — Methods from Check-ins (Checklist)

This appendix documents the methods explored during check-ins and what we learned from them. If a method is central to the project (e.g., Random Forest), details appear above and are referenced here.

## A0) Dependencies to run 
**Core libraries**
- numpy (numerical arrays)
- pandas (data loading/cleaning)
- matplotlib (plots)
- seaborn (EDA plots)

**Machine learning**
- scikit-learn (train/test split, preprocessing, models, metrics, cross-validation)
- torch (PyTorch) / tensorflow

## A1) EDA & Train/Test Splits - Project Check-in 1
- Visualized distributions of key audio features
- Checked missingness and invalid rows
- Used stratified holdout splits for classification tasks

## A2) Regression Analysis (Linear Regression + Regularization)
We applied linear regression to explore relationships between audio features. Example: predicting **acousticness** from **energy**.

What we learned:
- The relationship is **negative** (high energy tends to correspond to low acousticness).
- This regression is useful for **understanding feature relationships**, but it is not the main approach for genre classification.

Regularization:
- We experimented with **Lasso (L1)** regularization to inspect coefficient shrinkage and guard against overfitting in linear settings.

## A3) Logistic Regression (Binary Classification)
We applied logistic regression to a binary label: **explicit (0/1)** using features like speechiness, loudness, energy, liveness, and danceability.

What we learned:
- Logistic regression provides an interpretable baseline and helps understand which features correlate with explicit content.
- Regularization (via `C` or solver choice) can matter when features are correlated.

## A4) KNN / Decision Trees / Random Forests
We used Random Forests as both:
- A strong baseline for binary classification (explicit vs not explicit), including ROC/AUC analysis
- The main method for multi-class genre classification (Section 6–7)

We did **not** rely on KNN as the final model, because it is sensitive to feature scaling and struggles in higher-dimensional spaces. A single decision tree was used as a simpler baseline; the **Random Forest** performed best and was more stable.


Why RF works well here:
- Nonlinear decision boundaries
- Strong performance on tabular feature sets
- Built-in feature importance

## A5) PCA + Clustering
We applied PCA to reduce dimensionality on standardized audio features, then used K-means clustering.

What we learned:
- PCA shows that much variance can be captured by a smaller number of components.
- K-means can reveal broad “islands” of audio similarity (e.g., acoustic/low-energy vs high-energy clusters), but clusters do not map cleanly to genre labels.

Hyperparameter example:
- We scanned **k = 1…10** (elbow/inertia) to select a reasonable number of clusters.

## A6) Neural Network Attempt
We trained a simple feed-forward network:
- 2 hidden layers (128 → 64), ReLU
- Dropout (0.3) for regularization
- Early stopping on validation loss
- Softmax output for 10-way classification

Result:
- Lower accuracy than Random Forest on this tabular dataset, suggesting RF is better matched to the feature type and dataset size/structure.

## A7) Hyperparameter Tuning Examples
We tuned or explored the following parameters during development:
- **K-means k** (cluster count) via inertia/elbow
- **PCA explained variance threshold** (e.g., 90% variance)
- **Neural net architecture** (layer sizes, dropout) and training knobs (learning rate, batch size, early stopping)
- **Random Forest** parameters (e.g., number of trees, depth) — recommended to finalize via cross-validation (Section 8)



