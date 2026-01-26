# Behavioral Biometrics Classification Methods

## Research Context

**Goal:** Understanding whether accuracy differences across scenarios between behavioral and content-based models are statistically significant for user identification in social media fake profile detection.

**Modality Focus:** This document covers the **behavioral modality** — keystroke dynamics classifiers that analyze *how* users type rather than *what* they type.

---

## 1. ITAD Classifier

**Reference:** Ayotte et al., "Fast Free-text Authentication via Instance-based Keystroke Dynamics" (2020)

### Pipeline

```
Raw Keystroke Timing → Feature Dictionaries → Empirical CDF → Tail Area Density Scoring
```

### Method Overview

**ITAD (Instance-based Tail Area Density)** is a similarity metric that measures how well a test keystroke sample fits within a user's enrolled profile by computing the tail area under the probability density function.

### Why ITAD?

| Design Choice | Rationale |
|---------------|-----------|
| **Instance-based** | Can compare graphs even with a single test instance — unlike distribution-based methods that require 4+ instances per feature |
| **Tail area density** | Uses percentile position rather than raw distance — more robust to outliers than Scaled Manhattan or other distance metrics |
| **Median-relative scoring** | Measures how "typical" a sample is relative to the profile's central tendency |
| **No training required** | Direct template matching — suitable for enrollment with limited samples |

### How It Works

1. **Build reference profile:** For each user, store the distribution of timing values for each keystroke feature (e.g., all hold times for key 'a', all digraph times for 'th')

2. **Compute tail area density:** For each test sample value, calculate its position in the reference distribution:

   | Condition | Formula | Interpretation |
   |-----------|---------|----------------|
   | Test value ≤ Median | S = CDF(x) | Left tail area |
   | Test value > Median | S = 1 - CDF(x) | Right tail area |

3. **Aggregate:** Average all individual similarity scores across shared features

### Scoring Interpretation

| Score Range | Meaning |
|-------------|---------|
| **Close to 0.5** | Test sample is near the median — high similarity to profile |
| **Close to 0** | Test sample is in the extreme tails — low similarity to profile |

**Final score range: 0 to 0.5** (higher = more similar)

### Why ITAD Over Distance Metrics?

| Metric | Basis | Weakness |
|--------|-------|----------|
| Scaled Manhattan | Mean + Std deviation | Sensitive to outliers (typos, pauses) |
| **ITAD** | Tail area under PDF | Robust to outliers; uses full distribution shape |

### Strengths & Limitations

| Strengths | Limitations |
|-----------|-------------|
| Works with single instances per feature | Pairwise comparison — doesn't scale to large populations |
| Robust to outliers | Requires overlapping features between test and profile |
| No hyperparameter tuning | Score interpretation less intuitive than direct distance |
| Outperforms distance metrics on small datasets | |

---

## 2. Random Forest Classifier

### Pipeline

```
Raw Keystroke Timing → Statistical Feature Extraction → Ensemble Tree Voting → Majority Classification
```

### Method Overview

**Random Forest** is an ensemble method that builds multiple decision trees on random subsets of features and samples, then aggregates their votes for classification.

### Why Random Forest?

| Design Choice | Rationale |
|---------------|-----------|
| **Ensemble approach** | Reduces overfitting risk — critical when keystroke data has high individual variability |
| **Feature importance** | Reveals which timing features are most discriminative for user identification |
| **Handles mixed feature types** | Works well with the diverse statistical summaries (means, medians, quartiles) |
| **No feature scaling required** | Tree-based methods are invariant to monotonic transformations |
| **Robust to noise** | Bootstrap sampling and feature randomization reduce sensitivity to outliers |

### Data Transformation Rationale

**Why aggregate raw timings into statistics?**

| Raw Data | Transformed Data | Why Transform? |
|----------|------------------|----------------|
| `[120, 115, 125, 118, 122]` ms | mean=120, std=3.5, q1=117, q3=123 | Fixed-length feature vector required for ML models |
| Variable-length lists | 5 values per feature | Captures central tendency AND variability |
| Sensitive to sample size | Normalized representation | Enables comparison across sessions with different lengths |

### Hyperparameter Choices

| Parameter | Values Explored | Why? |
|-----------|-----------------|------|
| Trees (100-500) | More trees = more stable predictions, diminishing returns beyond 500 |
| Max Depth (10, 20, None) | Controls model complexity; unlimited depth risks overfitting |
| Min Samples Split (2, 10) | Prevents splits on very small groups; higher values = more generalization |

---

## 3. CatBoost Classifier

### Pipeline

```
Raw Keystroke Timing → Statistical Feature Extraction → Gradient Boosting → Softmax Classification
```

### Method Overview

**CatBoost** is a gradient boosting algorithm that builds decision trees sequentially, with each new tree correcting the prediction errors of the ensemble so far.

### Why CatBoost?

| Design Choice | Rationale |
|---------------|-----------|
| **Ordered boosting** | Reduces target leakage — prevents the model from "cheating" by using future information |
| **Built-in regularization** | L2 regularization on leaf values prevents overfitting on small datasets |
| **Early stopping** | Automatically stops training when validation performance plateaus |
| **State-of-the-art accuracy** | Gradient boosting consistently wins tabular data competitions |
| **Handles class imbalance** | Important when some users have more samples than others |

### Why Boosting Over Bagging (Random Forest)?

| Aspect | Random Forest (Bagging) | CatBoost (Boosting) |
|--------|-------------------------|---------------------|
| Tree construction | Independent, parallel | Sequential, corrective |
| Error handling | Averages out errors | Explicitly targets errors |
| Bias-variance tradeoff | Reduces variance | Reduces bias |
| Best for | High-variance models | High-bias models |

**For keystroke dynamics:** We use both because they complement each other — Random Forest provides stability while CatBoost often achieves higher peak accuracy.

### Early Stopping Rationale

- **Why 50 rounds patience?** Allows the model to escape local optima while preventing excessive overfitting
- **Why use it?** Keystroke datasets are relatively small; early stopping acts as implicit regularization

---

## 4. Score Level Fusion (Random Forest Weighted Average)

**Source File:** `keystroke-scripts/score_level_fusion.py`

### Pipeline

```
Individual Model Scores → RF Weight Learning → Weighted Score Averaging → Fused Prediction
```

### Method Overview

**Score Level Fusion** combines predictions from multiple classifiers (RandomForest, CatBoost, NaiveBayes, ExtraTrees) by learning optimal weights for each model, then computing a weighted average of their outputs.

### Why Score Level Fusion?

| Design Choice | Rationale |
|---------------|-----------|
| **Late fusion** | Combines model outputs rather than raw features — preserves each model's learned representations |
| **Learned weights** | Uses RF feature importance to determine which models contribute most, rather than equal weighting |
| **Model diversity** | Combines bagging (RF, ExtraTrees), boosting (CatBoost), and probabilistic (NaiveBayes) approaches |
| **Best-of-ensemble** | Target is the max across models — learns to approximate the best achievable performance |

### How It Works

1. **Collect scores:** Gather top-k accuracy values (k=1 through k=5) from each model for every scenario

2. **Train weight learner:**
   - Features: accuracy values from each model
   - Target: maximum accuracy across all models (best achievable)
   - Model: Random Forest Regressor

3. **Extract weights:** Use RF feature importances as model weights

4. **Apply fusion:** Compute weighted average of model scores, normalized so weights sum to 1

### Weight Learning Rationale

| Approach | How Weights Are Determined |
|----------|---------------------------|
| Equal weighting | All models contribute equally (1/N) |
| **RF-learned weighting** | Models that better predict the "best achievable" get higher weight |

The RF regressor learns which models are most predictive of high performance. Models that consistently perform well when the ensemble performs well receive higher feature importance scores.

### Why Not Feature-Level Fusion?

| Fusion Level | What's Combined | Trade-off |
|--------------|-----------------|-----------|
| **Feature-level (early)** | Raw keystroke features | Requires retraining; loses model-specific learning |
| **Score-level (late)** | Model predictions/probabilities | Preserves individual model strengths; simpler to implement |

Score-level fusion is chosen because it allows each model to learn its own representation of the keystroke data, then combines their "opinions" at decision time.

---

## 5. Comparison Summary

### Pipeline Comparison

| Classifier | Pipeline |
|------------|----------|
| **ITAD** | Raw Timing → Feature Dictionaries → Empirical CDF → Tail Area Density Score |
| **Random Forest** | Raw Timing → Statistical Feature Extraction → Ensemble Tree Voting → Majority Classification |
| **CatBoost** | Raw Timing → Statistical Feature Extraction → Gradient Boosting → Softmax Classification |
| **Score Fusion** | Individual Model Scores → RF Weight Learning → Weighted Score Averaging → Fused Prediction |

### Method Comparison

| Aspect | ITAD | Random Forest | CatBoost | Score Fusion |
|--------|------|---------------|----------|--------------|
| **Approach** | Instance-based tail area density | Ensemble bagging | Gradient boosting | Late fusion ensemble |
| **Training** | None (template matching) | Learns decision boundaries | Learns error correction | Learns model weights |
| **Feature Input** | Raw timing distributions | Aggregated statistics | Aggregated statistics | Model accuracy scores |
| **Output** | Similarity score (0–0.5) | Class probabilities | Class probabilities | Weighted average score |
| **Strengths** | Outlier-robust, single instances | Robust, feature importance | High accuracy, regularized | Combines diverse models |
| **Best For** | Verification (1:1) | Identification (1:N) | Identification (1:N) | Maximizing ensemble accuracy |

### Why These Four Methods?

1. **ITAD** — Represents the instance-based statistical paradigm; robust to outliers and works with limited enrollment data; proven to outperform distance metrics (like Scaled Manhattan) on small datasets

2. **Random Forest** — Represents ensemble learning; robust baseline that's hard to overfit; useful for understanding feature importance

3. **CatBoost** — Represents state-of-the-art gradient boosting; maximizes accuracy for identification tasks; handles the class imbalance common in multi-user scenarios

4. **Score Level Fusion** — Represents ensemble combination; leverages strengths of diverse classifiers (bagging, boosting, probabilistic) to maximize overall accuracy

### Feature Extraction Rationale

**Keystroke Timing Features Used:**

| Feature | What It Captures | Why Include It? |
|---------|------------------|-----------------|
| **KHT (Key Hold Time)** | How long each key is pressed | Reflects motor control and finger dexterity |
| **KIT Type 1 (IL)** | Time between key release and next press | Captures typing rhythm and flow |
| **KIT Type 2 (RL)** | Release-to-release timing | Reflects overall keystroke cadence |
| **KIT Type 3 (PL)** | Press-to-press timing | Standard inter-key interval |
| **KIT Type 4** | Press-to-release across keys | Captures overlap in key presses |

**Statistical Aggregation:**

| Statistic | What It Captures |
|-----------|------------------|
| **Mean** | Central tendency of timing |
| **Median** | Robust central tendency (ignores outliers) |
| **Q1, Q3** | Distribution spread and skewness |
| **Std** | Consistency/variability of typing |

---

## 5. Connection to Research Goals

### Behavioral vs. Content-Based

| Modality | What's Analyzed | Example Methods |
|----------|-----------------|-----------------|
| **Behavioral** (this doc) | *How* users type | ITAD, Random Forest, CatBoost on keystroke timing |
| **Content-Based** | *What* users write | NLP models on text content |
| **Fusion** | Both combined | Score-level or feature-level combination |

### Cross-Platform/Cross-Scenario Analysis

The classifiers are evaluated across scenarios to answer:

1. **Same platform, different session** — Does typing behavior persist over time?
2. **Different platforms, same content** — Is behavior platform-independent?
3. **Different platforms, different content** — Most challenging; tests true behavioral consistency

### Statistical Significance

By running multiple random seeds and computing mean/std of metrics, we can determine whether observed accuracy differences are:
- Due to genuine algorithmic/modality differences, or
- Simply due to random variation in train/test splits
