# Scenario Reference for Keystroke Biometrics Experiments

This document provides a complete reference for all experimental scenarios. It is designed to be used by Claude Code or other AI assistants to generate scenario implementations for other experiments in the project.

## Notation

```
[Pi, Vj, Sk] where:
  i ∈ {F, I, T}     = Platform (Facebook, Instagram, Twitter)
  j ∈ {1, 2, 3}     = Video/Topic
  k ∈ {1, 2}        = Session
```

Each user has exactly 1 sample per (Platform, Video, Session) combination = 18 samples per user.

---

## Scenario 1.1: Same Platform, Same Topic (S1 → S2)

**Purpose**: Baseline performance; first half of session effect test

**Description**: Train on session 1, test on session 2. Same platform and video.

### Sub-experiments (9 total)

| # | Train | Test |
|---|-------|------|
| 1 | [PF, V1, S1] | [PF, V1, S2] |
| 2 | [PF, V2, S1] | [PF, V2, S2] |
| 3 | [PF, V3, S1] | [PF, V3, S2] |
| 4 | [PI, V1, S1] | [PI, V1, S2] |
| 5 | [PI, V2, S1] | [PI, V2, S2] |
| 6 | [PI, V3, S1] | [PI, V3, S2] |
| 7 | [PT, V1, S1] | [PT, V1, S2] |
| 8 | [PT, V2, S1] | [PT, V2, S2] |
| 9 | [PT, V3, S1] | [PT, V3, S2] |

**Samples per user**: Train=1, Test=1

---

## Scenario 1.2: Same Platform, Same Topic (S2 → S1)

**Purpose**: Baseline performance; second half of session effect test

**Description**: Train on session 2, test on session 1. Same platform and video.

**Session Analysis**: Compare 1.1 vs 1.2 to determine if session is confounding.

### Sub-experiments (9 total)

| # | Train | Test |
|---|-------|------|
| 1 | [PF, V1, S2] | [PF, V1, S1] |
| 2 | [PF, V2, S2] | [PF, V2, S1] |
| 3 | [PF, V3, S2] | [PF, V3, S1] |
| 4 | [PI, V1, S2] | [PI, V1, S1] |
| 5 | [PI, V2, S2] | [PI, V2, S1] |
| 6 | [PI, V3, S2] | [PI, V3, S1] |
| 7 | [PT, V1, S2] | [PT, V1, S1] |
| 8 | [PT, V2, S2] | [PT, V2, S1] |
| 9 | [PT, V3, S2] | [PT, V3, S1] |

**Samples per user**: Train=1, Test=1

---

## Scenario 2.1: Cross Platform, Same Topic (S1 → S2)

**Purpose**: Platform effect comparison with Scenario 1.1 (controlled for session direction)

**Description**: Train on platform X, test on platform Y, same video. Session 1 → Session 2.

### Sub-experiments (18 total)

| # | Train | Test |
|---|-------|------|
| 1 | [PF, V1, S1] | [PI, V1, S2] |
| 2 | [PF, V2, S1] | [PI, V2, S2] |
| 3 | [PF, V3, S1] | [PI, V3, S2] |
| 4 | [PF, V1, S1] | [PT, V1, S2] |
| 5 | [PF, V2, S1] | [PT, V2, S2] |
| 6 | [PF, V3, S1] | [PT, V3, S2] |
| 7 | [PI, V1, S1] | [PT, V1, S2] |
| 8 | [PI, V2, S1] | [PT, V2, S2] |
| 9 | [PI, V3, S1] | [PT, V3, S2] |
| 10 | [PI, V1, S1] | [PF, V1, S2] |
| 11 | [PI, V2, S1] | [PF, V2, S2] |
| 12 | [PI, V3, S1] | [PF, V3, S2] |
| 13 | [PT, V1, S1] | [PF, V1, S2] |
| 14 | [PT, V2, S1] | [PF, V2, S2] |
| 15 | [PT, V3, S1] | [PF, V3, S2] |
| 16 | [PT, V1, S1] | [PI, V1, S2] |
| 17 | [PT, V2, S1] | [PI, V2, S2] |
| 18 | [PT, V3, S1] | [PI, V3, S2] |

**Samples per user**: Train=1, Test=1

---

## Scenario 2.2: Cross Platform, Same Topic (S2 → S1)

**Purpose**: Platform effect comparison with Scenario 1.2 (controlled for session direction)

**Description**: Train on platform X, test on platform Y, same video. Session 2 → Session 1.

### Sub-experiments (18 total)

| # | Train | Test |
|---|-------|------|
| 1 | [PF, V1, S2] | [PI, V1, S1] |
| 2 | [PF, V2, S2] | [PI, V2, S1] |
| 3 | [PF, V3, S2] | [PI, V3, S1] |
| 4 | [PF, V1, S2] | [PT, V1, S1] |
| 5 | [PF, V2, S2] | [PT, V2, S1] |
| 6 | [PF, V3, S2] | [PT, V3, S1] |
| 7 | [PI, V1, S2] | [PT, V1, S1] |
| 8 | [PI, V2, S2] | [PT, V2, S1] |
| 9 | [PI, V3, S2] | [PT, V3, S1] |
| 10 | [PI, V1, S2] | [PF, V1, S1] |
| 11 | [PI, V2, S2] | [PF, V2, S1] |
| 12 | [PI, V3, S2] | [PF, V3, S1] |
| 13 | [PT, V1, S2] | [PF, V1, S1] |
| 14 | [PT, V2, S2] | [PF, V2, S1] |
| 15 | [PT, V3, S2] | [PF, V3, S1] |
| 16 | [PT, V1, S2] | [PI, V1, S1] |
| 17 | [PT, V2, S2] | [PI, V2, S1] |
| 18 | [PT, V3, S2] | [PI, V3, S1] |

**Samples per user**: Train=1, Test=1

---

## Scenario 3.1: Cross Platform, Same Topic (1→2, Both Sessions)

**Purpose**: Cross-platform generalization - all 6 directional pairs

**Description**: Train on 1 platform, test on 1 different platform. Same video, both sessions. All 6 platform direction pairs (F→I, F→T, I→F, I→T, T→F, T→I).

### Sub-experiments (18 total)

| # | Train | Test |
|---|-------|------|
| 1 | [PF, V1, S1+S2] | [PI, V1, S1+S2] |
| 2 | [PF, V2, S1+S2] | [PI, V2, S1+S2] |
| 3 | [PF, V3, S1+S2] | [PI, V3, S1+S2] |
| 4 | [PF, V1, S1+S2] | [PT, V1, S1+S2] |
| 5 | [PF, V2, S1+S2] | [PT, V2, S1+S2] |
| 6 | [PF, V3, S1+S2] | [PT, V3, S1+S2] |
| 7 | [PI, V1, S1+S2] | [PF, V1, S1+S2] |
| 8 | [PI, V2, S1+S2] | [PF, V2, S1+S2] |
| 9 | [PI, V3, S1+S2] | [PF, V3, S1+S2] |
| 10 | [PI, V1, S1+S2] | [PT, V1, S1+S2] |
| 11 | [PI, V2, S1+S2] | [PT, V2, S1+S2] |
| 12 | [PI, V3, S1+S2] | [PT, V3, S1+S2] |
| 13 | [PT, V1, S1+S2] | [PF, V1, S1+S2] |
| 14 | [PT, V2, S1+S2] | [PF, V2, S1+S2] |
| 15 | [PT, V3, S1+S2] | [PF, V3, S1+S2] |
| 16 | [PT, V1, S1+S2] | [PI, V1, S1+S2] |
| 17 | [PT, V2, S1+S2] | [PI, V2, S1+S2] |
| 18 | [PT, V3, S1+S2] | [PI, V3, S1+S2] |

**Samples per user**: Train=2, Test=2

---

## Scenario 3.2: Cross Platform, Same Topic (2→1, Both Sessions)

**Purpose**: Cross-platform generalization with larger dataset (dataset size effect)

**Description**: Train on 2 platforms, test on 1 remaining platform. Same video, both sessions. Leave-one-platform-out.

### Sub-experiments (9 total)

| # | Train | Test |
|---|-------|------|
| 1 | [PF+PI, V1, S1+S2] | [PT, V1, S1+S2] |
| 2 | [PF+PI, V2, S1+S2] | [PT, V2, S1+S2] |
| 3 | [PF+PI, V3, S1+S2] | [PT, V3, S1+S2] |
| 4 | [PI+PT, V1, S1+S2] | [PF, V1, S1+S2] |
| 5 | [PI+PT, V2, S1+S2] | [PF, V2, S1+S2] |
| 6 | [PI+PT, V3, S1+S2] | [PF, V3, S1+S2] |
| 7 | [PT+PF, V1, S1+S2] | [PI, V1, S1+S2] |
| 8 | [PT+PF, V2, S1+S2] | [PI, V2, S1+S2] |
| 9 | [PT+PF, V3, S1+S2] | [PI, V3, S1+S2] |

**Samples per user**: Train=4, Test=2

---

## Scenario 4.1: Cross Platform, Cross Topic (1→2, Both Sessions)

**Purpose**: Hardest generalization case - exhaustive cross-platform cross-topic

**Description**: Train on 1 platform with 1 video, test on different platform with different video. All 36 combinations (6 platform pairs x 6 video pairs).

### Sub-experiments (36 total)

Platform pairs: F→I, F→T, I→F, I→T, T→F, T→I (6 pairs)
Video pairs where train ≠ test: V1→V2, V1→V3, V2→V1, V2→V3, V3→V1, V3→V2 (6 pairs)

| # | Train | Test |
|---|-------|------|
| 1-6 | [PF, V*, S1+S2] | [PI, V*, S1+S2] | (all 6 video pairs)
| 7-12 | [PF, V*, S1+S2] | [PT, V*, S1+S2] | (all 6 video pairs)
| 13-18 | [PI, V*, S1+S2] | [PF, V*, S1+S2] | (all 6 video pairs)
| 19-24 | [PI, V*, S1+S2] | [PT, V*, S1+S2] | (all 6 video pairs)
| 25-30 | [PT, V*, S1+S2] | [PF, V*, S1+S2] | (all 6 video pairs)
| 31-36 | [PT, V*, S1+S2] | [PI, V*, S1+S2] | (all 6 video pairs)

**Samples per user**: Train=2, Test=2

---

## Scenario 4.2: Cross Platform, Cross Topic (2→1, Both Sessions)

**Purpose**: Dataset size effect in cross-platform cross-topic case

**Description**: Train on 2 platforms with 1 video, test on remaining platform with different video.

### Sub-experiments (18 total)

| # | Train | Test |
|---|-------|------|
| 1-6 | [PF+PI, V*, S1+S2] | [PT, V*, S1+S2] | (6 video pairs where train_v ≠ test_v)
| 7-12 | [PI+PT, V*, S1+S2] | [PF, V*, S1+S2] | (6 video pairs where train_v ≠ test_v)
| 13-18 | [PT+PF, V*, S1+S2] | [PI, V*, S1+S2] | (6 video pairs where train_v ≠ test_v)

**Samples per user**: Train=4, Test=2

---

## Scenario 5.1: Same Platform, Cross Topic (S1 → S2)

**Purpose**: Topic effect on same platform - train session 1, test session 2

**Description**: Train on 1 video, test on different video. Same platform, session 1 → session 2.

### Sub-experiments (18 total)

3 platforms x 6 video pairs = 18 sub-experiments

| # | Train | Test |
|---|-------|------|
| 1 | [PF, V1, S1] | [PF, V2, S2] |
| 2 | [PF, V1, S1] | [PF, V3, S2] |
| 3 | [PF, V2, S1] | [PF, V1, S2] |
| 4 | [PF, V2, S1] | [PF, V3, S2] |
| 5 | [PF, V3, S1] | [PF, V1, S2] |
| 6 | [PF, V3, S1] | [PF, V2, S2] |
| 7 | [PI, V1, S1] | [PI, V2, S2] |
| 8 | [PI, V1, S1] | [PI, V3, S2] |
| 9 | [PI, V2, S1] | [PI, V1, S2] |
| 10 | [PI, V2, S1] | [PI, V3, S2] |
| 11 | [PI, V3, S1] | [PI, V1, S2] |
| 12 | [PI, V3, S1] | [PI, V2, S2] |
| 13 | [PT, V1, S1] | [PT, V2, S2] |
| 14 | [PT, V1, S1] | [PT, V3, S2] |
| 15 | [PT, V2, S1] | [PT, V1, S2] |
| 16 | [PT, V2, S1] | [PT, V3, S2] |
| 17 | [PT, V3, S1] | [PT, V1, S2] |
| 18 | [PT, V3, S1] | [PT, V2, S2] |

**Samples per user**: Train=1, Test=1

---

## Scenario 5.2: Same Platform, Cross Topic (S2 → S1)

**Purpose**: Topic effect on same platform - train session 2, test session 1

**Description**: Train on 1 video, test on different video. Same platform, session 2 → session 1.

### Sub-experiments (18 total)

3 platforms x 6 video pairs = 18 sub-experiments

| # | Train | Test |
|---|-------|------|
| 1 | [PF, V1, S2] | [PF, V2, S1] |
| 2 | [PF, V1, S2] | [PF, V3, S1] |
| 3 | [PF, V2, S2] | [PF, V1, S1] |
| 4 | [PF, V2, S2] | [PF, V3, S1] |
| 5 | [PF, V3, S2] | [PF, V1, S1] |
| 6 | [PF, V3, S2] | [PF, V2, S1] |
| 7 | [PI, V1, S2] | [PI, V2, S1] |
| 8 | [PI, V1, S2] | [PI, V3, S1] |
| 9 | [PI, V2, S2] | [PI, V1, S1] |
| 10 | [PI, V2, S2] | [PI, V3, S1] |
| 11 | [PI, V3, S2] | [PI, V1, S1] |
| 12 | [PI, V3, S2] | [PI, V2, S1] |
| 13 | [PT, V1, S2] | [PT, V2, S1] |
| 14 | [PT, V1, S2] | [PT, V3, S1] |
| 15 | [PT, V2, S2] | [PT, V1, S1] |
| 16 | [PT, V2, S2] | [PT, V3, S1] |
| 17 | [PT, V3, S2] | [PT, V1, S1] |
| 18 | [PT, V3, S2] | [PT, V2, S1] |

**Samples per user**: Train=1, Test=1

---

## Summary Table

| Scenario | Platform | Topic | Session | Train/User | Test/User | Sub-exp |
|----------|----------|-------|---------|------------|-----------|---------|
| 1.1 | Same | Same | S1 → S2 | 1 | 1 | 9 |
| 1.2 | Same | Same | S2 → S1 | 1 | 1 | 9 |
| 2.1 | Cross (all 6) | Same | S1 → S2 | 1 | 1 | 18 |
| 2.2 | Cross (all 6) | Same | S2 → S1 | 1 | 1 | 18 |
| 3.1 | Cross (1→2, all 6) | Same | Both | 2 | 2 | 18 |
| 3.2 | Cross (2→1) | Same | Both | 4 | 2 | 9 |
| 4.1 | Cross (1→2, all 6) | Cross (all 6) | Both | 2 | 2 | 36 |
| 4.2 | Cross (2→1) | Cross (all 6) | Both | 4 | 2 | 18 |
| 5.1 | Same | Cross (all 6) | S1 → S2 | 1 | 1 | 18 |
| 5.2 | Same | Cross (all 6) | S2 → S1 | 1 | 1 | 18 |

**Total sub-experiments**: 171

---

## Valid Comparisons for Effect Isolation

| Effect | Comparison | Confounds |
|--------|------------|-----------|
| **Session** | Scenario 1.1 vs 1.2 | None (internal analysis) |
| **Platform (S1→S2)** | 1.1 vs 2.1 | None |
| **Platform (S2→S1)** | 1.2 vs 2.2 | None |
| **Topic (same platform, S1→S2)** | 1.1 vs 5.1 | None |
| **Topic (same platform, S2→S1)** | 1.2 vs 5.2 | None |
| **Topic (cross-plat, 1→2)** | 3.1 vs 4.1 | None |
| **Topic (cross-plat, 2→1)** | 3.2 vs 4.2 | None |
| **Size (same-topic)** | 3.1 vs 3.2 | None |
| **Size (cross-topic)** | 4.1 vs 4.2 | None |

---

## Implementation Reference

```python
from scenarios import (
    generate_scenario_1_1,
    generate_scenario_1_2,
    generate_scenario_2_1,
    generate_scenario_2_2,
    generate_scenario_3_1,
    generate_scenario_3_2,
    generate_scenario_4_1,
    generate_scenario_4_2,
    generate_scenario_5_1,
    generate_scenario_5_2,
    generate_all_scenarios,
    get_scenario_by_id,
)

# Get a specific scenario
scenario = get_scenario_by_id("2.1")

# Iterate over sub-experiments
for sub_exp in scenario.sub_experiments:
    train_filter = sub_exp.train_filter  # {"platform_id": [1], "video_id": [1], "session_id": [1]}
    test_filter = sub_exp.test_filter    # {"platform_id": [2], "video_id": [1], "session_id": [2]}

    # Apply filters to your dataframe
    train_data = df[
        (df['platform_id'].isin(train_filter['platform_id'])) &
        (df['video_id'].isin(train_filter['video_id'])) &
        (df['session_id'].isin(train_filter['session_id']))
    ]
```

---

## Running Experiments & Generating Results TSV

### Quick Start

```bash
# Setup environment
make quickstart
source .venv/bin/activate

# Run all scenarios (171 sub-experiments)
python ml_scenario_runner.py -c config_scenarios.json

# Run specific scenarios
python ml_scenario_runner.py --scenarios 1.1 1.2 2.1 5.1 5.2

# Run with specific models
python ml_scenario_runner.py --models CatBoost RandomForest

# Disable GPU (if needed for other tasks)
python ml_scenario_runner.py --no-gpu
```

### Output Files

After running experiments, the following files are generated in the output directory:

| File | Description |
|------|-------------|
| `sub_experiment_results_*.csv` | Detailed results for each sub-experiment |
| `scenario_results_*.csv` | Aggregated results per scenario/model/seed |
| `ml_baseline_results_*.tsv` | **Template-formatted TSV for results spreadsheet** |
| `scenario_comparison_*.png` | Visualization of scenario performance |
| `model_comparison_scenario_*.png` | Model comparison per scenario |

### TSV Output Format

The `ml_baseline_results_*.tsv` file is formatted to match `scenario-template-14Dec2025.tsv`:

```
Scenario Group    Scenario       Train                              Train samples/user    Test                               Test samples/user    Notes    k=1      k=2      k=3      k=4      k=5      Best Model
Same platform...  Scenario 1.1   [Pi, Vj, Sk] i={F}, j={1}, k={1}   1                     [Pi, Vj, Sk] i={F}, j={1}, k={2}   1                    ...      0.2451   0.3562   0.4123   0.4567   0.5012   CatBoost
                                 [Pi, Vj, Sk] i={F}, j={2}, k={1}   1                     [Pi, Vj, Sk] i={F}, j={2}, k={2}   1                    ...      0.2234   0.3345   ...
...
                                 mean                               ...                                                                                    0.2342   0.3453   0.4012   0.4456   0.4901
                                 std                                ...                                                                                    0.0234   0.0345   0.0401   0.0445   0.0490
```

**Key features:**
- Uses the **best performing model** per scenario (by top-1 accuracy)
- Reports top-k accuracy for k=1,2,3,4,5
- Includes mean and std rows per scenario
- Tab-separated for easy copy-paste into Excel/Google Sheets

### Generating TSV for Other Experiments

To generate TSV results for a different experiment type (e.g., LLM embeddings, fusion models):

1. **Implement the experiment** following the same scenario structure
2. **Collect results** with the same metrics (top-1 through top-5 accuracy)
3. **Use the TSV generation pattern** from `ml_scenario_runner.py`:

```python
from scenarios import generate_all_scenarios

def generate_template_tsv(results_df, output_path, experiment_name="My Experiment"):
    """
    Generate TSV matching the results template.

    Args:
        results_df: DataFrame with columns:
            - scenario_id, sub_experiment_name
            - test_top_1_accuracy, test_top_2_accuracy, ..., test_top_5_accuracy
        output_path: Path for output TSV
        experiment_name: Name for the "Best Model" column
    """
    all_scenarios = generate_all_scenarios()

    tsv_rows = [[
        "Scenario Group", "Scenario", "Train", "Train samples/user",
        "Test", "Test samples/user", "Notes",
        "k=1", "k=2", "k=3", "k=4", "k=5", "Model/Method"
    ]]

    for scenario_id in sorted(results_df["scenario_id"].unique()):
        scenario = all_scenarios.get(scenario_id)

        for sub_exp in scenario.sub_experiments:
            sub_data = results_df[results_df["sub_experiment_name"] == sub_exp.name]
            if sub_data.empty:
                continue

            row = [
                scenario.description if sub_exp == scenario.sub_experiments[0] else "",
                f"Scenario {scenario_id}" if sub_exp == scenario.sub_experiments[0] else "",
                sub_exp.train_notation,
                scenario.train_samples_per_user,
                sub_exp.test_notation,
                scenario.test_samples_per_user,
                "",
                f"{sub_data['test_top_1_accuracy'].mean():.4f}",
                f"{sub_data['test_top_2_accuracy'].mean():.4f}",
                f"{sub_data['test_top_3_accuracy'].mean():.4f}",
                f"{sub_data['test_top_4_accuracy'].mean():.4f}",
                f"{sub_data['test_top_5_accuracy'].mean():.4f}",
                experiment_name if sub_exp == scenario.sub_experiments[0] else ""
            ]
            tsv_rows.append(row)

        # Add mean/std rows...

    with open(output_path, "w") as f:
        for row in tsv_rows:
            f.write("\t".join(str(x) for x in row) + "\n")
```

---

## File References

- Scenario implementation: `scenarios.py`
- ML experiment runner: `ml_scenario_runner.py`
- Configuration: `config_scenarios.json`
- Results template: `scenario-template-14Dec2025.tsv`
- This reference: `SCENARIO_REFERENCE.md`
