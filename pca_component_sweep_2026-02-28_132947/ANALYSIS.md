# PCA Component Sweep: Analysis

## Background — What We Already Knew

In the earlier PCA vs RF comparison experiment, we asked: "Can PCA tell us which features matter?" We used the standard PCA approach — keep enough components to explain 95% of variance (that turned out to be ~17–23 components depending on the sub-experiment), look at which original features load heavily on those components, and keep those features. That gave us **384 features** out of 620 total.

We then trained Random Forest on just those 384 features and compared to the baseline (all 620 features). The results were split:

- **Scenarios 3.2 and 4.2** (data-rich, 4 training samples per user): PCA did great. 98.3% and 97.2% accuracy vs 99.7% and 98.7% baseline. Barely any loss.
- **Scenarios 1.x, 2.x, 5.x** (data-poor, 1 sample per user): PCA was terrible. ~60–65% accuracy vs ~80–83% baseline. Massive gap.
- **Scenarios 3.1 and 4.1** (middle ground, 2 samples per user): PCA was decent. ~86–89% vs ~96–97% baseline. Noticeable gap but not catastrophic.

That raised a natural question: **why does PCA feature selection work so much better when there's more training data?** Two possible explanations:

**(a)** Maybe PCA just needs to select **more features** to work well with limited data, and the 95% variance threshold was too aggressive (too few components, too few features selected).

**(b)** Maybe PCA needs **more training data** to accurately estimate which features matter in the first place, and with only 61 training samples it's just making bad choices no matter what.

## What This Sweep Experiment Does

The sweep disentangles (a) and (b) by doing two things:

1. Instead of using a fixed threshold, it tries **every possible number of PCA components** from 1 all the way up to the maximum, and measures accuracy at each one. If explanation (a) is right, we'd see the low-data scenarios catch up to baseline if we just use more components.

2. For scenarios 3.2 and 4.2 (which naturally have 4 samples/user), it **artificially reduces** the training data to 2/user and 1/user, then re-runs the full sweep. This holds everything else constant — same scenario, same features, same test data — and isolates the effect of training data volume alone.

## What The Sweep Found

### Answer to the central question: it's mostly about the training data, not the number of components.

**Plot A** (Accuracy vs K) tells this story clearly across the three panels:

- **Right panel (4 samples/user):** The curve rockets up and flattens almost immediately. By K=5–7 components, you're already at 95% of baseline. Adding more components barely helps. PCA "gets it right" with very few components because it has enough data to work with.

- **Left panel (1 sample/user):** The curve is a slow, painful climb. It's still rising at K=40–50 and never fully reaches the baseline (the dashed line). Even if you use ALL components and ALL the features PCA selects, you still fall ~1–2% short. More components help some, but they can't fully compensate for having too little data.

- **Middle panel (2 samples/user):** In between. It catches up to baseline around K=25–30.

### The subsampling experiment is the smoking gun

**Plot D** is the most important result. It takes scenarios 3.2 and 4.2 — which naturally perform great with PCA — and artificially starves them of training data. Look at what happens:

- **Green line (4/user):** Shoots up and flattens by K=10. Near-perfect accuracy.
- **Orange dashed line (2/user):** Same scenario, same features, but half the training data. Reaches ~95% and needs K=25–30 to get there.
- **Blue dotted line (1/user):** Same scenario, a quarter of the training data. Maxes out at ~80% no matter how many components you use.

This is the same scenario, the same feature space, the same test set. The only difference is how much training data PCA and the classifier get to work with. That 1/user blue line looks just like the 1-spu scenarios in the main experiment — confirming that the poor performance of PCA in scenarios 1.x/2.x/5.x was not because of something inherent to those scenarios, but because they simply don't have enough training data.

### Connecting back to the 384-feature result

The earlier experiment used K=17–23 components (the 95% variance threshold), which selected 384 features. Where does K=20 sit on the sweep curves?

- For 4 spu scenarios: K=20 is well past the plateau. PCA had already "figured it out" by K=5. That's why 384 features worked so well for 3.2 and 4.2.
- For 1 spu scenarios: K=20 is only partway up the climb, recovering about 83% of baseline accuracy. That's consistent with the ~65% accuracy we saw in the earlier comparison (which was even lower because the earlier experiment intersected PCA features across all scenarios, further constraining the set).

## The Bottom Line

PCA feature selection doesn't fail on low-data scenarios because it picks the "wrong" features. It fails because **with only ~61 training samples, PCA can't reliably estimate the structure of a 620-dimensional space**. The component loadings are noisy, so the features it selects are partly right and partly noise. More components can partially compensate (pulling accuracy from 60% up to 80%), but you can't fully overcome the fundamental data limitation. When you have 244 training samples (4/user), PCA nails the structure with just 5–7 components.

## Convergence Reference

How many PCA components are needed to reach 90% and 95% of baseline accuracy, by scenario:

| Scenario | Samples/User | K for 90% of Baseline | K for 95% of Baseline |
|----------|-------------|----------------------|----------------------|
| 1.1      | 1           | 27                   | 40                   |
| 1.2      | 1           | 27                   | 45                   |
| 2.1      | 1           | 28                   | 40                   |
| 2.2      | 1           | 28                   | 45                   |
| 3.1      | 2           | 9                    | 24                   |
| 3.2      | 4           | 5                    | 6                    |
| 4.1      | 2           | 18                   | 29                   |
| 4.2      | 4           | 5                    | 7                    |
| 5.1      | 1           | 28                   | 40                   |
| 5.2      | 1           | 28                   | 45                   |
