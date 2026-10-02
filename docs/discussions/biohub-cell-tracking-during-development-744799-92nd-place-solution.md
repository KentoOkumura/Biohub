# 92nd Place Solution

- archived_at: 2026-10-02T11:53:17.113635+00:00
- source: https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/744799
- author: tedo
- posted_at_utc: 2026-10-01 01:09:31.326000
- votes: 1
- comments: 0

## 1. TL;DR

Public 0.957 (246th) → private 0.928 (92nd / 4,017). We started from a public notebook and changed only what happens after the networks.

| # | Change | Public | Private |
| --- | --- | --- | --- |
| 0 | Public 0.947 notebook, unmodified | 0.947 | 0.916 |
| 1 | + Valley veto : cancel a division when no intensity dip appears between the daughters | 0.953 | 0.919 |
| 2 | + Centre correction : ridge regression from U-Net features to a sub-voxel offset | 0.955 | 0.927 |
| 3 | + ILP re-balance : relinker off, candidate edges p ≥ 0.3, +0.3 on every edge cost | 0.957 | 0.928 |

## 2. Starting point

`reyhanksatria/biohub-cell-tracking-0-947-lb` with pilkwang's public weights: 3D U-Net (detection) → Transformer (edge probabilities) → ILP → 9 post-processing stages. No retraining. Our own models stopped at public 0.90 / private 0.875, so we moved to this pipeline ten days before the deadline.

## 3. Approach: look at the 3D data, prefer changes with a reason

The public weights were trained on the train clips, so every local score leaks. To keep that risk down, we spent most of our time visualising the 3D data and tracing each error to its cause, and we preferred changes with a physical reason (like the valley veto) over changes that only moved the numbers.

In practice:

- We re-implemented the post-processing locally and matched it to the notebook's own logs stage by stage, so we could replay any clip on CPU.

- We built case books that show each division error in 3D: the raw intensity volume, the detections, and the GT.

- We scored all 199 clips with the official metric. The notebook validates on 8 clips, which contain 12 divisions and only one false division the metric can count. On those 8 clips the valley veto did not change the score at all, yet it raised the public LB by 0.006. On 199 clips it removed 71 of 115 countable false divisions.

- Anything learned was validated across embryos (fit on `44b6`, score on `6bba`, and the reverse).

![3D_visualization](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F5409146%2F8b3fb61de41f9465008f7d046e91cd81%2F3D_visualization.png?generation=1790816914350065&alt=media)

Division case book: raw intensity (t−2 … t+2 projections and a 3D ray-marched volume) with the parent, both children and the GT nodes.

## 4. What worked

### 4.1 Valley veto (public +0.006, private +0.003)

The public pipeline adds a second child to single-child parents with geometric gates. On 199 clips this gave division TP 24 / FP 115 / FN 127. Almost all false positives had one shape: an annotated non-dividing cell plus an unannotated detection next to it, mostly one nucleus detected twice.

Two real daughters have a dark gap between them; a nucleus detected twice does not. On the line between the two daughters we measure `r = min intensity / lower end peak` and keep a division only when the dip appears:

```text
keep  iff  r(t) ≥ 0.85  and  ( r(t+1) < 0.85  or  r(t+2) < 0.35 )

```

t+2 gets a stricter threshold: there, real divisions show a deep dip (median r = 0.11), while the false forks rejected by the t+1 rule show almost none (median 0.93). Reusing 0.85 at t+2 brought back 23 false forks; 0.35 brings back one.

![valley_veto_cases](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F5409146%2Fcdfed095cba2473eb31f1ade079ccada%2Fvalley_veto_cases.png?generation=1790816391352498&alt=media)

Top: a real division; the gap appears at t+1 (r = 0.08). Bottom: one nucleus detected twice; no gap at any frame (r ≥ 0.90), so the veto removes it.

| Rule | 199 clips (Δ score) | Division TP / FP / FN | Public | Private |
| --- | --- | --- | --- | --- |
| No veto | — | 24 / 115 / 127 | 0.947 | 0.916 |
| Dip at t+1 | +0.0020 | 21 / 44 / 130 | 0.953 | 0.919 |
| Dip at t+1 or t+2 < 0.35 | +0.0029 | 23 / 45 / 128 | 0.953 | 0.919 |

Both embryos improved. The veto has no trained parameters, only two thresholds read off the intensity profiles.

### 4.2 Centre correction with ridge regression (public +0.002, private +0.008)

Detections sit on the detector's 1.625 µm grid (x and y are downsampled 4×). The public x138 notebook corrects detection centres with a small MLP head (without it, x138 drops from 0.953 to 0.946). We built our own:

- Input: 32-channel U-Net features at the detection and its 6 neighbours (224 values).

- Target: GT centre − detection (µm), on 132,366 matched pairs from all 199 clips.

- Model: ridge regression, output capped at 1.99 µm, applied right after detection.

| Model (224 features, 169 clips) | 4-fold over clips | Across embryos |
| --- | --- | --- |
| Ridge | +0.0036 | +0.0028 |
| LightGBM | +0.0031 | +0.0014 |
| MLP | +0.0041 | +0.0019 |

The MLP won within an embryo, the ridge across embryos. The test is other embryos, so we shipped the ridge. The private LB agreed:

| Centre correction | Public | Private |
| --- | --- | --- |
| None | 0.953 | 0.919 |
| x138's MLP head | 0.956 | 0.923 |
| Our ridge | 0.955 | 0.927 |

### 4.3 ILP re-balance (public +0.002, private +0.001)

Missed annotated cells were not a detector problem: dropping the detection threshold to 0.3 recovered only 0.14 pt of nodes. Most of them were, as far as we could tell, nodes the ILP dropped because they had no candidate edge. So we:

- widened candidate edges from p ≥ 0.48 to p ≥ 0.3;

- changed the ILP edge cost from −p to −p + 0.3, so weakly linked tracks are dropped as a whole;

- turned off the relinker, which otherwise overwrites the ILP's choice.

| Configuration | Public | Private |
| --- | --- | --- |
| Public relinker | 0.955 | 0.927 |
| Relinker off | 0.956 | 0.925 |
| Relinker off + candidates 0.3 + θ 0.3 | 0.957 | 0.928 |

Final picks: relinker off + candidates 0.3 + θ 0.3 (private 0.928), and the flow-aware relinker from section 5 (0.927). They differ in about 9% of output edges.

## 5. Big locally, flat on the leaderboard: a flow-aware relinker

Replaying the post-processing stage by stage showed the relinker losing 3,107 edges net. The edges it broke belonged to cells moving fast together with their neighbours (median step 7.5 µm vs 1.8 µm overall), so we shifted its gate by the local flow.

|  | Public relinker | Flow-aware | Relinker off |
| --- | --- | --- | --- |
| 199 train clips | 0.913 | 0.931 | 0.934 |
| Public LB | 0.955 | 0.956 | 0.956 |
| Private LB | 0.927 | 0.927 | 0.925 |

A +0.02 local gain gave nothing on private. The public weights were trained on these clips, so their ILP edges are close to the answer there, and any change that relies more on them looks better locally than it is. On a new embryo the ILP makes more mistakes and the relinker repairs some of them: turning it off cost 0.002 on private.

## 6. What did not work

| Idea | Local | Public | Private |
| --- | --- | --- | --- |
| Fine-tune the public weights on 199 clips as a 3rd ensemble member | +0.0028 | 0.940 | 0.916 |
| Reject a second child within 4.5 µm of the image border | +0.0006 | 0.948 | 0.919 |
| ILP division weight 1.2 → 0.4 (with relinker off) | −0.0026 | 0.943 | 0.921 |
| Flow penalty inside the ILP cost | −0.003 to −0.005 | — | — |
| Lower detection threshold, recover in post-processing | +0.14 pt recall for +23% nodes | — | — |
| Division rejectors by counting nuclei | constants did not transfer between models | — | — |
| Our own models: D4 TTA, rot90 aug, stacked frames, (t, t+2) pairs, dense pseudo-labels | all lost to the baseline | — | — |

## 7. Takeaways

- Rules with a physical reason transferred best. The valley veto has no trained parameters and helped on both leaderboards.

- Validate learned parts across embryos. That split picked the ridge over the MLP when the public LB pointed the other way, and private was +0.004 for it.

- Small validation sets hide real effects. 8 clips showed zero for the valley veto; 199 clips showed 71 fewer false divisions.

## 8. Acknowledgements

Thanks to Biohub and Kaggle, and to reyhanksatria (base notebook), pilkwang (public weights), anvithpothula (x138 and its centre-correction head) and hengck23 (discussions on GT jumps and FOCUS-3D).
