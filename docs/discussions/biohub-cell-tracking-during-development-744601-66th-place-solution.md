# 66th Place Solution

- archived_at: 2026-10-02T11:53:15.916333+00:00
- source: https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/744601
- author: mige551
- posted_at_utc: 2026-09-30 13:32:27.089000
- votes: 3
- comments: 0

Thank you to the organizers and to everyone who shared notebooks and discussions. This was a solo entry. Our solution does not train its own detector or linker: it takes the strongest public tracking pipeline as a fixed base and adds one stage, an image-based division classifier with a fork-editing policy, which recovers the cell divisions the linker drops. The stage is the only original component, and it is the part that transferred to the private test set.

## 1. Overview

The final pipeline has six stages. Stages 1–2 are public work used unchanged; stages 3–6 are ours.

- Detection and linking (public 0.953 notebook): two temporal 3D U-Net detectors and a centre-prior U-Net, Transformer edge features with test-time augmentation, harmonic fusion of the edge logits, ILP linking.

- Edge-side post-processing (same notebook): flow-based motion re-link, re-admission of discarded detector peaks, sub-threshold gap filling, a coordinate-refinement head, a geometric safe-division rule.

- Candidate selection: parents of existing forks and out-degree-1 nodes with an orphan track start within 13 µm in the next frame.

- Division classification (DivCNN): a parent-centred 3D CNN over six consecutive frames scores each candidate.

- Fork editing: veto weak forks, add a second daughter where the classifier fires and an orphan is nearby.

- Robust production: one worker per GPU, a session-time deadline, per-dataset fallback, an independent audit.

Nodes are never added, moved or removed by stages 3–6, so the stage composes with any base tracker. Our final submission scored 0.969 on the public leaderboard and 0.932 on the private leaderboard (66th of 4,020). On the public 0.947 pipeline that we used for most of the competition, the stage was worth +0.012 public and +0.015 private (0.947 → 0.959, 0.916 → 0.931). On the 179 annotated training videos the same stage takes the base from 0.9094 (division Jaccard 0.096) to 0.9278 (0.277); that corpus is out-of-fold for the classifier but in-sample for the base.

![Figure 1 — the final pipeline: a public edge-tracking base and the DivCNN division stage](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F18476336%2F4593fb6b5dc7ea702beba423e62f0f42%2Fwriteup_fig1_pipeline.png?generation=1790777639934228&alt=media)

## 2. Data and Validation

The training set contains 199 videos from two embryos (`44b6`: 71 videos, `6bba`: 128), each 100 frames of 64 × 256 × 256 voxels at (1.625, 0.406, 0.406) µm. All distances below are in physical units. Annotations are sparse: 151 annotated divisions in the whole training set, 141 of them in the 179 videos that carry scored annotations.

The metric is the node-count-adjusted edge Jaccard plus 0.1 × division Jaccard. Because the ground truth is sparse, a predicted edge is scored only when it agrees with or contradicts an annotated edge; in one of our graphs 6,633 of 199,247 edges (3.3 %) were scored at all. Bulk levers on unannotated edges are therefore nearly inert, while every annotated division is scored, and at the base's operating point one recovered division moves the score as much as about 50 recovered ordinary edges.

Validation had two layers. The classifier was trained in five folds grouped by video (stratified by embryo and number of positives), giving an out-of-fold score for every labelled node; policies were then swept on all 179 annotated videos with the official scoring implementation. The base's own components, however, are public checkpoints trained on all 199 videos, so the corpus is in-sample for them. In practice the corpus predicted the direction of the division term but not the effect of anything that edits the base's own edges (Section 9); the operating point of the stage was chosen on the public leaderboard with at most five probes per day. Test embryos are disjoint from training ones; following the 3rd-place write-up, the public and private test embryos (60 and 106 videos) have lower cell density than the training embryos, which is consistent with the timing of our hidden reruns and with the drop every team saw between the two leaderboards.

## 3. Base Tracker

Most of the competition we used a line-by-line reproduction of the public 0.947 notebook (forks by `sjlee101` of the "Harmonic Fusion" pipeline built on `pilkwang`'s detector datasets and support pack), with its train-based post-processing sweep kept and the hidden test directory enumerated at runtime. In the last week we switched to `anvithpothula/biohub-0-953-lb-original`, which adds the edge-side stages listed in Section 1 and a CC0 coordinate-refinement head. We changed nothing in either notebook; our stage reads their `submission.csv` and the raw test images. The division stage was designed against the 0.947 base, so the switch is also the cleanest test of whether the stage composes with a different base: it gave 0.965–0.969 public against 0.953 for the base alone.

## 4. Division Census

Scoring the base's graphs on the 179 annotated videos gave division TP / FP / FN = 24 / 110 / 117. All 3,895 forks it emitted came from the geometric safe-division rule, and of the 134 that could be evaluated only 24 were right. For each of the 141 annotated divisions we then asked what the graph had done with it:

| Outcome in the base graph | Divisions |
| --- | --- |
| parent not detected | 3 |
| one daughter not detected | 37 |
| parent and both daughters detected, already forked | 32 |
| parent and both daughters detected, second daughter an orphan track start | 20 |
| parent and both daughters detected, second daughter linked to another node | 49 |

Detection was not the bottleneck: 69 of the 101 fully detected divisions were lost at the fork. A division step is two to four times an ordinary step and the daughters land 8–13 µm apart, so a distance-driven linker prefers the nearer continuation, and the geometric rule cannot tell a real split from two neighbouring cells. Only the image can.

## 5. Division Classifier (DivCNN)

![Figure 2 — an annotated division in the exact 33 µm window DivCNN v7 sees, frames t−2 … t+3 (white: parent, cyan: daughters)](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F18476336%2Fde25eab0f557efa8153feb3c26024658%2Fwriteup_fig2_division.png?generation=1790777705037220&alt=media)

### Input

For a candidate node at time t, six frames (t−2 … t+3) of an 11 × 41 × 41 crop centred on the node: z at native 1.625 µm spacing, xy mean-pooled 2× to 0.81 µm, i.e. 18 × 33 × 33 µm. The first version used a 16 µm native-resolution window and could not see the second daughter; widening the window raised out-of-fold average precision from 0.19 to 0.29. Intensities are scaled by each video's 0.1st and 99.9th percentiles and clipped at 2.5. Crops are stored with a margin (13 × 47 × 47) for shift augmentation.

### Network and training

Four stages of 2 × [Conv3d 3³ + BatchNorm + ReLU] at 32 / 64 / 128 / 192 channels with anisotropic max-pooling ((1,2,2), (2,2,2), (1,2,2)), global mean ⊕ max pooling, and a 384 → 128 → 1 head with dropout 0.3; 2.6 M parameters. BCE with class-balanced batches (20 % positives), AdamW, one-cycle schedule, 3,000 steps per fold, augmentation by random shift, D4 in xy, z-flip and intensity jitter. Five video-grouped folds; a five-fold run takes about 2.5 h on an Apple M4 Max.

### Labels and self-training

Positives are every annotated node with out-degree ≥ 2, plus the base-graph node matched to it within 7 µm, centred on the predicted position so that training and inference see the same kind of centre. Negatives are annotated continuing cells more than two undirected hops from any division (grandparents, daughters and grand-daughters are excluded as ambiguous), hard negatives — unmatched base-graph nodes 4–14 µm from a dividing parent at t−1 … t+1 — and a hashed sample of unannotated base-graph nodes (the true division rate per node is about 0.1 %, so the label noise is negligible next to the gain in negative diversity). The first store held 90,407 crops with 289 positives.

One round of self-training added 967 pseudo-positives: unannotated nodes with an out-of-fold logit ≥ 5 and geometric fork evidence (an orphan within the add radii at t+1). It was the single largest step of the classifier line (+0.004 on both leaderboards). A second round with 1,253 more pseudo-positives added nothing on the corpus and lost 0.003 on the public leaderboard.

### Inference

Five fold models averaged over four flip views. Only candidates from stage 3 are scored, a few thousand nodes per video instead of ~30,000, so a video costs about 100 s on a T4.

## 6. Fork Editing Policy

- Veto. An existing fork whose parent logit is below −1 keeps only the child nearest the constant-velocity prediction of the parent's track; the other edge is removed.

- Add. An out-degree-1 node with logit ≥ 1.5 gains the orphan q at t+1 that minimises |p − q| + 0.15 · |c1 − q|, subject to |p − q| ≤ 13 µm and |c1 − q| ≤ 16 µm, where c1 is the existing child. No new fork is created adjacent (± 1 frame) to an existing one, each node gains or loses at most one edge, and in-degree ≤ 1 / out-degree ≤ 2 hold by construction.

Re-parenting a daughter that is already linked to another node — the largest census bucket — was implemented in three forms (plain, a midpoint-crop "twin" test, a linked-node penalty) and rejected: every variant lost edges for at most two additional division true positives. The final policy edits orphans only. On the four public videos it vetoed 33 forks and added 46, out of about 117,000 edges.

## 7. Production Stage

The hidden test set is larger than the public one, and a stage that processes videos in name order under a fixed deadline would leave the tail of one embryo at base quality. The production stage therefore runs one worker per visible GPU with videos balanced by node count, alternates embryos smallest-first, derives its deadline from the session time the base pipeline actually used (`min(fixed, 11 h − elapsed − margin)`), and falls back to the base edges for any dataset a worker did not finish. Every output was audited by an independent script (consecutive ids, the runtime dataset set, in-volume coordinates, resolved endpoints, consecutive frames, degree limits, no duplicate edges) before submission.

## 8. Results

Table 1. Building the stage on the 179 annotated training videos. Same base graphs, same videos, official scoring implementation; classifier scores are out-of-fold (v7, five folds, 4-view TTA), the base is in-sample.

| ID | Configuration / change | Adjusted edge Jaccard | Division TP / FP / FN | Division Jaccard | Total score |
| --- | --- | --- | --- | --- | --- |
| B0 | Public 0.947 pipeline (DS-090), no stage | 0.8998 | 24 / 110 / 117 | 0.096 | 0.9094 |
| B1 | + veto: forks with parent logit < −1 lose the farther child | 0.9000 | 24 / 10 / 117 | 0.159 | 0.9160 |
| B2 | + orphan add at threshold 3.0 | 0.9001 | 45 / 19 / 96 | 0.281 | 0.9282 |
| B3 | add threshold 2.5 (final B) | 0.9001 | 46 / 25 / 95 | 0.277 | 0.9278 |
| B4 | add threshold 2.0 | 0.9000 | 47 / 38 / 94 | 0.263 | 0.9263 |
| B5 | add threshold 1.5 (final E) | 0.9000 | 47 / 46 / 94 | 0.251 | 0.9251 |

The veto alone removes 100 of the base's 110 division false positives without losing a true positive; the add recovers 21–23 of the 117 missed divisions. Adjusted edge Jaccard moves by at most +0.0003, as it should for a stage that only edits fork edges. The corpus optimum is threshold 3.0; both leaderboards preferred looser thresholds.

Table 2. Submissions. All runs use the same weights and radii; only the base and the policy change.

| Base | Classifier | add threshold | veto | Public | Private |
| --- | --- | --- | --- | --- | --- |
| public 0.947 pipeline | — | — | — | 0.947 | 0.916 |
| public 0.947 pipeline | v1 (4 frames, 16 µm window), veto only | — | 0 | 0.946 | 0.914 |
| public 0.947 pipeline | v2 (33 µm window) | 3.0 | 0 | 0.953 | 0.924 |
| public 0.947 pipeline | v3 (v2 + self-training) | 2.0 | −1 | 0.959 | 0.931 |
| public 0.947 pipeline | v3 | 1.0 | −1 | 0.959 | 0.933 |
| public 0.947 pipeline | v7 (v3 recipe, 6 frames) | 2.0 | −1 | 0.959 | 0.930 |
| public 0.953 notebook | v3 | 2.0 | −1 | 0.965 | 0.932 |
| public 0.953 notebook | v7 | 3.0 | 0 | 0.963 | 0.927 |
| public 0.953 notebook | v7 ( final B ) | 2.5 | −1 | 0.966 | 0.928 |
| public 0.953 notebook | v7 ( final E ) | 1.5 | −1 | 0.969 | 0.932 |
| public 0.953 notebook | v7 | 1.0 | −1 | 0.968 | 0.933 |
| public 0.953 notebook | v7 | 0.5 | −1 | 0.966 | 0.934 |

![Figure 3 — the same controlled comparisons on the public and on the private share](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F18476336%2F9edebea4d61b7ecc2eeac1ff3a88c7e9%2Fwriteup_fig3_public_vs_private.png?generation=1790777752742835&alt=media)

Across our 92 scored submissions the two leaderboards correlate at r = 0.97 with a mean drop of 0.021. What transferred: the division stage (+0.015 private on the 0.947 base), self-training (+0.004), and the direction of the add threshold — private scores rise monotonically as the threshold falls, from 0.927 at 3.0 to 0.934 at 0.5. What did not: the 0.953 notebook's additional edge-side stages (+0.006 public, +0.001 private with the same stage on top), the public 0.947 notebook itself (0.916 private against 0.917–0.919 for our own August pipeline), and the six-frame v7 classifier, which tied v3 on the public share and was 0.001–0.002 worse on the private share. Our second final was chosen as a conservative hedge (threshold 2.5, 0.928 private); the private share rewarded the opposite direction, and the loosest threshold we submitted would have scored 0.934.

## 9. What Did Not Work

- Vetoing the base's geometric forks with the classifier alone (corpus +0.004): 0.946 public, 0.914 private, below the base. Adds transfer; vetoes barely do.

- A learned edge verifier (four-frame midpoint crop with endpoint markers, OOF AP 0.957): the base's false-positive rate among scored edges is 0.18 %, so at p < 0.1 the verifier flags 2.4 % of edges and half of the labelled ones are correct. All eight repair policies were negative on the corpus; prune-only lost 0.001 on both leaderboards.

- Ensembles, 8-view TTA, a width-48 model, self-training round 2: flat on the corpus, 0.001–0.005 worse on Public.

- Training our own detector and linker on all 199 videos: about two epochs against the public checkpoints' ~50 on a weekly GPU quota; 0.677 public. Pair and triplet rankers on the public linker's frozen features: none passed its out-of-fold gate. Detector-threshold and secondary-weight tuning: +0.002 … +0.004 on in-sample validation videos, −0.001 … −0.005 on the public leaderboard.

- Node-count levers (minimum track length, isolated-node pruning, edge-admission thresholds): inert or negative, because the metric scores 3.3 % of the edges and the node term penalises every extra node.

## 10. Code

- Detailed write-up notebook with all figures, ablations, and clean reproduction code for the stage and the audit:[https://www.kaggle.com/code/mige551/biohub-solution-public-base-temporal-divcnn](https://www.kaggle.com/code/mige551/biohub-solution-public-base-temporal-divcnn)

- DivCNN v7 weights (five fold models): [https://www.kaggle.com/datasets/mige551/biohub-ds159-divcnn-v7](https://www.kaggle.com/datasets/mige551/biohub-ds159-divcnn-v7)

- Base notebooks: `anvithpothula/biohub-0-953-lb-original` and its coordinate head, the public 0.946 / 0.947 forks by `sjlee101`, and `pilkwang`'s detectors and tracking support pack.

## Citation

mige551. 66th Place Solution. [https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/writeups/66th-place-solution](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/writeups/66th-place-solution). 2026. Kaggle
