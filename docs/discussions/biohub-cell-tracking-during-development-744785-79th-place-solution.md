# 79th Place Solution

- archived_at: 2026-10-02T11:53:17.411685+00:00
- source: https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/744785
- author: Yudai Yamauchi
- posted_at_utc: 2026-10-01 00:05:06.879000
- votes: 1
- comments: 0

# 79th Place Solution

Learned division recovery on top of a public 3D tracking pipeline

Team: Biomed Foresight — Yudai Yamauchi and Tsukasa Miyaji
 Private rank: 79 (preliminary) · Private LB: 0.930 (preliminary) · Best Public LB: 0.966
 Code: [GitHub repository](https://github.com/YamauchiYudai/biohub-cell-tracking-solution)

Thank you to the organisers and to everyone who shared notebooks, pretrained models and ideas during the competition. Our solution builds on that public work. This writeup describes the parts we added, with a focus on recovering cell divisions.

## 1. Overview

We kept the public detector, association Transformer and global ILP tracker, then added a small image model and targeted graph repairs. Learned division recovery was our largest measured Public LB gain: 0.954 to 0.964. The final selected pair reached a best Public score of 0.966; Kaggle currently shows a team Private score of 0.930 and rank 79, subject to verification. We have not independently verified the two selected submissions' individual Private scores.

![Public tracker and team additions](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F9840636%2Fdfa717bced88c34a14a8a97ae5f7b8b7%2Fsolution_pipeline.png?generation=1790813312856549&alt=media)

Figure 1. The deployed pipeline. Grey boxes show inherited stages and the output; blue boxes contain our relinking and coordinate changes; orange highlights the learned division stage. The graph-repair box combines public repairs with our b1c relink change. The F03 head is used in one selected submission only.

Three observations guided our work:

- The inherited ILP did not produce forks. Its division cost made a second daughter edge unfavourable. Divisions were left to rule-based post-processing, which missed many events.

- Repeated frames broke motion assumptions. Some movies contain identical consecutive frames followed by a larger movement. A relinker using the previous displacement then predicts the wrong next position.

- Sparse labels make deletion risky. Many predicted cells and divisions cannot be judged by the metric. Removing no annotated true division does not establish that no real division was removed.

We addressed these with a three-frame division CNN, jump-aware relinking and smoothing, and a final coordinate-consensus step. We retained the original detection and association models.

## 2. Data, metric and validation

The training set contains 199 movies from two embryos: 71 from 44b6 and 128 from 6bba. Native voxel spacing is 1.625 × 0.40625 × 0.40625 µm. Only a subset of cell lineages is annotated, including 151 divisions across 87 movies.

The official score combines a cell-count-adjusted edge Jaccard with 0.1 × division Jaccard. We used the organisers' evaluator, including its submission-coordinate rounding and aggregation. We inspected node counts and edge/division TP, FP and FN alongside the combined score: adding cells can recover edges while worsening the count adjustment.

### 2.1 What our local scores measure

The public pretrained detector, Transformer and DeepCenter weights had seen all 199 training movies. The four visible test movies are copies of training movies. A score from the full inherited pipeline on these movies is an in-sample diagnostic, not an out-of-fold estimate of hidden-test performance.

For the T3 division model, we trained on one embryo and evaluated on the other in both directions. Thresholds were selected from movie-grouped CV within the training embryo. This tests transfer of the added model, but does not make the complete pipeline out-of-fold because the inherited models remain in-sample.

We also used an external Zebrahub set, `val_biohub`, to screen detector and association changes. Its tracks were generated automatically with Ultrack, so it was a diagnostic rather than human-verified ground truth. It was used for evaluation only, not for gradients or fine-tuning.

### 2.2 Exact CPU replay

We cached GPU inference outputs once and replayed the submission notebook's own post-processing on CPU. The 88-movie set contains the 87 movies with annotated divisions plus one additional movie: 22 from 44b6 and 66 from 6bba. Replay reproduced the graphs entering the division stage and per-movie division counts; its final aggregate score agreed with the real pipeline within 0.00003.

For late post-processing experiments, we checked the score change per embryo, whether it survived removing the most helpful movie, and whether the end-to-end GPU notebook reproduced it. A typical promotion gate was at least +0.001 overall with neither embryo worse. The F03 head was a documented exception: it failed its offline gate and was retained in only one of the two final selections after a positive Public result.

## 3. Learned division recovery

### 3.1 Why add a stage after the ILP?

In our inherited configuration, an edge contributes −p and opening a division costs 1.2. A second daughter edge therefore contributes 1.2 − p > 0 for p in [0, 1]. Under these settings it is unfavourable to open a fork; we observed no two-child nodes in the inherited ILP output. This is a property of our configuration, not a limitation of ILP tracking in general.

The public notebook adds rule-based safe divisions afterwards, using distance, symmetry, mutual-neighbour checks, daughter separation and a DeepCenter veto. A measurement submission with that stage disabled fell from 0.954 to 0.914 on Public. That motivated improving division recovery rather than replacing the linker.

![Division recovery before and after T3](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F9840636%2F477f6063bee00a5f5c4328a97e24f010%2Fdivision_recovery.png?generation=1790813358587491&alt=media)

Figure 2. A schematic of the graph change. The parent P already has daughter D1. When an unlinked track start D2 passes the candidate gates and the T3 score supports division, we add the missing P→D2 edge. This stage adds an edge between existing nodes; it does not invent a detection. The drawing is not microscopy data.

### 3.2 T3: image evidence over three frames

T3 is a small 3D CNN receiving the raw crop around a candidate parent at t−1, t and t+1 as three channels. The purpose is to distinguish a dividing nucleus from a nearby, unrelated track start using temporal image evidence rather than geometry alone.

![Three-frame division input](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F9840636%2Fb4e30d4d14850b73b5da038ed689d113%2Fdivision_timeline.png?generation=1790813374239102&alt=media)

Figure 3. The three-frame input, illustrated schematically. A parent at t is followed by two daughters at t+1. The same spatial crop is used for all three frames. These are drawn nuclei, not competition images.

| Model setting | Value |
| --- | --- |
| Input | 3 frames × 16 × 48 × 48 native voxels (Z × Y × X) |
| Physical crop | 26 × 19.5 × 19.5 µm |
| Network | 3D convolutions, XY pooling first, global pooling and a binary head |
| Normalisation | Per-crop robust intensity scaling over all three frames |
| Positives | Annotated dividing parents, including ±1-frame offsets |
| Negatives | Annotated one-child cells away from division events |
| Training augmentation | XY flips/rotations, small spatial shifts and intensity scaling; positives repeated 10× |
| Submission ensemble | 3 seeds trained on all 199 movies, averaged over 2 test-time views |

Unannotated cells were not treated as negative division examples. Cross-embryo results were AUC/AP 0.864/0.554 for 44b6→6bba and 0.942/0.694 for 6bba→44b6.

### 3.3 Candidate gates and graph decisions

We enumerate a parent with one child and a track start at the next frame. Both daughters must be within 12 µm of the parent and within 16 µm of each other. An addition also requires:

- A cached association probability above 0.48 for the missing daughter edge.

- A parent-centred angle of at least 90° between the daughters.

- The daughters moving further apart at t+2.

Among eligible starts, we select the daughter closest to the mirror position of the existing child. We add the fork when T3 ≥ 0.3, and delete a rule-based fork only when T3 < 0.01. A temporal/spatial exclusion prevents counting one division twice. The final V5a variant exempts learned additions from the rule-based stage's per-frame budget.

The asymmetry is deliberate. Sparse annotations provide weak evidence for deleting real divisions. In the 88-movie replay, 18 of 28 scored additions were correct (64% precision). This is a training-movie diagnostic; it is not precision measured on hidden embryos.

## 4. Tracking and coordinate refinements

### 4.1 Jump-aware relinking (b1c)

Of 66 audited 6bba movies, 55 contain bit-identical consecutive frames; none of the 22 audited 44b6 movies does. The previous transition's flow is zero at a duplicated frame, while the next transition can contain a larger displacement. The inherited relinker then assigns cells to nearby neighbours instead of their true continuations.

We classify transitions as frozen, jump or normal. On jump transitions only, we add a global translation estimated by point-set cross-correlation and refined by mutual-nearest-neighbour ICP. Normal transitions keep the inherited behaviour. On the four visible movies, the score moved from 0.9180 to 0.9290; Public moved from the upstream 0.953 to 0.954.

### 4.2 Jump-aware smoothing (J2)

The original local line fit can average coordinates across two different alignments. J2 stops the track walk at frozen transitions and compensates for the translation at jumps before fitting. J2 with V5a added 0.00353 on the 88-movie replay, with both embryos positive. The displayed Public score remained 0.964.

### 4.3 Frozen-frame consensus and the coordinate head

For one-to-one chains across identical frames, we average the smoothed coordinates of the repeated nucleus. We leave a chain alone if the mean would move any node by more than 1.625 µm. Nodes, edges and divisions are unchanged. This added 0.00103 on replay and remained positive without the best movie.

F03 fine-tunes the public V1284 coordinate head with a distillation penalty toward the original head. It did not pass our movie-fold and cross-embryo offline comparison, but gave a positive visible-movie run and raised Public from 0.964 to 0.966 when combined with consensus. We selected both `fc_f03` and `fc`, which differ only in this head. Their individual Private scores are still unverified.

## 5. Results and ablations

![Division ablation and replay diagnostics](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F9840636%2Ff331c9cccd693182d97410e781c39c9c%2Fdivision_evidence.png?generation=1790813391322679&alt=media)

Figure 4. Left: observed Public scores from the divisions-off measurement, rule-based control and T3 addition. Middle: S2 division TP/FP/FN on the 88-movie in-sample replay. Right: precision of scored T3 additions on that replay. These panels describe different evaluations; the counts are not hidden-test labels.

| Pipeline / experiment | Local evidence | Public LB |
| --- | --- | --- |
| Public x138 notebook | Upstream reference | 0.953 |
| + b1c relink | Four visible movies: 0.9180→0.9290 | 0.954 |
| Same control, divisions disabled | Measurement of the inherited division stage | 0.914 |
| + T3 division recovery (S2) | Added-fork precision 18/28; division J improves in both embryos | 0.964 |
| + J2 and V5a | Replay +0.00353 over S2 | 0.964 |
| + frozen-frame consensus ( fc , selected) | Replay +0.00103 over J2 + V5a | 0.964 |
| + F03 head ( fc_f03 , selected) | Failed offline gate; positive visible-movie run | 0.966 |

The 0.010 division gain is an observed Public difference. We cannot attribute the team's Private 0.930 to individual components without matched private ablations. Likewise, a better training-movie replay does not establish better transfer to unseen embryos.

For the S2 replay, divisions were 44 TP / 35 FP / 107 FN, giving pooled division Jaccard 0.237. The counts were 6/10/20 on 44b6 and 38/25/87 on 6bba. These diagnostics motivated the division work, but are not a final out-of-fold score of the complete selected pipeline.

## 6. What did not work

| Experiment | Observation | Decision / lesson |
| --- | --- | --- |
| Our own detector | Classical baseline 0.616; our 3D U-Net 0.669; public pretrained stack much stronger | Adopt the public stack and focus on its measured failures |
| Alternative association models | Seven variants scored 0.870–0.946 against a 0.946 control | Keep the inherited linker; audit missing candidates and detection errors first |
| More aggressive fork deletion | S1 scored 0.950; combined T3/ranking variant S3 scored 0.958 | Absence of annotated positives is weak evidence that a division is false |
| Local-softmax detector fine-tune | −0.046 internally and on the external diagnostic | Reject before another submission |
| Weaker smoothing | Replay gain depended strongly on one movie; visible GPU result worsened | Retain the jump-aware change instead of lowering the smoothing weight |
| More gap filling, re-admission and relocalisation | Small or unstable replay gains below the promotion bar | Keep the existing settings |

The association experiments included hard-negative margins, rank priors, dense matching and temporal-memory variants. They were experiments, not parts of the selected pipeline.

## 7. Remaining weaknesses and lessons

Division recall remains low. On the S2 replay there were 107 missed divisions versus 44 recovered ones. Among 120 missed edges attributed to division events, 80 involved daughter localisation errors. A division classifier cannot repair a daughter that is missing or too far from its annotation to match.

Detection and localisation limit graph repairs. Our S2 failure attribution placed roughly 63% of remaining missed edges on the detection/localisation side. Expanding a candidate graph or replacing the linker alone cannot recover those cells.

The embryos differ, and our diagnostics are uneven. Frozen-frame changes apply to 6bba, while 44b6 has no such transitions in the audited set. The replay over-represents movies with divisions, inherited models are in-sample, and the external tracks are automatic. These are useful controls with clear limitations.

Public gains did not fully carry through to Private. Our best Public score was 0.966 and the displayed Private team score is 0.930. This difference does not identify which component failed to transfer. The F03 selection also shows that we did not consistently follow our offline gate. With more time, we would prioritise evaluation of the complete graph on unseen embryos and obtain matched evidence before choosing late changes.

The main lesson is to inspect the cells, candidate edges and forks behind a score. A targeted division stage helped more on Public than the new association models we tried, but a local improvement still needs evidence that it transfers to the test domain.

## 8. Code, resources and acknowledgements

The [repository](https://github.com/YamauchiYudai/biohub-cell-tracking-solution) contains the selected submission notebook, reusable team components, training/evaluation entry points, configuration and CPU parity tests.

- [Submission notebook](https://github.com/YamauchiYudai/biohub-cell-tracking-solution/blob/main/notebooks/final_submission.ipynb): the runnable `fc_f03` path.

- [Division implementation](https://github.com/YamauchiYudai/biohub-cell-tracking-solution/tree/main/src/biohub_tracking/division): candidates, image crops, T3 and graph decisions.

- [Tracking refinements](https://github.com/YamauchiYudai/biohub-cell-tracking-solution/tree/main/src/biohub_tracking/postprocessing): relinking, smoothing and consensus.

The submitted notebook's code-cell fingerprint matches the recorded submission. Team weights are referenced as Kaggle datasets, but were not anonymously accessible at our last check; exact weighted inference is not yet publicly reproducible. The training recipe is included. Competition data and weights are not bundled here.

We thank Anvith Pothula for the x138 notebook and V1284 head, Teddy Tennant for the frontier947 repairs, Reyhan Ksatria for the earlier 0.947 post-processing pipeline, pilkwang for the pretrained detectors, Transformer and DeepCenter, and royerlab/the organisers for the data, baseline and official metric.

Team-authored code and figures use MIT; redistributed upstream notebook material retains Apache 2.0. Sources and scope are recorded in [THIRD_PARTY_NOTICES.md](https://github.com/YamauchiYudai/biohub-cell-tracking-solution/blob/main/THIRD_PARTY_NOTICES.md). The figures above are original schematics or recorded aggregate measurements. No raw competition microscopy or annotations are embedded.
