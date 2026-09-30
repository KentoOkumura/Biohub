# 6th Place Solution: Six-Model Detect-and-Link Ensemble + Global ILP

- archived_at: 2026-09-30T12:00:28.358068+00:00
- source: https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/744582
- author: Cyrus
- posted_at_utc: 2026-09-30 11:24:45.572000
- votes: 2
- comments: 0

# Biohub - Cell Tracking During Development — 6th Place Solution

Final standing: private 0.953, public 0.961, 6th place.

## Overview

Our solution is a learned tracker in three parts:

- Models. Six networks, of two architectures. Each network detects cells and scores their links to the previous frame in a single pass.

- Ensemble. The six networks share one set of detected cells. Their link scores are averaged after each network's own gradient-boosted re-scorer has re-ranked them.

- Solver. A global integer linear program (ILP) turns cells and link scores into tracks and divisions. A drift-compensated smoother then refines the positions.

The models were improved by several rounds of teacher → student pseudo-labelling. The whole pipeline runs on both of Kaggle's T4 GPUs within the 12-hour limit.

## ARCHIVEBLOCK00000END

## 1. Task and metric

Data. Each movie has 100 frames. Each frame is 64 × 256 × 256 voxels, with anisotropic spacing: 1.625 µm in z and 0.406 µm in y and x.

Output. Cell centres in every frame, plus the edges that link each cell at time t to its successor(s) at t + 1. A division is a cell with two successors. This is close to the Cell Tracking Challenge setting [2], run here as a Kaggle competition [1].

Metric. Two terms. The main one is an edge Jaccard index with a penalty on the node count, per movie:

$$J_{adj} = \max\left(0,\; J_{edge} \cdot \left(1 - 0.1\,\frac{N_{pred} - N_{est}}{N_{est}}\right)\right)$$

$N_{est}$ is the organisers' estimate of the number of cells. Movies are averaged with weights TP + FP + FN of their edges. A division Jaccard, pooled over all movies, is added with weight 0.1:

$$\text{score} = \overline{J_{adj}} + 0.1 \cdot J_{div}$$

Annotations are extremely sparse. The 199 training movies contain 133k annotated cells, which is only about 2.8 % of the cells present. The annotations are whole lineages, and there are only 151 divisions in total. Two things follow:

- Most of the image is neither a labelled positive nor a labelled negative.

- Most link errors are identity swaps with unannotated neighbours.

## 2. Models

### 2.1 Common design

Both architectures share the same interface:

- Input. A 3-frame window (t−1, t, t+1) at native resolution: no resampling and no cropping at inference. Each frame is normalised on its own (min–max, then z-score).

- Output. A cell heatmap, sub-voxel offsets, and a set of link predictions, all described below.

![source image](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F3842406%2F4c062b8f6b84200c4f328c5f58db8729%2F2_model.png?generation=1790767309078318&alt=media)

 Encoder. The encoder is a 3D U-Net-style [3, 4] conv stack (Conv3d → GroupNorm [5] → GELU [6]). Its stride schedule handles the voxel anisotropy in two steps:

- Two lateral-only reductions reach an isotropic 1.625 µm grid.

- A 3D reduction reaches a coarse 3.25 µm grid.

The encoder processes each frame independently, so at inference every frame is encoded once and reused by all the windows that contain it.

Temporal fusion. On the coarse grid, each voxel of frame t predicts a bounded displacement into each neighbour frame (up to about 6.5 µm). It samples the neighbour's features there and attention-weights them against its own. This aligns moving cells before they are compared. At initialisation the displacements are zero and the attention is uniform over the three frames.

Detection. The decoder upsamples back to a 64 × 128 × 128 grid with skip connections. It predicts:

- a centre heatmap, turned into cells by non-maximum suppression and a threshold of 0.5;

- a sub-voxel offset per cell.

Each cell's descriptor is sampled from the features at its continuous centre.

Association head. This is where tracking happens. For each pair of frames:

- Each cell descriptor predicts a velocity and a per-axis uncertainty.

- Each candidate link gets geometric features: the displacement, and the residual displacement − velocity · Δt, normalised by the predicted uncertainty.

- Several layers of bidirectional sparse graph attention [7, 8] run over the candidate links. Cells at t and at t + 1 update each other.

- The head outputs, for each cell, a softmax over its candidate parents plus a "new cell" class. It also outputs a division score per parent, a score for each pair of daughters, and the velocity.

Because "no parent" is an explicit and well-calibrated probability, it becomes the solver's appearance cost (§5.5). Scoring links with attention over cell descriptors is related to Trackastra [9]. Ours is sparse, runs on a k-NN candidate graph, and is trained jointly with the detector.

### 2.2 The two architectures

|  | IsotropicLineageNet | MultiScaleLineageNet |
| --- | --- | --- |
| Temporal fusion | motion-guided attention on the coarse grid | gated fusion at two scales (coarse + isotropic), several learned samples per neighbour; the gate is zero-initialised, so training starts as a per-frame detector |
| Detection vs linking features | shared | separate residual task adapters |
| Cell descriptor | centre sample + local mean | + learned sampling offsets (≤ 3 µm) with attention, in the spirit of deformable sampling [10] |
| Association head | attention blocks | + in-graph motion refinement : a soft assignment of each cell to its candidate successors updates its velocity and uncertainty midway through the head |
| Linker training input | annotated positions with jitter | annotated cells moved onto the detector's own matched detections (75 %) |
| Size | 2.5–4.8 M params | 5.7 M params |

### 2.3 The six models in the ensemble

| model | architecture | width / linker depth | params | training | TTA views |
| --- | --- | --- | --- | --- | --- |
| A | Isotropic | 48 ch / 3 blocks | 2.5 M | 3 rounds of noisy-student pseudo-labelling | 4 |
| B | Isotropic, wide | 64 ch / 3 blocks | 4.4 M | fine-tuned on ensemble pseudo-labels | 4 |
| C | Isotropic, wide + deep | 64 ch / 4 blocks | 4.8 M | fine-tuned on ensemble pseudo-labels | 4 |
| D | MultiScale | 64 ch / 4 blocks | 5.7 M | from scratch on ensemble pseudo-labels | 3 |
| E | MultiScale | 64 ch / 4 blocks | 5.7 M | from scratch on ensemble pseudo-labels | 2 |
| F | MultiScale | 64 ch / 4 blocks | 5.7 M | from scratch on ensemble pseudo-labels | 2 |

Every model was trained on a different random, movie-disjoint train/validation split, so the members make different mistakes. All six vote on the cells with equal weight.

## 3. Training

Three-state detection targets. Every voxel is in one of three states:

| state | definition | loss weight |
| --- | --- | --- |
| positive | a Gaussian around an annotated centre (within 3 µm) | full |
| verified background | darker than the frame median and more than 6 µm from any annotation | full |
| unknown | everything else, where the unannotated ~97 % of cells live | 0 |

A count prior supplies the missing signal about how many cells the unknown region holds. It matches the summed heatmap mass to the organisers' per-movie cell-count estimate.

Losses. Each loss is computed separately on ground truth and on pseudo-labels, then combined as $L = L_{GT} + 0.5 \cdot L_{pseudo}$. That keeps the sparse ground truth from being drowned out.

| loss | weight |
| --- | --- |
| penalty-reduced focal loss on centres [11, 12] | 1.0 |
| offset (smooth L1) | 1.0 |
| parent cross-entropy over candidates + "new cell" (15 % of true parents dropped on purpose to teach it) | 1.0 |
| division BCE (positive weight 20) | 0.5 |
| daughter-pair BCE | 0.25 |
| velocity Gaussian NLL | 0.1 |
| count prior | 1.0 |

Samples. Full-frame 3-frame crops. About a fifth of the crops (21 %) are centred on divisions. The linker trains on k-NN candidate graphs that include distractor intensity peaks.

Augmentation.

- Lateral D4 only: light attenuation makes z directional, so z is never flipped or rotated in.

- Gamma and Gaussian noise.

- Synthetic stage drift (models A, D, E, F).

- Noisy-Student noise [13]: shot noise and depth-dependent gain for all models; feature dropout 0.2 and weight decay 0.01 for the Isotropic models.

- Low-contrast haze. Each frame is blended with a blurred copy of itself at a random contrast, which mimics hazy, deep movies. It was worth +0.005 [+0.001, +0.009] on validation against its exact control (a single seed).

Optimisation.

- AdamW [14] with warm-up and cosine decay [15], gradient clipping, and an EMA of the weights [16] used at inference.

- bf16 mixed precision [17] on RTX 3090/4090 GPUs.

- 24 epochs (A) or 32 epochs (B–E). F is an intermediate checkpoint after 24 of its 32 epochs.

## 4. Pseudo-labels and teacher → student retraining

With about 3 % of cells annotated, a model trained on ground truth alone has two blind spots:

- it is never shown how to separate touching cells;

- it is never penalised for linking to an unannotated neighbour.

We fixed both with offline Noisy-Student self-training [13, 18]. The teacher is the whole pipeline, not one model.

![source image](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F3842406%2Ffa9a0f065a4311ca3391c067eddb44c4%2F3_pseudo_loop.png?generation=1790767350904177&alt=media)

How the labels are built.

- Topology comes from the ILP solution. With raw positions its edge precision is 0.945, against 0.885 for the model's raw argmax.

- Positions come from the raw detections, not the smoothed tracks. The smoother helps the metric but moves points away from the true centres. The raw positions are bias-corrected per imaging cohort.

- Ground truth always wins.
- A pseudo cell within 4 µm of an annotated cell is that cell.

- Pseudo links that contradict an annotated parent or child are removed.

- Low-confidence items (probability < 0.5) are removed.

- Divisions and velocities are supervised by ground truth only.

- The teacher's probabilities become per-item loss weights.

- The 4 public test movies are never labelled. Validation movies are labelled from the teacher's out-of-fold decode, so the later students (B–F) saw pseudo-labels on part of the validation set.

- The last two label sets followed the leaderboard. Each was regenerated from the best public recipe at the time: 3-model ensembles (A + a wide and a deep student) scoring 0.958 and then 0.959 public. The final set has about 5.0 M cells and 4.9 M links over 195 movies, about 38× the annotated cells on the same movies.

Lineage of the final models.

![source image](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F3842406%2Ff6485fc98636d02cb0d9defadf6bea0c%2F4_lineage.png?generation=1790767361776785&alt=media)

The first round gave the largest single gain: +0.008 on validation over the ground-truth-only model. Plain self-distillation stopped helping after that: round 2 was flat, and round 3's gain came from the low-contrast augmentation. The later gains came from new architectures and new splits trained on ensemble labels.

## 5. Inference

### 5.1 Decoding and test-time augmentation (TTA)

- Windows. Each frame is decoded from the 3-frame window centred on it.

- TTA. Each model runs 2–4 lateral rotations. Outputs are un-rotated (including the offset vectors) and averaged in logit space. Cell descriptors are sampled from the view-averaged features, so every head benefits from TTA, not only detection. On an early single model, going from 1 to 4 views was worth +0.05 score (+0.03 raw edge Jaccard).

- Encoder cache. The per-frame encoder runs once per frame and view, not once per window. It is kept in fp16 and makes decoding 1.4–1.9× faster.

### 5.2 Ensemble: shared cells, averaged links

![source image](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F3842406%2Fef7a7921f56c7afa50786fce8c2a5fef%2F5_ensemble.png?generation=1790767370147933&alt=media)

- Cells. The six models' centre logits and offsets are averaged, and peaks are extracted once.

- Links. Each model samples its own descriptors at the shared cells. Its own association head scores the same candidate links, and its own re-scorer re-ranks them. The six parent distributions, including "new cell", are then averaged in probability space. Division scores and velocities are averaged directly.

### 5.3 Candidate links

For each pair of frames, the candidates are the union of:

- the 4 nearest cells at t for each cell at t + 1;

- the 4 nearest cells at t + 1 for each cell at t.

Both searches are limited to 20 µm. The true parent is a candidate for about 99.4 % of annotated links. Candidates are not pruned by probability: the softmax already has a "new cell" option.

### 5.4 Edge re-scorer

Identity swaps between neighbours were the largest error class in our validation analyses (51–57 % of edge errors). A LightGBM [19] model per ensemble member re-ranks each cell's candidate parents.

Features, 23 per candidate link:

- the model's log-probability, its rank, the margin to the best rival, the number of candidates, and the "new cell" probability;

- raw and drift-corrected distances, split into z and lateral parts, and the velocity residual;

- the detection confidence of both cells, and the parent's division score;

- competition for the parent: its best score to another cell, how many cells rank it first, and its out-degree;

- local cell density, depth, and relative time.

Keeping "new cell" untouched. The new parent log-probability is

$$\log P'(\text{parent}) = \log(1 - P_{new}) + \text{log-softmax}(\text{tree scores over the candidates})$$

So the trees change which parent wins, not the probability that a cell is new.

Training.

- 300 trees, 31 leaves, binary objective.

- Each member has its own trees, fitted on that member's own scores on the ensemble's shared cells, over its own held-out validation movies. During validation the trees are applied out-of-fold (5 folds by movie), so no movie is scored by trees that saw it.

- Worth +0.004 to +0.007 per model on validation.

Deployment. The trees are exported to flat arrays and evaluated with a compiled parallel tree walk [20], so no gradient-boosting library is needed at inference.

### 5.5 Global ILP

Each cell has four binary variables: exists, appears, disappears, divides. Each candidate link has one variable. The constraints are flow conservation:

$$\text{appear}j + \sum_i \text{edge}{ij} = \text{node}j \qquad \text{disappear}_i + \sum_j \text{edge}{ij} = \text{node}_i + \text{div}_i \qquad \text{div}_i \le \text{node}_i$$

This is the conservation-tracking family of models [21, 22]. The costs come from the networks, plus four tuned constants (disappearance 12, appearance bias 0.5, temperature 0.9, distance weight 0.2):

| term | cost |
| --- | --- |
| cell | − centre logit (break-even at p = 0.5) |
| appearance | − log P_T(new cell) + 0.5 (cells after the first frame; first-frame cells appear for free) |
| disappearance | 12 (the models have no disappearance head) |
| division | max(0, − division logit) |
| link | − log P_T(parent) + 0.2 · drift-corrected distance |

P_T is the re-scored, ensemble-averaged distribution P′ with temperature T = 0.9 applied, renormalised over the candidates plus "new cell".

Drift-corrected distance. The microscope stage drifts between frames: 1.3 µm median, 7.3 µm at the 99th percentile. Identity swaps concentrate on those frames. For each pair of frames we estimate the global shift as the median displacement of the models' confident links (P ≥ 0.7). Each link then pays for its distance after that shift is removed. This was worth +0.009 on validation, mostly by removing false links.

LP-first solving. Apart from the division rows, the constraint matrix is a network matrix, so the LP relaxation is almost integral. On the hardest movie, 783 of 749k variables were fractional.

- Solve the LP with HiGHS [23].

- Fix every integral variable.

- Re-solve a small MILP over the fractional variables and their neighbourhood.

This is 38–166× faster than a direct branch-and-bound solve with SCIP [24]. It is a heuristic, but it certifies its own gap: in our tests it landed 0–0.25 % above the LP bound (0.04 % on the hardest movie). SCIP runs only when the gap exceeds 0.5 %, and a greedy solver is the last resort.

Because every cell and every appearance has a price, the ILP produces no dangling fragments. No gap-closing or pruning heuristics are needed after the solve; only a structural check (in-degree ≤ 1, out-degree ≤ 2) runs.

### 5.6 Drift-compensated smoothing

Detected positions are coarse: the z grid step is 1.625 µm, and z carries most of the localisation error. So we apply a temporal line-fit smoother:

- For each cell, take up to 2 neighbours in each direction along its track. The forward walk stops at a division; a daughter's backward walk continues into its mother. Cells with fewer than 3 chain nodes are left as they are.

- Fit a straight line per axis.

- Move the cell to 0.2 · original + 0.8 · fit. Track ends get the one-sided fit.

Smoothing directly would blur the frames where the whole stage jumps. So we first remove the cumulative stage drift, estimated as the median displacement of the solved one-to-one links. We smooth in that drift-free frame and then add the drift back. This was worth +0.005 over plain smoothing.

Positions are finally clamped to the volume and rounded to voxel indices.

## 6. Validation

- Local metric. We re-implemented the metric, including the per-movie weighting and the division term. The CSV written by our Kaggle kernel on the 4 public test movies scores exactly the same locally as our own run, so what we measured is what ran on Kaggle.

- Splits. Each model has its own movie-disjoint split: 165 training and 30 validation movies. The 4 public test movies are held out of every model. One fixed set of 30 validation movies (model A's) ranks every change, including ensembles.

- Tuning. Every model or inference change gets its own small solver grid: disappearance cost, node bias, appearance bias, distance weight, division bias, temperature and smoothing weight. Grid points are compared with 2-fold cross-validation across the validation movies.

- Decisions. Comparisons use a paired bootstrap over movies (4,000 resamples) and are read from the confidence interval. Arms are ranked end to end on the tracking metric. Within a training run, checkpoints are selected by validation loss.

The final submission scored 0.940 on the validation movies and 0.928 on the 4 public test movies (adjusted edge Jaccard only: with 151 divisions in the whole corpus, the division term is too noisy to rank changes by). The validation number is at the best grid point (disappearance 12, appearance bias 0.5, temperature 0.9) and is therefore optimistic.

## 7. Running on two Kaggle GPUs within 12 hours

![source image](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F3842406%2F8d87226b3438c703fa5e0ec4d96d548e%2F6_kaggle_gpus.png?generation=1790767382702527&alt=media)

- One worker per GPU, one shared queue. Each worker loads all six models once. Workers claim movies by atomically renaming a per-movie file, so a faster GPU simply takes more movies. CPU threads for the tree walk and the maths libraries are split between the workers.

- fp16, not bf16. The T4 has no native bf16. Switching the T4 from emulated bf16 to fp16 [17] cut single-model prediction time by 3.5× (19.1 → 5.4 min on the 4 public test movies), with no measurable score change.

- Planned budget. Kaggle run time is predicted from local GPU timings with a linear model fitted on earlier kernel runs. It gave the expected cost of the per-model TTA view counts (4/4/4/3/2/2) before submitting.

- Deadline guard. Inside each worker, before each movie, the worker estimates the movie's cost from its recent history. It picks the richest level that still fits the remaining time: all views, then 75 %, then 50 %, then 1 view. Quality degrades gradually rather than all at once.

- Every movie gets a prediction. A movie that fails is re-run with fewer views. If that fails, it is run with the primary model alone. As a last resort, a model-free tracker runs: difference-of-Gaussians detection [25] plus drift-corrected nearest-neighbour links. The submission is therefore always complete, even if a movie fails.

- Other safeguards. Expandable CUDA memory segments prevent fragmentation out-of-memory errors on the 16 GB cards. All dependencies install offline.

## 8. Key takeaways

- Let the network output probabilities the solver can use. A parent softmax with an explicit "new cell" class gives calibrated appearance and link costs, and makes post-processing heuristics unnecessary.

- Take pseudo-labels from the whole pipeline. Use ILP topology, raw detector positions and authoritative ground truth, and regenerate the labels from the current best pipeline.

- Diversify splits as well as architectures. Members trained on different movies make different mistakes, which is what the ensemble averages over.

- Model the camera, not only the cells. Correcting for stage drift in both the solver and the smoother removed many false links and improved positions.

- Engineer the time budget. fp16 on the T4, the encoder cache, two-GPU sharding and a graded deadline guard are what made a 6-model ensemble with TTA fit in 12 hours.

## References

- Biohub. Biohub - Cell Tracking During Development. Kaggle competition, 2026. [https://www.kaggle.com/competitions/biohub-cell-tracking-during-development](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development)

- M. Maška et al. The Cell Tracking Challenge: 10 years of objective benchmarking. Nature Methods 20, 1010–1020, 2023.

- O. Ronneberger, P. Fischer, T. Brox. U-Net: Convolutional Networks for Biomedical Image Segmentation. MICCAI 2015.

- Ö. Çiçek, A. Abdulkadir, S. S. Lienkamp, T. Brox, O. Ronneberger. 3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation. MICCAI 2016.

- Y. Wu, K. He. Group Normalization. ECCV 2018.

- D. Hendrycks, K. Gimpel. Gaussian Error Linear Units (GELUs). arXiv:1606.08415, 2016.

- P. Veličković, G. Cucurull, A. Casanova, A. Romero, P. Liò, Y. Bengio. Graph Attention Networks. ICLR 2018.

- A. Vaswani et al. Attention Is All You Need. NeurIPS 2017.

- B. Gallusser, M. Weigert. Trackastra: Transformer-based cell tracking for live-cell microscopy. ECCV 2024. arXiv:2405.15700.

- J. Dai, H. Qi, Y. Xiong, Y. Li, G. Zhang, H. Hu, Y. Wei. Deformable Convolutional Networks. ICCV 2017.

- H. Law, J. Deng. CornerNet: Detecting Objects as Paired Keypoints. ECCV 2018.

- X. Zhou, D. Wang, P. Krähenbühl. Objects as Points. arXiv:1904.07850, 2019.

- Q. Xie, M.-T. Luong, E. Hovy, Q. V. Le. Self-training with Noisy Student improves ImageNet classification. CVPR 2020.

- I. Loshchilov, F. Hutter. Decoupled Weight Decay Regularization. ICLR 2019.

- I. Loshchilov, F. Hutter. SGDR: Stochastic Gradient Descent with Warm Restarts. ICLR 2017.

- B. T. Polyak, A. B. Juditsky. Acceleration of Stochastic Approximation by Averaging. SIAM Journal on Control and Optimization 30(4), 838–855, 1992.

- P. Micikevicius et al. Mixed Precision Training. ICLR 2018.

- D.-H. Lee. Pseudo-Label: The Simple and Efficient Semi-Supervised Learning Method for Deep Neural Networks. ICML 2013 Workshop on Challenges in Representation Learning.

- G. Ke, Q. Meng, T. Finley, T. Wang, W. Chen, W. Ma, Q. Ye, T.-Y. Liu. LightGBM: A Highly Efficient Gradient Boosting Decision Tree. NeurIPS 2017.

- S. K. Lam, A. Pitrou, S. Seibert. Numba: A LLVM-based Python JIT Compiler. LLVM-HPC 2015.

- B. X. Kausler, M. Schiegg, B. Andres, M. Lindner, U. Köthe, H. Leitte, J. Wittbrodt, L. Hufnagel, F. A. Hamprecht. A Discrete Chain Graph Model for 3d+t Cell Tracking with High Misdetection Robustness. ECCV 2012.

- M. Schiegg, P. Hanslovsky, B. X. Kausler, L. Hufnagel, F. A. Hamprecht. Conservation Tracking. ICCV 2013, 2928–2935.

- Q. Huangfu, J. A. J. Hall. Parallelizing the dual revised simplex method. Mathematical Programming Computation 10, 119–142, 2018.

- S. Bolusani et al. The SCIP Optimization Suite 9.0. arXiv:2402.17702, 2024.

- D. G. Lowe. Distinctive Image Features from Scale-Invariant Keypoints. IJCV 60(2), 91–110, 2004.

## How to cite

```text
@misc{saeedi2026biohubtracking,
  author       = {Saeedi, Jamal},
  title        = {Biohub - Cell Tracking During Development: 6th Place Solution},
  year         = {2026},
  howpublished = {Kaggle competition write-up},
  url          = {https://www.kaggle.com/competitions/biohub-cell-tracking-during-development}
}

```
