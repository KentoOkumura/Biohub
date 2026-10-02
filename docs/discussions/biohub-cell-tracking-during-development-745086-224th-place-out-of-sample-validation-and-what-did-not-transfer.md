# 224th Place: Out-of-Sample Validation and What Did Not Transfer

- archived_at: 2026-10-02T11:53:19.196754+00:00
- source: https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/745086
- author: FOYSAL
- posted_at_utc: 2026-10-02 06:47:34.154000
- votes: 1
- comments: 0

# 224th place (team Hack2Publish): out-of-sample validation, many measured negatives, and a public LB that misled us

Thanks to the Biohub / Royer Lab organisers for a hard and interesting problem, and to everyone who shared notebooks, weights and forum posts. Our final result was built on public work, credited below.

This is not a "secret sauce" write-up. We finished 224th (private 0.92247, public 0.95648) with a lightly modified public pipeline. What we can offer is a large set of measured results:

- many post-processing and localisation ideas that look good locally do not transfer;

- the reasons, with numbers;

- how our own submissions moved between the public and private LB.

We hope it saves someone time in the next tracking competition.

Note on scores. The leaderboard showed public scores truncated to 3 decimals. Kaggle now exposes 5 decimals for every submission, and we use those here. Some "+0.001" steps we saw during the competition were display artefacts: 0.95498 → 0.95509 showed as 0.954 → 0.955.

## TL;DR

- Final pipeline. The public x138 / Harmonic Fusion V3 notebook (UNet detector + transformer linker → ILP → long post-processing chain) plus three changes. Change 1 is the one that helped on the private LB:
- A V1284 coordinate head refit on our own feature captures. Private 0.91780 → 0.92312 (+0.0053), public +0.0011. All three head refits we submitted beat the public head on private.

- A track-end trim: public +0.0001, private −0.0009.

- Readmit threshold 0.965 → 0.94: public +0.0014, private +0.0003.

- Biggest lesson: in-sample validation misleads. The public detector and linker were trained on (almost) all training movies, so any local validation of post-processing with those checkpoints is in-sample. Example: switching motion relink off looked like +3.8 pp locally and scored −0.001 on the LB.

- What we did about it:
- We trained our own model on one embryo (6bba) and validated on the other (44b6).

- We replayed post-processing changes on those held-out movies with identity gates (bit-exact reproduction of the chain first).

- We judged results against rules written down before each run.

- Measured negatives on held-out movies:
- Division post-processing: every safe-division gate variant ≤ +0.0011; a GT oracle for "early split" twins only +0.0013.

- Raw-image z calibration: per-movie image z offsets do not predict GT offsets (sign accuracy 0.39–0.42).

- Line-fit smoothing: every proposed change lost; the existing smoothing is worth +0.035.

- Track pruning and trimming: negative.

- On the LB only: swapping in a public secondary model lost 0.019 public and 0.015 private.

- Public vs private.
- Across the 11 submissions carrying our refit heads, Spearman(public, private) = 0.26. Differences of a few thousandths on the public LB carried almost no information.

- The unmodified public 0.953 notebook scored lower on private (0.91780) than our older 0.946-public Anchor variant (0.92269).

## 1. Final pipeline

Base: the public x138 notebook by anvithpothula (`biohub-0-953-lb-original`). We forked raunakdey07's Harmonic Fusion V3, which is the same code.

- It comes from the same lineage as the public 0.947 notebooks (e.g. zhehaoliang's "Anchor"), which go back through raykkretzschmar's union / harmonic-fusion notebooks.

- Everything runs on pilkwang's support pack and weights.

| stage | what it does |
| --- | --- |
| Detection | Two UNet + transformer seeds, harmonic forward/reverse association fusion (weight 0.15), dual-seed retention guard 0.90 |
| Coordinate refinement | V1284 head : a small MLP on frozen UNet features that moves each detection by < 2 µm and keeps float coordinates |
| Linking | ILP (tracksdata), motion relink (Hungarian, neighbourhood-flow seed mode), readmission of ILP-discarded detections, gap closing and gap-fill from low-threshold peaks |
| Divisions | add_safe_divisions_postlink : geometric gates (parent/sister distances, both daughters continue to t+2 and separate by ≥ 2.25 µm, mutual-NN, symmetry) plus a DeepCenter veto (threshold 0.25) |
| Clean-up | division geometry filter, isolated-node prune, short-track filter (min length 6), end-trim (added by us), line-fit smoothing (window 2, weight 0.8), integer writer max(0, round(v)) |

### Three changes on top of x138

| submission | public | private |
| --- | --- | --- |
| x138 / HFv3 verbatim | 0.95387 | 0.91780 |
| + V1284 head refit on 44b6 | 0.95498 | 0.92312 |
| + end-trim | 0.95509 | 0.92218 |
| + readmit min score 0.94 ( final pick 1 ) | 0.95648 | 0.92247 |

The head.

- We captured detector features and GT pairs on all 199 training movies (we call this CAP), and refit the head on the 44b6 movies.

- We also submitted an all-199 refit and a 44b6 seed ensemble. On private: 0.92285 (all-199), 0.92312 (44b6), 0.92223 (ensemble). All three beat the public head (0.91780). The differences among them are small.

- Why 44b6? Every head learns its training embryo's z offset (GT minus detection: +0.13 µm on 44b6, +0.63 µm on 6bba). A head fitted on 6bba was 13.8% worse than the public head on 44b6, and the public head behaves like a 44b6-trained head.

The end-trim.

- Rule: drop the terminal node of a track end or start whose attaching edge probability is < 0.70. Never at the first or last frame; applied after the short-track filter.

- Where it came from: a CSV posted in a public Kaggle dataset whose README described it as a high-scoring submission. We compared it with the x138 output and recovered a rule that reproduces its track ends. We never saw or sought any code behind it.

- Our own out-of-sample tests of the rule were negative (section 5.3). On the LB it was +0.0001 public and −0.0009 private.

The readmit change came from John Taylor's forum post.

Final selection.

- Our picks were readmit094 (public 0.95648, private 0.92247) and "head + trim" (0.95509 / 0.92218).

- An earlier draft of our selection kept the head alone as the hedge, because our held-out trim tests were negative. On the last day we swapped it for the trim variant.

- The head alone was our best private submission (0.92312). The held-out evidence was right, and our final choice followed the public LB instead.

### Timeline

| date | public LB | what |
| --- | --- | --- |
| August | 0.912 → 0.929 | forks of older public baselines (section 7) |
| early–mid September | 0.946–0.947 | variants of the public 0.947 Anchor; in-sample attribution, own-model calibration |
| 26 Sep | 0.953 | x138 / Harmonic Fusion V3 adopted |
| 27 Sep | 0.954 → 0.955 | our 44b6 head; + end-trim (displayed values, truncated) |
| 28 Sep | 0.956 | + readmit 0.94 (final best public) |
| 29 Sep | 0.953–0.955 | final-day line-fit arms (all lower) |

## 2. All our late submissions: public vs private

| submission | public | private |
| --- | --- | --- |
| HFv3 + 44b6 head | 0.95498 | 0.92312 |
| HFv3 + all-199 head | 0.95294 | 0.92285 |
| + trim + readmit 0.92 | 0.95564 | 0.92274 |
| Anchor + sub-voxel detector peaks xyz (P32) | 0.94618 | 0.92269 |
| + trim + readmit 0.94, radius 5 µm | 0.95590 | 0.92268 |
| + trim + readmit 0.94 + line-fit weight 0.9 | 0.95524 | 0.92262 |
| + trim + readmit 0.94 (final pick 1) | 0.95648 | 0.92247 |
| HFv3 + 44b6 seed-ensemble head | 0.95184 | 0.92223 |
| HFv3 + 44b6 head + trim (final pick 2) | 0.95509 | 0.92218 |
| + trim + readmit 0.94 + line-fit z weight 1.0 | 0.95475 | 0.92210 |
| + trim + readmit 0.94 + line-fit weight 1.0 | 0.95366 | 0.92193 |
| Anchor + sub-voxel detector peaks xy (P32) | 0.94588 | 0.92152 |
| + trim + ILP division weight 0.4 | 0.95490 | 0.92058 |
| HFv3 / x138 verbatim | 0.95387 | 0.91780 |
| Anchor + z-edge shift +1 slice (P31) | 0.94527 | 0.91432 |
| HFv3 + 44b6 head + a public secondary model swap | 0.93586 | 0.90800 |

("+ trim …" rows are all on top of HFv3 + 44b6 head.)

What this table says:

- Public differences of a few thousandths were noise for selection. Spearman(public, private):
- 0.26 over the 11 submissions with our refit heads;

- 0.31 with x138 verbatim added;

- 0.40 over the 15 rows without the model swap.

- Our best public submission ranks 7th of 15 on private.

- x138 verbatim did worse on private than our older Anchor variant. It is +0.0077 public over P32 xyz but −0.0049 private. This is one pair of submissions, so it is a single data point. But hundreds of teams sat at ≥ 0.953 on the public LB in the last days, many of them on this notebook.

- What held up on private was the learned coordinate head refit. Every post-processing addition on top of it scored below the head alone on private (0.92058–0.92274 vs 0.92312).

## 3. How we validated (the part we would keep)

### 3.1 The in-sample trap

Our first big attribution study (E15) took the 0.947 Anchor on 48 training movies and switched each post-processing stage on and off, scored with the host's original (pre-July-patch) scorer:

| stage | local Δ (in-sample) | LB Δ (measured later) |
| --- | --- | --- |
| motion relink off | +3.77 pp (40 of 48 movies better) | −0.001 (0.945 vs 0.946) |
| safe-division off | −1.0 pp | −0.035 (0.911) |
| learned relink bonus off | −0.42 pp | 0.000 |

Why it misleads. The public detector and linker were trained on essentially all 199 training movies. On those movies the ILP probabilities are near-oracle, so geometric overrides look harmful locally and neutral or helpful on unseen embryos. Other competitors reported the same pattern.

It also works in the other direction: line-fit smoothing measured +0.67 pp in-sample (E15) and +0.035 on held-out movies (section 5.2). In-sample proxies can understate as well as inflate.

### 3.2 Our own out-of-sample model

- TRAIN6: the official UNet + transformer architecture trained on 6bba only (6 epochs), scored on held-out 44b6 movies.
- The chain hurts movies whose raw graph is already good.

- The best raw-edge cut-off moved from 0.05 (epoch 0) to 0.25 (epoch 5).

- Calibration against the LB (S6): we submitted this model through the exact same chain and writer.

| change | held-out CV Δ (6 movies) | LB Δ |
| --- | --- | --- |
| whole chain on vs off | +0.048 | +0.050 |
| epoch 0 → 5 (model quality) | +0.069 | +0.043 |
| relink on vs off (epoch 0) | +0.080 | +0.074 |
| relink on vs off (epoch 5) | −0.018 | 0.000 |

- The LB level was about held-out CV + 0.10.

- Rule of thumb we used afterwards:
- Chain-level held-out deltas of ≈ 0.05 or more transferred about 1:1. Model quality transferred at about 60%.

- Deltas around 0.02 on 6 movies did not transfer.

- In-sample proxies inflated small gains about 4× in two cases: a public sweep proxy +0.004 → LB +0.001, and visible-test scoring +0.0022 → LB +0.001. For the coordinate head, in-sample error reduction was 10–20× the out-of-sample one.

### 3.3 Identity-gated replays

For each question we re-ran the TRAIN6 model and the Anchor chain on held-out 44b6 movies. That was 6 movies for the first diagnostics (D1–D4, T6H, E1), 21 for the division replays (all 44b6 held-out movies with GT divisions), and 24 for the line-fit replay.

Each replay:

- captured the exact graph entering the stage under study;

- re-ran that stage under each variant, then the CSV writer's rounding;

- scored edges with the host scorer (edge numbers are unaffected by the July patch for our graphs) and divisions with the current (2026-07-17) directed division metric. Our support-pack scorer predates that patch and is more permissive for divisions.

Fail-closed rules. A replay counted only if:

- our copy of the stage reproduced the chain's output bit for bit (identity A);

- the rebuilt graph equalled the saved chain output exactly (identity B);

- coverage was complete and the host aggregation check matched.

Otherwise the result was labelled UNRELIABLE. The pass rule (e.g. "Δ ≥ +0.002 and ≥ 2/3 of movies non-worse") was written before each run.

Code review. We used LLM-assisted code review before every Kaggle run. It caught real bugs, for example a capture wrapper that the notebook's own `s6_write()` would have silently overwritten, giving zero captures.

## 4. Metric and data facts that mattered

- Score = adjusted edge J + 0.1 × division J.
- Adjusted J = J · (1 − 0.1 · (N_pred − N_est)/N_est), per movie, weighted by TP + FP + FN.

- The bonus for predicting fewer nodes than N_est is not capped. In practice the x138 output sits near N_pred/N_est ≈ 1, so node-count games do not pay.

- Matching: nodes match within 7 µm (voxel z, y, x = 1.625, 0.40625, 0.40625 µm). Edges must go t → t+1. One z slice is 23% of the matching radius.

- GT is sparse: ~151 division events in 199 training movies (Luka Duvanov's count).

- The two training embryos differ. On the forum, Ace reported annotation density ~1% vs ~9% and division counts 26 vs 125. This is part of why anything learned on one embryo (heads, offsets) transfers poorly.

- Metric history. Early 0.963–0.966 public scores exploited a weak-component division rule. This was patched on 2026-07-17 and rescored (megayak and lepus2 documented it). The widely forked offline division metric still uses the old rule and reads about 2× the official value. Use the official `division_metrics` code.

- The public LB was ~29% of the test, and the embryo mix of each split was unknown.

- Hidden-rerun behaviour:
- x138 skips motion relink for a movie if any frame has > 2,600 nodes, and after a 7.5 h repair deadline. Raising that deadline to 10 h scored 0.953, so it was not binding.

- The public 0.948 Geometric Fusion notebook timed out on the hidden rerun.

- A notebook that crashed after writing `submission.csv` was still scored.

- Line-fit can emit z = 64, past the 64-slice volume. The scorer does not reject it.

## 5. What we measured: post-processing

### 5.1 Divisions: the post-processing lever is closed

On held-out 44b6 (21 movies, 26 GT divisions) the TRAIN6 chain ends at 4 TP / 10 FP / 22 FN (division J 0.11). Forum reports and our safe-division ablation suggest the LB division J is about 0.2–0.35.

Where divisions are lost (D6):

| stage | TP of 26 |
| --- | --- |
| pre-ILP candidates | 12 with a strict direct fork (one node linked to both daughters). The looser "23" of the old metric is misleading: the fully forked candidate graph scores only 5 TP under the current metric |
| after ILP | 0 |
| after the full chain | 4 |

The ILP never forks, and relink drops ILP edges anyway. At division weight 1.2 a second child costs `−p + 1.2 > 0`. Then motion relink rebuilds the edges one-to-one, as zhincez (Lê Quang Cảnh) and Luka Duvanov pointed out on the forum. So, wherever relink runs, every final fork comes from `add_safe_divisions_postlink`. Consequences:

- The ILP division weight acts only indirectly. Our div 0.4 arm: public −0.0002, private −0.0016.

- The safe-division caps never bound in our visible runs (`cap_skipped = 0`). So re-ranking proposals changes nothing, except when two sources claim the same orphan.

Safe-division gate replay (D6C, reliable). DeepCenter was trained on 44b6, so its numbers here are in-sample and optimistic:

| variant | Δ score | div TP / FP / FN | Δ edge TP |
| --- | --- | --- | --- |
| Anchor base (DeepCenter 0.20) | 0 | 4 / 10 / 22 | 0 |
| DeepCenter 0.15 / 0.25 / 0.30 | −0.0004 / +0.0002 / +0.0011 | 4/11/22, 4/9/22, 4/6/22 | 0 / −1 / −5 |
| mutual-NN k ≤ 2, mutual-NN off, t+2 escape | 0.0000 | 4 / 10 / 22 | 0 |
| soft division (missing t+2 allowed) | −0.0006 | 4 / 11 / 22 | 0 |
| GT oracle : attach "early split" twin tracks to the mother | +0.0013 | 5 / 13 / 21 | 0 (edge J down in all 4 movies) |

- Early-split twins. In 11 of the 26 GT divisions, the second daughter already has a parent, typically an unannotated node right next to it at the mother's frame. The detector splits the mother one frame early, and the second half runs as its own track.

- Even a GT-informed oracle that attaches these tracks gains only +0.0013. 7 of the 11 cannot be attached, and the attached ones mostly become division FP under the directed metric.

- On the LB:
- safe-division off cost −0.035 (0.911), so it is load-bearing;

- requiring learned support removed 55 of 124 forks and scored 0.938;

- adding transformer-supported forks scored 0.947, which is noise.

- Others reported the same pattern: vitalops' orphan-division admission scored 0.927 / 0.940. megayak showed that only ~35 of 151 GT divisions survive the public gates, and that opening the gates lowers the score because the tightest pairs are duplicates. Division is a ranking problem, not a recall problem.

- Value per fork. On average a fork is worth about +0.0003 LB (from the −0.035 / 124 forks of the safe-division ablation). With LB division J of 0.2–0.35, a new fork must be a GT hit roughly J/(1+J) ≈ 18–26% of the time to break even.

### 5.2 Line-fit smoothing: worth +0.035, and already near its optimum

The chain ends with `linefit_smooth_output_graph`: a linear fit over a ±2-frame window, blended at weight 0.8, before integer rounding. Several research answers proposed "fixing" it:

- it walks across forks, so a daughter borrows the mother's positions;

- it fits one-sided, extrapolating windows at track ends.

We tested every proposal on the exact pre-linefit graph of 24 held-out movies (D6L, reliable, bit-exact capture). The replay uses the Anchor chain, which has no readmit and no end-trim, so the end-policy variants are indicative for production.

| variant | Δ score | Δ edge TP | movies non-worse |
| --- | --- | --- | --- |
| current (window 2, weight 0.8) | 0 | 0 | — |
| fork-safe walk | −0.0001 | −1 | 23 / 24 |
| fork-safe, daughter start unchanged | −0.0004 | −2 | 22 / 24 |
| strict interiors only, both sides required | −0.0047 | −18 | 15 / 24 |
| z weight 0.4 (y/x 0.8) | −0.0101 | −35 | 13 / 24 |
| clean ±2 → ±1 → unchanged, revert if moved > 1 µm | −0.0318 | −133 | 3 / 24 |
| line-fit off | −0.0347 | −144 | 4 / 24 |

- Attribution (current linefit applied to one node class at a time; marginal, not additive):
- full ±2 interiors: +0.031;

- partial windows: +0.0025;

- track ends: +0.0011;

- fork-crossing windows: +0.0008 (crossing forks slightly helps);

- track starts: ≈ 0.

- More smoothing. The held-out trend over z weight 0 → 0.4 → 0.8 was still rising, so we tried more smoothing on the LB:
- weight 0.9 / z weight 1.0 / weight 1.0 gave public 0.95524 / 0.95475 / 0.95366 and private 0.92262 / 0.92210 / 0.92193;

- weight 0.8 gave 0.95648 / 0.92247.

- These differences are small, but none beat 0.8 on public, and only weight 0.9 matched it on private (+0.0002). A ±3 window was never submitted.

### 5.3 Node trimming and pruning

| change | held-out (TRAIN6, 6 movies) | LB |
| --- | --- | --- |
| prune short tracks (min length 8) / low-probability tracks (0.7) | −0.006 / −0.043 | — |
| our first edge-probability end-trim (τ 0.65, prob-0 and boundary ends skipped) | all 84 settings negative | — |
| the trim rule as used (τ 0.70, prob-0 ends included, first/last frame protected) | all 36 settings negative (−0.002 to −0.007) | public +0.0001, private −0.0009 |

Weak-edge track ends are 2.5× less often annotated than an average node, but break-even needs 6–8×. One lost true edge costs as much as the node-count bonus of about a thousand trimmed nodes. The visible-test score (in-sample) liked the trim (+0.0022). Held-out and private did not.

### 5.4 Knobs: all within ±0.002

| knob | Δ public | Δ private |
| --- | --- | --- |
| readmit min score 0.965 → 0.94 | +0.0014 | +0.0003 |
| readmit 0.94 → 0.92 | −0.0008 | +0.0003 |
| readmit radius 4 → 5 µm | −0.0006 | +0.0002 |
| ILP division weight 1.2 → 0.4 | −0.0002 | −0.0016 |
| Anchor: relink off, learned bonus 0 / 2, leaf prune, velocity, public safe-division settings | 0 to ±0.001 | — |

The same saturation was reported on the 0.947 and 0.953 stacks:

- imissher, vitalops, sjlee101 and thtennant all reported knob changes of ±0.001;

- Justin tried 10 single knobs and all lost on the LB; his best offline change (+0.0011) scored −0.005.

## 6. What we measured: localisation (where most errors are)

### 6.1 The error anatomy (D1, D3)

We used Soheil Ayati's split of missed edges into missing endpoints vs wrong associations. On held-out 44b6:

- ~60% of missed edges have a missing endpoint, not a wrong link.

- The unmatched GT node almost always has a detection just outside the 7 µm radius (median 7.8 µm).

- 88–90% of missed edges originate from misplaced nodes.

- z-edge pile-up:
- 38% of raw misses sit at z ≥ 56 (31–34% after the chain), against 15% of cells. There, the prediction is about 3 slices below the GT in 89% of cases.

- Our detector (and the Anchor's) almost never predicts the top or bottom slice (z = 63 or z = 0).

- Oracles: a ≤ 2 µm nudge would rescue all 38 far nodes of our CV set. The localisation oracle is worth +0.056 to +0.110 and the association oracle +0.036 to +0.068. The problem is predicting the direction.

### 6.2 Heuristic relocations: none transferred

| method | held-out | LB |
| --- | --- | --- |
| centre-of-mass relocation (D3) | node recall 0.964 → 0.974; raw score up, but the score after the chain went down | — |
| sub-voxel detector peaks (D4) | raw +0.0086; after the chain −0.0016 to +0.0055 with ±0.035 per-movie swings | 0.945 / 0.946 (base 0.946) |
| z-edge shift +1 slice (D2) | +0.008, but 77% from one movie | 0.945 |

Eric reported the same for sub-voxel peaks: +0.0078 locally, −0.002 on the LB, twice. Matching more nodes also exposes more wrong edges.

### 6.3 The V1284 head: why it works, and why it is fragile

- Out of sample it helps. TRAIN6 heads on held-out 44b6 gave +0.004 to +0.011 on the chain, which is only a few edges (+2 to +13 TP of 1,301).

- Mechanism: boundary rescue. Of the 38 detections just beyond 7 µm, the heads bring 10–11 inside and push 3–5 out. A random-direction control gives ≈ 0.

- Most of the rescue rides on z. But z moves make already-matched nodes worse 58–61% of the time.

- The out-of-sample rescue came from the 6bba head's upward z push, which happened to point toward GT for far 44b6 nodes. Centring the head removed it.

- Judge heads by boundary rescue, not centre error. In-sample error reduction (−34% / −57%) shrank to about −3% out of sample.

- Failed variants:
- heads trained with far pairs included rescued only marginally more (net +5.7 / +6.3 vs +5.0; the pre-set bar was +8);

- seeds alone varied the net rescue from +2 to +7;

- the in-sample boundary effect even flipped sign between capture shards (+19 vs −6).

- Fixes that failed both cross-embryo checks: centring, standardisation, blending, α scaling and test-time normalisation.

### 6.4 Image-based z self-calibration (closed)

Two of four deep-research answers ranked "estimate each test movie's z offset from its own raw image" first; the other two ranked it third. We tested the premise on the captured detection–GT pairs of all 199 training movies (CAPZ, reliable).

- Scope: a signal screen on detector positions before the head, not a final-score test.

- Estimators: axial quadratic peak, centre of mass over ±2 slices, axial Gaussian.

| estimator | sign accuracy (per movie) | Spearman ρ | ρ within 44b6 | ρ within 6bba |
| --- | --- | --- | --- | --- |
| quadratic | 0.42 | 0.10 | −0.15 | 0.22 |
| centre of mass | 0.39 | 0.08 | 0.01 | 0.28 |
| axial Gaussian | 0.42 | 0.07 | −0.05 | 0.23 |

- The image offset does not predict the GT offset. Its sign is right less often than chance.

- The image-to-GT bias differs by embryo: −0.11 µm on 44b6 vs −0.84 µm on 6bba. Our reading, which is an inference: the image centre does not relate to the annotation in the same way in both embryos.

- Per-node image moves agree in sign 52–57% of the time and push more boundary nodes out than in.

Together with section 6.3, this is our best explanation for why localisation was so hard to improve from the public pipeline without new training.

## 7. Other things we tried

- Our own models. TRAIN6 at epochs 0–5 scored 0.778–0.895 on the LB, and bhpepper's synthetic 5-fold SWA package scored 0.859. The public weights were trained on all movies for far longer, and we did not have the compute to match them.

- Linker retraining (12 checkpoints): no broad gain.

- A public secondary model swapped into the tuned chain: public 0.93586, private 0.90800. The x138 fusion and thresholds are tuned to pilkwang's secondary.

- Public detector v1327-w3 (not submitted): its public notebooks scored 0.907–0.942. One team reported a jump from 0.948 to 0.957 with a private pipeline that used it, so it is not a demonstrated negative.

- External data.
- Public fine-tuning attempts of the public weights with Ultrack / Zebrahub pseudo-labels or synthetic data showed no LB gain, so we did not pursue them.

- This does not apply to own-model pipelines: the top teams (Sergio uses an external Ultrack embryo for validation; Tang recommends retraining with external or synthetic data) used them differently.

- Crowding does not raise the error rate (FN ratio 0.96, CI 0.41–1.90).

- The LB change of our variants did not follow their visible node count (Spearman −0.02 over 8 variants).

- August iterations, on older public baselines (0.912 → 0.929):
- harmonic fusion +0.003;

- retention guard +0.001;

- safe-division divergence gate +0.007;

- searched divisions +0.004 (only +0.001 later on the Anchor).

- All of these ideas are now part of the public x138 stack.

## 8. What we would do differently

- Build the out-of-sample harness first. We spent the first weeks optimising post-processing against in-sample CV. The first step should be a model trained on one embryo and validated on the other, run through the production chain.

- Spend compute on the detector and the linker, not on post-processing. The held-out oracles put the room in localisation (+0.056 to +0.110) and association (+0.036 to +0.068), while the post-processing levers were saturated. Tang's public advice was the same: detection first, then linking, then division.

- Do not select final submissions by public differences of a few thousandths. Our held-out trim tests were negative and our head change had out-of-sample support. Following that evidence instead of the public LB would have kept our best private submission.

- Treat the metric version as part of the experiment. Our local scorer predated the July division patch. Division numbers measured with it were too optimistic until we embedded the current official code.

## Acknowledgements

Code, weights and notebooks:

- pilkwang: the support pack, the detector and linker weights, and DeepCenter.

- raykkretzschmar: the union / harmonic-fusion lineage.

- zhehaoliang: the 0.947 Anchor.

- amanatar: Geometric Fusion and the optimised notebook.

- anvithpothula: x138 and the V1284 head.

- raunakdey07: Harmonic Fusion V3.

- bhpepper: the synthetic SWA package.

Forum findings that shaped our experiments:

- John Taylor: the readmit and division-weight hints.

- zhincez (Lê Quang Cảnh) and Luka Duvanov: relink and GT statistics.

- Soheil Ayati: the missed-edge split.

- megayak: division ranking and the metric history.

- Tang and Sergio Alvarez: training and validation advice.

- Eric, Justin, imissher, vitalops, sjlee101, thtennant, Jin-Ruoting, hikaggler, Mark_RowSet and Ace: measured results we calibrated against.

The organisers, for the dataset and the patched metric.

## Team Hack2Publish

| member | Kaggle | GitHub |
| --- | --- | --- |
| Foysal | foysalemonshanto | Foysal348 |
| Md. Hamid Hosen | hosen42 | hamidhosen42 |
| rokaiyasomapti | rokaiyasomapti | — |
| esfersami50 | esfersami50 | — |

All numbers come from our own submissions and held-out runs unless attributed. Held-out results use our TRAIN6 model and the Anchor chain, not the exact production chain, so treat them as directional for production.
