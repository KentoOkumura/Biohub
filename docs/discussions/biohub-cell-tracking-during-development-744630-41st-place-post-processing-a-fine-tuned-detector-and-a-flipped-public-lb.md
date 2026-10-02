# 41st place: post-processing, a fine-tuned detector, and a flipped public LB

- archived_at: 2026-10-02T11:53:16.152861+00:00
- source: https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/744630
- author: datnt114
- posted_at_utc: 2026-09-30 16:20:32.660000
- votes: 4
- comments: 0

Thanks to Biohub and Kaggle for a hard and interesting competition.

Most of all, thanks to the people who shared public pipelines: pilkwang (DeepCenter / TemporalUNet3D weights and support pack), thtennant (frontier947 gapfill), anvithpothula (x138 + V1284 coordinate-refinement head), and John Taylor for the discussion on tuning knobs against the public LB. My solution is built directly on top of their work.

## TL;DR

- Base: the public x138 notebook (0.953 public). I did not rewrite the pipeline. Every change is an env-flag patch, so each one can be measured on its own.

- Post-proc 1: raise `MIN_TRACK_LEN` from 6 to 8.

- Post-proc 2: a division classifier (mean of a small CNN3D and a ConvNeXt-nano) adds a missing second daughter. It only attaches daughters that are free (have no parent yet).

- Detector fine-tuning with intensity augmentation, ending in a weight soup.

- The public LB had only 4 clips and ranked my submissions in the opposite order to private. My local validation had the right direction.

| Submission | Public | Private | Selected |
| --- | --- | --- | --- |
| x138 + ML8 + fine-tuned detector (no-rot primary ep18 + secondary ep13) + division ens | 0.946 | 0.938 | no |
| x138 + ML8 + fine-tuned detector soup (2 seeds) + secondary ep13 + division ens | 0.944 | 0.936 | ✅ (final) |
| x138 + ML8 + division ens | 0.956 | 0.931 | ✅ |

How each post-processing step moved the score:

| Step | Public | Private |
| --- | --- | --- |
| x138 (public notebook) | 0.953 | 0.917 |
| + MIN_TRACK_LEN 8 | 0.953 | 0.917 |
| + division CNN3D | 0.955 | 0.928 |
| + division ens (CNN3D + ConvNeXt) | 0.956 | 0.931 |

The division step added +0.003 on public but +0.014 on private.

![pipeline](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F2730329%2Fc311f3150417f9fc1932e2dec74dc19a%2Ffig1_pipeline.png?generation=1790784999429900&alt=media)

## The pipeline on real data

To make each step concrete, here is one training video (`6bba_784a78c9`, t=25) run through the pipeline. It shows the predictions after each stage, together with the ground truth.

![steps](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F2730329%2Fcd8265c126234c43774dafb414705b57%2Ffig2_steps_on_data.png?generation=1790785011174823&alt=media)

- (a) Input. Each video is a 3D+t volume with shape (T, Z, Y, X) = (100, 64, 256, 256). The images are shown as max-projections over Z.

- (b) Detection. The detector puts one center on each nucleus. Note that the ground truth labels only a small subset of cells (16 of about 155 at this frame). The metric is computed only around the labelled cells.

- (c) Linking. The public linker (node transformer + ILP + relink/gapfill) is already strong. I left it untouched.

- (d) Post-proc 1. Tracks of length 6–7 are mostly false positives, often at the image border. Raising `MIN_TRACK_LEN` to 8 removes 16 such tracks in this video.

- (e)–(f) Post-proc 2. The ILP linked the mother to only one daughter. The division classifier flags the mother, and a nearby free node at t+1 becomes the second daughter. Both predicted daughters then match the ground truth (green circles).

## Local validation

Every public checkpoint was trained on all 199 training videos (2 embryos, 44b6 and 6bba), so there is no clean holdout. I used two sets anyway:

- val24: 24 videos, used for edge quality. The full x138 pipeline runs locally with a prediction cache and is scored with the official royerlab metric (`adjusted_edge_jaccard + 0.1 * division_jaccard`).

- div59: 59 videos with 93 GT divisions, used for anything that touches divisions. The division classifiers are always scored on the embryo they were not trained on, so this part is a real out-of-embryo test. Its sign agreed with the LB on every post-processing knob I submitted.

I always compared results per video. A +0.003 gain that wins on 22 of 24 videos is real. A +0.003 gain that comes from a single video is not.

## Post-proc 1: MIN_TRACK_LEN 6 → 8

![min track len](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F2730329%2Fa641caa1730018337d3b5df4833826a2%2Ffig4_min_track_len.png?generation=1790785030044827&alt=media)

- The change gives +0.003 on val24 and improves 22 of 24 videos.

- Values 9 and 10 scored slightly higher, but by less than +0.0006, so I stayed with 8. At 15, real tracks start to disappear.

- Public LB: 0.953 → 0.953 (no change).

## Post-proc 2: division classifier

On div59 the base pipeline finds only 14 of 93 GT divisions. The ILP almost never produces a fork, so most of the division term was left on the table.

Classifier input. For each node, I take frames t-1, t and t+1 around it, max-project z±3 slices, crop 64×64 around the node, and stack the three frames as RGB. The patch is normalised by the (1, 99.9) percentiles of frame t. A dividing nucleus appears as one blob at t that splits into two at t+1.

![classifier input](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F2730329%2F73f4fa227632e0250163fe36ed0fdac7%2Ffig3_mitosis_input.png?generation=1790785020096397&alt=media)

Models.

- CNN3D: a small 3D CNN with 3 Conv-BN-SiLU blocks, trained for 25 epochs with class-balanced batches.

- ConvNeXt-nano (`convnext_nano.in12k_ft_in1k`) on the RGB patch resized to 128, trained with AdamW at lr 3e-4 for 8 epochs.

- DINOv2-small: tried as well.

- Training data: positives are GT nodes with 2 children (111 in train). Negatives are 80 single-child nodes per video.

- CV: leave-one-embryo-out.

Rule for adding a division.

- Score every node that currently has exactly one child.

- If the score is ≥ 0.3, attach the nearest free node at t+1 (a node with no parent) as the second daughter.

- Cap the number of added forks at 0.15% of all nodes.

The step runs on the final `submission.csv`, so it never touches the detector or the linker.

![division](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F2730329%2Fa847e9c7a53cbaa81e6ee78b7aa41f58%2Ffig5_division.png?generation=1790785040587889&alt=media)

- Stealing a daughter that already has a parent raises division TP (26 → 32) but breaks correct edges: edge Jaccard drops from 0.9043 to 0.9031 on div59, and val24 falls from 0.9401 to 0.9325.

- No single model wins both CV folds, so I used a mean ensemble. It was the most stable choice on div59.

- Public LB: 0.953 → 0.955 with CNN3D alone, then 0.956 with the ensemble.

- The division gain on div59 transferred to public at only about a third of its size.

## Detector fine-tuning

- Start: the public 400-epoch checkpoints, for both the primary model (support pack) and the secondary model (seed314159, which carries 80% of the detection weight).

- Optimiser: AdamW, lr 3e-5 with cosine decay to 3e-6, batch size 8, 30 epochs, with a snapshot saved every epoch.

- New augmentation, added on top of the public brightness + flip:
- Gamma in [0.7, 1.4].

- Contrast ×[0.75, 1.3].

- Gaussian noise with σ ≤ 0.03.

- Optionally, rot90 in the YX plane.

- Embryo-fold check first: train on one embryo and evaluate on the other. Intensity + rot90 augmentation improved detector acc×recall by +0.004 and +0.008 on the unseen embryo.

- Full runs: trained on rented Vast.ai GPUs. A separate machine scored every checkpoint through the full val24 pipeline.

![finetune](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F2730329%2F0354cb5a48dfea0f3fe569a342e2d671%2Ffig6_finetune.png?generation=1790785051103836&alt=media)

What I learned from the checkpoint sweep:

- A single fine-tuned model is worse than the public weights inside the pipeline. The pipeline's thresholds were tuned for the public pair of models, so the gain only appears when the primary and secondary models are replaced together.

- rot90 helps the detector on its own but hurts linking. So the primary model is trained without rotation, and only the secondary model uses the full augmentation.

- A weight soup of two no-rot seeds (s1 ep18 + s2 ep13) gave the best local edge score, 0.9340 vs 0.9299.

## Public vs private

![lb](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F2730329%2F2e63146d6f0437232ce8b00b2ee058b9%2Ffig7_public_vs_private.png?generation=1790785060922769&alt=media)

- On public, the fine-tuned detector lost 0.010–0.012. On private, it won 0.005–0.007. Local val24 had already shown +0.003 to +0.004 on edges.

- I nearly dropped both fine-tuned submissions because of the public LB. In the end I kept one as a hedge, and that choice was worth about 0.005 on private.

- Local validation did get the order of the two fine-tuned variants wrong: soup ranked above n18 locally, but n18 was better on private. That gap is 0.002, which is within the noise.

## What did not work

| Idea | Result (local) |
| --- | --- |
| Stealing a daughter that already has a parent | val24 0.9325 vs 0.9401 (same thr 0.5, free-only); div59 flat, edges worse |
| Vetoing low-confidence forks from the ILP | div59 0.9149–0.9158 vs 0.9162 |
| Fixing "jumpy" frames (phase correlation + relink) | div59 0.9151–0.9164 vs 0.9162 |
| Primary detector fine-tuned with rot90 | val24 0.9193 vs 0.9376 |
| ± sweep around the final post-proc (thr 0.25/0.35, cap 0.10/0.20%, ML 7/9) | all within ±0.0005, so it was already at the peak |
| Older attempts (Jul–Aug): higher resolution, NMS 3.0 µm copied from another fork, capping nodes at k×N_est, pseudo-labels, symmetric TTA, filtering edges by detection score | all neutral or negative |

Error analysis: 68% of edge errors are wrong links, not missing nodes. Most division errors are a daughter linked to the wrong mother. A better linker would have been worth more than a better detector, but I ran out of time.

## Lessons

- A 4-clip public LB is noise at the 0.01 level. An in-sample local set with per-video comparison was a better guide.

- When local validation and a tiny public LB disagree, keep at least one final pick that is the best submission locally.

- Measure knobs on the set that matches what they change: val24 for edges, div59 (cross-embryo) for divisions.

- Start training early. Fine-tuning started only in the last 4 days. With a clean cross-embryo split from week 1, I would have trusted it sooner and pushed further.

## Tools

- Hardware: 1× RTX 3090 locally for the harness, the division classifiers and the prediction cache. Vast.ai GPUs for detector fine-tuning and checkpoint scoring.

- Libraries: PyTorch, timm, zarr/geff, scipy, pandas and `uv`.

- Metric: the official royerlab metric repo.

- Kaggle CLI: `build_kernel.py` patches the x138 notebook and keeps its original `docker_image`, because newer images break pydantic.

- Claude Code: used to diff public kernels, write patches and eval scripts, and orchestrate the Vast machines.

Thanks for reading, and congrats to the winners!
