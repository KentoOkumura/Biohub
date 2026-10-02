# 181st Place Solution: Retraining the Coordinate-Refinement Head

- archived_at: 2026-10-02T11:53:16.833246+00:00
- source: https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/744762
- author: Ryota Kawamura
- posted_at_utc: 2026-09-30 22:55:10
- votes: 1
- comments: 0

Thanks to Biohub and Kaggle for a hard and beautiful dataset. This solution is built directly on public work: anvithpothula (x138 pipeline and the V1284 coordinate-refinement head), kunaldesale2408 (the 0.953 notebook I forked) and pilkwang (TemporalUNet3D / DeepCenter weights and the support pack). Thank you for sharing.

## TL;DR

- I joined 9 days before the deadline, so I did not rebuild the pipeline. I kept the public 0.953 notebook and changed one small component: the head that nudges each detected cell centre by up to 2 µm.

- The public head was trained on 20 movies. I retrained it on all 199 training movies (111,129 detection–ground-truth pairs), validated by held-out movie: mean centre error 1.33 → 1.06 µm.

- Public LB: tied at 0.953. Private: 0.917 → 0.924 (+0.007), 181st of 3,947.

| Submission | Public | Private | Selected |
| --- | --- | --- | --- |
| public 0.953 notebook | 0.953 | 0.917 | ✅ |
| + retrained head | 0.953 | 0.924 | ✅ final |
| + retrained head + linking settings | 0.953 | 0.929 | — |

## The pipeline on real data

To make the change concrete, here is one training frame run through the pipeline, with the ground truth.

![The pipeline on one training frame](https://raw.githubusercontent.com/Ryota-Kawamura/biohub-cell-tracking-during-development/main/docs/images/real_pipeline.png)

- (a) Input. Each movie is a 3D + time volume of a zebrafish embryo, shown here as a max-projection over z.

- (b) Detection. The public TemporalUNet3D puts one centre on each nucleus. Only a few cells carry ground truth, and the metric only looks around them.

- (c) Linking. The public linker (candidate edges, ILP, relink, gap closing) turns centres into tracks. I left it untouched.

- (d) Centre refinement, the only stage I changed. The head reads the detector's 224-dim U-Net feature at each centre and predicts a 3D shift (arrows). The crops are the six annotated cells with the largest detector error in this frame, chosen by error, not by outcome: in cell 3 the public head does better. The retrained head shown is from the held-out fold, so this movie was not used to train it.

## 1 · Why this component

![pipeline](https://raw.githubusercontent.com/Ryota-Kawamura/biohub-cell-tracking-during-development/main/docs/images/pipeline.png)

Retraining the detector was not realistic in 9 days. The refinement head, by contrast, is a 7k-parameter MLP on frozen features: cheap to retrain on CPU, and it had seen only a tenth of the training movies.

## 2 · Data and validation

- Capture (2 GPU notebooks): run the public detector over all 199 training movies and save each detected centre with its 224-dim feature.

- Pairs: Hungarian matching to ground truth within 3 µm (voxels are anisotropic: z = 1.625, y = x = 0.40625 µm) → 111,129 pairs. Target = ground truth − detection.

- Validation: 4 folds by movie, so no movie is in both train and validation. The public LB could not resolve a gain this small, so every decision was made on this number.

## 3 · The head

Same architecture as the public head: `Linear(224, 32) → SiLU → Linear(32, 3)`, output bounded by `2d / (1 + ‖d‖)`, last layer zero-initialised. AdamW (lr 3e-3, wd 1e-4), smooth L1 (β = 0.5), batch 1024, 300 epochs; CPU only.

![held-out error per movie and per detection](https://raw.githubusercontent.com/Ryota-Kawamura/biohub-cell-tracking-during-development/main/docs/images/heldout_errors.png)

Held out by movie, the mean centre error goes 1.46 µm (detector only) → 1.33 µm (public head) → 1.06 µm (retrained head), and the retrained head is better on 158 of 199 movies. The whole error distribution moves left, not just a few outliers.

![one cell over five frames](https://raw.githubusercontent.com/Ryota-Kawamura/biohub-cell-tracking-during-development/main/docs/images/cell_over_time.png)

Scaling the learned shift after training (×0.8 to ×2.0) or averaging 5 seeds only lost score on both leaderboards: the trained scale was already right.

### The bug that cost a day

![training targets in two coordinate frames](https://raw.githubusercontent.com/Ryota-Kawamura/biohub-cell-tracking-during-development/main/docs/images/target_frame.png)

My first retrained head scored lower (0.943 public). I had built the targets in the voxel-centre frame, but the notebook writes nodes at `index × 4` with no offset. The two frames differ by 1.5 full-resolution pixels (0.61 µm) in y and x, so the head learned a constant shift that was wrong for every cell. Rebuilding the targets in the output frame fixed it: 0.953 public, 0.924 private.

## 4 · Results

![public vs private](https://raw.githubusercontent.com/Ryota-Kawamura/biohub-cell-tracking-during-development/main/docs/images/public_vs_private.png)

Seven of my submissions tied at 0.953 public. I picked the one backed by held-out evidence, and on private they spread from 0.917 to 0.929. The best (0.929) added linking settings (relink velocity 0.25, division gates 11 / 16 / 12 µm, DeepCenter safe-division 0.15) that the public LB gave me no reason to choose.

## 5 · What did not work

| Idea | Result |
| --- | --- |
| Single-setting tuning (detection threshold, ILP weights, relink bonus, division gates) | 19 variants, none above its base on public |
| Shift scale ×0.8 / ×1.25 / ×1.5 / ×2.0 | private 0.923 / 0.923 / 0.919 / 0.892 (vs 0.924) |
| 5-seed head ensemble | private 0.924, no gain |
| Head trained in the voxel-centre frame | 0.943 public (the bug above) |
| Unmodified 0.948 public notebook | ~4 h on the public test, over 12 h on the hidden test: timed out |

## 6 · Takeaways

- Measure small components directly. 0.27 µm was invisible on the public LB and worth +0.007 on private.

- When public scores tie, choose by held-out evidence.

- Check the coordinate convention end to end before trusting an offline gain.

## Project Links

- Code, figures and experiment log: [GitHub](https://github.com/Ryota-Kawamura/biohub-cell-tracking-during-development)

- Notebooks, in order: [biohub-cap-a](https://www.kaggle.com/code/ryotakawamura94/biohub-cap-a) + [biohub-cap-b](https://www.kaggle.com/code/ryotakawamura94/biohub-cap-b) (capture) → [biohub-headtrain](https://www.kaggle.com/code/ryotakawamura94/biohub-headtrain) (retrain the head) → [biohub-b11-khead4](https://www.kaggle.com/code/ryotakawamura94/biohub-b11-khead4) (final submission)
