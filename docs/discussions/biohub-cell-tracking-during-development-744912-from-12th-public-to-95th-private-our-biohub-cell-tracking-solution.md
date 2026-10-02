# From 12th Public to 95th Private: Our Biohub Cell Tracking Solution

- archived_at: 2026-10-02T11:53:18.919169+00:00
- source: https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/744912
- author: Chekhlov Dmitrii
- posted_at_utc: 2026-10-01 11:27:17.795000
- votes: 17
- comments: 0

Thanks to the organizers and to everyone who shared code and discussed the competition, especially [Pilkwang Kim](https://www.kaggle.com/code/pilkwang/biohub-cell-tracking-learned-graph-w-gap-recovery). We built on the organizer and Pilkwang tracking pipeline, then trained additional models and refined its lineage graph.

## Validation and training

The training videos label only some cells, so treating every unannotated location as background creates false negatives. We began with the sparsely labeled `6bba` embryo and an 80/20 video split (`seed=314159`). We checked the [competition metric](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/overview/evaluation), adjusted edge Jaccard plus `0.1 ×` division Jaccard, every five epochs. Detection loss largely converged by epoch 90. Freezing the backbone and node head, then training the edge head to about epoch 170, improved our development score by roughly `0.004`.

We used the epoch-160 model (`0.861` locally) to find likely unannotated cells. For each prediction that was not a ground-truth label, we excluded its center and one-voxel neighborhood (`3×3×3`) from the detection loss. We masked these predictions rather than turning them into positive labels. This pseudo-unlabeling let us raise the negative loss weight from `0.01` to `0.1`; our training runs improved in score and needed roughly one tenth of the training time.

The final ensemble also drew on models trained with more data. The `a0all` model received 35 joint-training epochs on all 199 labeled videos, followed by 30 epochs of frozen-backbone edge-head training on 173. For slot 2, we fine-tuned `ep40c` for eight epochs on 126 `6bba` videos, keeping BatchNorm fixed and using learning rate `1e-5`, then averaged weights from epochs 3–8. We used the same fine-tuning and weight-averaging procedure for the `a0all` slot-4 model.

## Final inference

The selected `biohub-slot4-xyz2` entry combined two public Pilkwang checkpoints with four team checkpoints. Five supplied detections with raw blend weights `[0.2, 0.4, 0.2, 0.2, 0.2]`; the sixth supplied an additional edge head. Each used its matching XY downsampling method (`stride` or `mean_pool`, factor 4). We used a `0.96875` detection threshold and no test-time augmentation. Candidate links used the maximum of the blended edge probability and three specialist edge-head probabilities.

An integer program selected a lineage graph with at most one parent and two children per cell. Graph repair closed short plausible gaps, re-linked endpoints using motion and edge scores, and removed unsupported short tracks. Learned pair and division classifiers vetoed doubtful divisions, followed by a sister-distance check. We corrected the XY grid-origin offset by `+2` voxels per axis and refined positions with a nine-member coordinate model.

The final `xyz2` CSV pass handled duplicated observations in byte-identical consecutive raw frames. It formed coordinate consensus for one-to-one links within `4 µm` and removed interior tracklets of at most three nodes.

## Results

| Submission | Public | Private |
| --- | --- | --- |
| Slot-2 fine-tuning with weight averaging | 0.969 | 0.929 |
| Add slot-4 fine-tuning with weight averaging | 0.971 | 0.928 |
| Add xyz2 CSV pass — selected final entry | 0.971 | 0.928 |

Against its exact parent, `xyz2` improved the local score on 26 development videos by `0.000674` (17 wins, one loss) and on 100 other `6bba` videos by `0.0011`. The displayed leaderboard score did not change. Broader BatchNorm adaptation, wider edge fusion, and one-frame track extension reduced public scores in earlier experiments, so we did not retain them.

Our initial validation split was within one embryo, while later all-data models had seen some local validation videos. Those comparisons helped select changes on identical inputs, but they were not a clean estimate for a new embryo. The selected submission finished at `0.971` public and `0.928` private.
