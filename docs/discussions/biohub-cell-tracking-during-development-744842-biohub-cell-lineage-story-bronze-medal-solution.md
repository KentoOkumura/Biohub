# Biohub Cell Lineage Story -🥉bronze medal solution

- archived_at: 2026-10-02T11:53:17.850471+00:00
- source: https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/744842
- author: Kirderf
- posted_at_utc: 2026-10-01 06:30:40.386000
- votes: 2
- comments: 0

# Tracking Every Cell in a Zebrafish Embryo

### A four-dimensional field guide to the BioHub cell-tracking competition — and one team's run at it

A short, visual walk through the competition — the movie, the lineage graph, the metric, and the result. Every figure below is computed directly from the competition's ground-truth data. Final outcome: 0.954 public / 0.922 private. ( best solution, not the two final pick )

## 1 · The movie — 4-D light-sheet microscopy

Each film is a `(T, Z, Y, X)` OME-Zarr volume — typically `100 × 64 × 256 × 256` uint16, about 104 µm per side. Two properties dominate everything downstream:

- Anisotropy — the voxel is 1.625 µm deep (Z) but only 0.406 µm wide (Y/X). Any distance heuristic in raw voxel units is quietly wrong.

- Sparse labels — the ground truth annotates only ~1–9% of nuclei per film (≈1% on embryo `44b6`, ≈9% on `6bba`). The scorer's `estimated_number_of_nodes` reveals the true cell count is far larger — the training signal is a tiny subsample of truth.

![2d](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F3924325%2F26c1c90f7f6a99777dde5176a404df08%2Ffig1_2d_microscopy.png?generation=1790835970486130&alt=media)

One Z-slice of a real training volume — fluorescent nuclei in a developing zebrafish embryo. Every dot is a cell the tracker must follow across ~100 frames.

## 2 · The answer is a graph , not a mask

The deliverable is a lineage graph (GEFF): nodes `(t, z, y, x)` for detected cells, edges stepping exactly one frame forward, and out-degree-2 forks for divisions. This is the real ground-truth graph of film `6bba_05db0fb1` — 1,229 cells, 52 lineages, 3 divisions, colored by time:

![3d](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F3924325%2F46a471d23406245ed1273689a54faee7%2Ffig2_3d_lineage.png?generation=1790836071067836&alt=media)

White/orange diamonds mark division events — a parent node forking into two daughters. Across all 199 training films there are only 151 of these.

## 3 · Time is the fourth dimension

Nuclei drift, crowd, and divide. The "who became whom" decision the scorer grades happens between every consecutive pair of frames — 99 separate arbitrations per film.

![source image](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F3924325%2F5f313d000cfa46d80938efb789cd2897%2Ffig3_4d_montage.png?generation=1790836134266865&alt=media)

Six moments from one film — bright = current frame, dim = 8-frame trail. Watch how the cloud compresses, rotates, and densifies; cells in dense regions are where linking fails.

## 4 · Divisions — the rare forks worth 10% of the grade

`score = adjusted_edge_jaccard + 0.1 · division_jaccard`. With ~151 sparse GT forks and an unweighted loss, models never learn to propose second daughters — and post-hoc geometric rescue produces roughly 3,000 false forks per true positive. Most public solutions leave this term at zero.

![source image](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F3924325%2F0d85323aae6671ea15a09148d61220b2%2Ffig4_division.png?generation=1790836150705573&alt=media)

A real mitosis: the white diamond is the parent at t, the teal circles are its two daughters at t+1.

## 5 · The metric — an edge only counts if both cells match

Predicted nodes match GT nodes within 7 µm in scaled space (optimal bipartite matching); an edge is correct only when both endpoints match and the link exists in GT. The adjusted term adds a ~10%-per-node over/under-emission penalty — and the organizers hint it can push scores past 1.0 ("get your number of nodes correct using texture regression, like head counting").

![source image](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F3924325%2F518f6ee65ef981a093f7408ef482d4bc%2Ffig5_metric.png?generation=1790836160883484&alt=media)

GT links as arrows; orange circles show the 7 µm matching radius. Both endpoints of a predicted edge must land inside the right circles.

## 6 · The run — and the leaderboard inversion

The stack: 3-D UNet detector → transformer edge-scorer (two seeds, harmonic-probability fusion) → ILP global solve → motion-relink + gap-close + safe-division post-processing.

The four submissions below all come from that family. The result that matters most:

| submission | public LB | private (hidden embryo) |
| --- | --- | --- |
| x138 mnn-head iter12 | 0.944 | 0.922 |
| x138 dense iter12 | 0.954 | 0.921 |
| x138 retention iter12 | 0.937 | 0.921 |
| x138 candret98 iter12 | 0.954 | 0.920 |

![source image](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F3924325%2F43dc825971a2e31c68cb2c1fe68d199a%2Ffig6_results.png?generation=1790836171766621&alt=media)

The variant with the lowest public score finished with the highest private score. The public board — computed on 4 films that are byte-identical copies of training data — anti-predicted generalization to the hidden third embryo. Trust leave-one-embryo-out, not the mirror.

## What we would tell a newcomer

- Detection is solved; linking is the game; divisions are the frontier. ~83% of missed edges had both cells detected — the loss is arbitration, not perception.

- The public leaderboard is a mirror, not a window. Every gain tuned on the public twins risked flipping sign on the private embryo.

- Post-processing is saturated. Divisions can't be rescued geometrically; the remaining score needs a model — edge-centric attention, SSL pretraining on the unlabeled 88 GB, or learned solver costs.

- Score can exceed 1.0. The node-count term is a designed bonus — density/head-count regression is the officially-hinted unlock.

That's it, happy Kaggling!
