# 301st Place: Bounded Native Links for 3D Cell Tracking

- archived_at: 2026-10-02T11:53:18.484028+00:00
- source: https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/744895
- author: MOHAMMADJAFAR ZAMANI
- posted_at_utc: 2026-10-01 09:48:50.158000
- votes: 1
- comments: 0

# 301st Place: Bounded Native Links for 3D Cell Tracking

A bounded native-link guard on a dual-model 3D tracking pipeline — 0.957 public score and a bronze medal.

## Result and scope

My final placement was 301st of 4,017 teams, with a bronze medal, according to Kaggle's competition completion email. The best recorded public score of the selected notebook was 0.957. I do not have a separately verified private score to report. The selected submission was Biohub Final D1 Bounded Native Guard, version 2. Its final change was a small rule in an established tracking pipeline: restore a limited number of strong links from the original model when motion relinking has displaced them. No new image model was trained for this submission.

## The problem

[Biohub — Cell Tracking During Development](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development) asks competitors to detect cells in 3D microscopy, associate them over time and identify divisions to reconstruct lineages. The practical motivation is to reduce the manual work needed to follow dense, visually similar cells during development. The [competition data](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/data) consists of fluorescence microscopy movies of zebrafish embryos. The training annotations are sparse, so unlabeled detections cannot simply be treated as mistakes. The [evaluation](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/overview/evaluation) combines temporal-edge accuracy and division detection. That context shaped my final intervention. Rather than add a broad new association rule, I tried to correct a narrow class of disagreements between learned links and the motion postprocessor.

## The inference pipeline

The notebook performs offline 3D cell detection and lineage tracking. It loads existing UNet/Transformer tracking checkpoints, combines predictions from two models and spatial test-time views, and uses a separate DeepCenter 3D checkpoint as an additional gate for selected gap and division decisions. Candidate detections and learned edge probabilities form a time-directed graph. The postprocessing includes motion relinking, recovery of discarded detections, short-gap handling, division checks and graph constraints. The final export contains cell coordinates and consecutive-frame edges in the competition's node/edge CSV schema. The submission notebook loads attached checkpoints and performs no new training. Its code implements image normalization, detector thresholds, model inputs and conversion of output coordinates. Reproducing it requires the compatible model-artifact inputs. This write-up does not release those weights or claim to document their training recipes.

## D1: bounded restoration of native links

Motion relinking can replace an edge that the original model assigned between two detections. I call those original associations native links. D1 revisits only native one-to-one edges displaced by motion relinking. A candidate must satisfy four conditions:

- The learned edge probability is at least 0.60.

- The physical distance between its endpoints is at most 5.5 µm.

- Both endpoints existed before discarded detections were readmitted.

- The original source has one outgoing edge and the original target has one incoming edge. The guard excludes raw division edges. It also refuses a restoration when the conflicting motion links involve a readmitted detection or a division-like outgoing configuration. Eligible links are ordered by probability and distance, with deterministic tie-breaking. The restoration budget is min(256, ceil(0.002 × raw edge count)), with at least one slot when raw edges exist. A restored native link can displace conflicting motion links; subsequent graph repair still enforces the required output topology. The following is a condensed explanation, not a drop-in implementation:

```text
eligible = [
    edge for edge in raw_edges
    if edge not in motion_edges
    and endpoints_are_original(edge)
    and raw_outdegree(edge.source) == 1
    and raw_indegree(edge.target) == 1
    and learned_probability(edge) >= 0.60
    and physical_distance_um(edge) <= 5.5
]
budget = min(256, max(1, ceil(0.002 * len(raw_edges))))
for edge in deterministic_confidence_distance_order(eligible):
    if budget == 0:
        break
    if conflicts_touch_readmitted_node_or_division(motion_edges, edge):
        continue
    replace_conflicting_motion_edges_with_native_edge(edge)
    budget -= 1

```

The full source also checks graph degrees. D1 adds no detections and does not globally relax association. It gives a small, bounded set of confident original links a second opportunity after motion postprocessing.

## What the development replay showed

The decisive comparison was a paired replay of the complete postprocessing and the competition scoring code on 12 fixed training movies. Before testing the guard, I reproduced the original D1 graph hashes and official metric columns.

| Development observation | Measured result |
| --- | --- |
| Combined metric delta over 12 movies | +0.0011053 |
| Delta on four additional movies | +0.0010566 |
| Positive leave-one-movie-out comparisons | 12 of 12 |
| True-positive edge change | +3 |
| False-positive edge change | −6 |
| False-negative edge change | −3 |
| Division-count change | 0 |
| These movies had been exposed to training. This is development analysis, not independent validation on unseen embryos . The leave-one-movie-out comparisons test the stability of this paired result; they do not undo that exposure. |  |
| The rise from earlier 0.955 public submissions to 0.957 is a separate leaderboard observation. It is consistent with the change, but a leaderboard comparison alone does not establish that D1 caused the entire increase. |  |

## Execution and output checks

The selected version completed a visible 20 min 20 s run on GPU T4 × 2. The completed run reported no repair/deadline fallback. I checked the final CSV against the test-dataset inventory and exact column order. The checks covered finite integer coordinates within input-derived bounds, unique node and edge IDs, existing endpoints, consecutive-frame links, indegree at most one and outdegree at most two. These checks establish that the exported artifact meets the expected structural conditions. They do not establish biological correctness or hidden-test accuracy. One adjacent-frame division chain in the final graph was flagged for biological review.

## Ideas I did not include

A later 0.65 line-fit variant gained slightly on the original eight movies but lost on four additional training movies. A temporal-anchor priority extension had a slight negative paired delta. A raw-division-neighborhood veto produced exactly the same graphs and metrics as D1. A geometry-only Voronoi guard changed some serialized coordinates but left the official metric unchanged on all 12 movies. A local rigid-motion consistency deletion rule reduced the combined metric. Many removed edges had unknown adjudication status, so I cannot describe them as false positives. A proposed learned edge-reliability rule failed its prespecified support gate: each training group lacked enough adjudicated positive examples. It therefore has no measured score. I did not combine these negative or inconclusive ideas with the selected submission.

## Resources and reproducibility

The exact D1 version 2 code, completed run and final output receipt define this submission. Some descriptive cells were inherited from earlier experiments and are stale. The independent replay artifacts can be supplied separately; private notebooks should not be described as publicly reproducible resources. Useful public references:

- [Competition overview](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development)

- [Competition data](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/data)

- [Competition evaluation](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/overview/evaluation)

- [Competition host's code and scoring repository](https://github.com/royerlab/kaggle-cell-tracking-competition)
