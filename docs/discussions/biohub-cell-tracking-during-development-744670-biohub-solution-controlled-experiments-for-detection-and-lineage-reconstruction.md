# Biohub solution: controlled experiments for detection and lineage reconstruction

- archived_at: 2026-10-02T11:53:16.518986+00:00
- source: https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/744670
- author: Jordi Corbilla
- posted_at_utc: 2026-09-30 18:02:04.359000
- votes: 1
- comments: 0

My final selected solution scored 0.955 public and 0.920 private, placing 398th of 4,017 teams at My post-deadline check. Medal confirmation was still pending.

I built on community foundations, with credit to Anvith Pothula and the original notebook contributors. The inherited pipeline combined 3D cell detection, learned temporal association, bidirectional probability fusion and constrained lineage reconstruction.

My experiments focused on detection recall and conservative graph repair: recovering discarded detections, closing gaps and preserving plausible divisions.

The strongest final build, BH-084, used a detector threshold of 0.9575 and a nearby discarded-peak readmission threshold of 0.945. I selected it alongside my incumbent, BH-070, to retain two valid, distinct graphs.

I followed a structured trial-and-error process:

- Freeze a reference build and state a specific hypothesis.

- Test individual changes before testing their interaction.

- Check that the intended mechanism actually activated.

- Audit output validity and compare coordinate-level graph changes.

- Record notebook versions, hashes, submission IDs and results, including failures.

A valid output was only the first gate. For example, I excluded a three-frame rescue experiment because it rescued zero three-frame tracks, despite producing a valid graph. Other ideas changed the graph but failed to improve the reported public score.

The clearest lesson came after closing: BH-070 and BH-084 both displayed 0.955 publicly, but scored 0.918 and 0.920 privately, respectively.

Displayed public ties did not establish equivalent generalization.

Keeping a distinct alternative helped, while careful rejection gates protected My limited compute and submission budget.

Thanks to the organizers and community for the benchmark, tools and shared work.
