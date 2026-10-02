# 0.957 Public → 0.916 Private: A Postmortem of Three Validation Traps

- archived_at: 2026-10-02T11:53:16.295050+00:00
- source: https://www.kaggle.com/competitions/biohub-cell-tracking-during-development/discussion/744647
- author: Arturo Gutiérrez Aguilar
- posted_at_utc: 2026-09-30 16:56:04.146000
- votes: 1
- comments: 0

I ended the public leaderboard around rank 231 / 4,020 with 0.957. After the reveal, my two selected submissions scored 0.917 and 0.916 private; my first revealed rank was 1,097 / 4,017.

This was not just ordinary shake-up. My experiment history shows how I gradually replaced validation with feedback from the public leaderboard.

| Submission | Public | Private | Selected? |
| --- | --- | --- | --- |
| Final: division cost 0.4 + readmission 0.94 | 0.957 | 0.916 | Yes |
| Final: division cost 0.8 + readmission 0.94 | 0.956 | 0.917 | Yes |
| Earlier: x138 + five V1284 heads | 0.952 | 0.923 | No |

## The warning I had before the deadline

I ran the official metric over all 199 training videos in my local laboratory. One candidate improved locally by +0.00140, yet fell from the 0.955 baseline to 0.947 public. Another gained +0.00088 locally and scored 0.953 public. In fact, all five candidates in that batch looked better locally, but all five scored worse publicly: 0.947–0.953.

The reason was visible in my own audit: the public detector used by the pipeline had already been trained on all 199 videos. The local metric was excellent for finding broken graphs and inactive code, but it was an in-sample diagnostic rather than an estimate of generalization. Even the incremental bootstrap interval for my most promising change crossed zero.

## Why 0.957 still looked convincing

Near the deadline, a public discussion suggested lowering the ILP division cost from 1.2 to 0.4 and the localized readmission threshold from 0.965 to 0.94. On my 0.955 pipeline, each ablation displayed 0.956 publicly and their combination displayed 0.957. That looked like a clean two-factor confirmation.

But all three measurements came from the same 29% public split. I had made 38 submissions, read public discussions, and tested many related tracking rules. The public leaderboard had silently become part of my training loop. My five final candidates were nearby settings of the same two knobs, so selecting two of them provided almost no protection against a distribution shift.

I also verified exact graph parity between my laptop and Kaggle. That proved reproducibility and ruled out an implementation mismatch; it never provided evidence that the graph changes were biologically correct on unseen embryos.

## What I would change next time

- Hold out complete biological/acquisition groups before training any component. A detector that has seen every validation video invalidates the downstream score, even when the linker itself is new.

- Give each metric a fixed role. In-sample official metric for debugging; group-held-out metric for selection; public LB only for sparse external checks.

- Pre-register a minimum improvement and inspect uncertainty. I should not promote +0.001 when its paired confidence interval includes zero.

- Count leaderboard queries as multiple comparisons. After many submissions, a small public gain is increasingly likely to be selection noise.

- Select a diverse final portfolio. I should have selected the distinct 0.952/0.923 model alongside one public winner instead of two neighboring threshold variants.

The private result does not prove that one threshold alone caused the failure. It shows that my selection procedure could not distinguish a transferable improvement from a public-specific one.

I published the full experiment trail—including negative results, notebooks, hashes, local audits, and final receipts—here:

[https://github.com/Arturo-GA/biohub-cell-tracking-lab](https://github.com/Arturo-GA/biohub-cell-tracking-lab)

I hope the failure is useful to teams working on future small-data scientific competitions. I would be interested to know how other teams built independent validation with so few biological groups.
