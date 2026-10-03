# Comparison with Eric Liu's original project

Checked 3 October 2026 against [the original Stanford report](https://cs229.stanford.edu/proj2021spr/report2/81974051.pdf), especially Table 2 on PDF page 5.

**We have not demonstrated that our project outperforms Eric Liu's.**

| Reported measure | Eric Liu | Our recorded result |
|---|---:|---:|
| Preferred NN test accuracy | 58% | No matched replication |
| Highest listed test accuracy (random forest) | 68% | No matched replication |
| V3 nested development accuracy | Not the same protocol | 53.27% |
| V3 nested development balanced accuracy | Not reported in Table 2 | 53.25% |
| Frozen LR ten-seed mean accuracy / balanced accuracy | Not the same protocol | 53.18% / 53.05% |
| Combined-feature pilot accuracy / balanced accuracy | Not the same population | 57.35% / 54.17% |

His preferred NN has reported test precision, recall and F1 of 56% each. Its 62% CV accuracy is separate from its 58% test accuracy. His table's 68% random-forest accuracy is the highest listed accuracy even though the narrative prefers the NN across metrics. Do not call 58% his highest accuracy or claim he reached 75%.

The original study describes 554 tracks from seven artists, chart-reaching songs versus other works from the same artists/albums. Our inherited labels distinguish year-end hits from other charted songs. Our evaluation explicitly separates artist-name groups; the report describes a train/test split and five-fold CV without documenting an artist-disjoint guarantee. This does not establish that his evaluation leaked data.

A fair superiority claim would require the same task, recordings/features, splits and metrics with uncertainty. Neither the numerical comparison nor the current evidence supports that claim. Our reproducibility checks are useful engineering evidence, not proof of better prediction.
