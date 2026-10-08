# Leakage and evidence audit

| Control | Implemented behavior | Real-data status |
|---|---|---|
| Chart absences | Complete Saturday issues, 100 ranks, reconciled canonical identity, full release-to-cutoff query | Blocked: archive absent |
| Aliases/versions | Evidence-backed IDs only; unresolved same-work versions excluded | Blocked: no reviewed cohort |
| Group leakage | Transitive shared performers, work, recording, album, match links, exact audio/features, reviewed duplicate IDs and conservative audio signature | Synthetic regression verified |
| Duplicate conflicts | Opposing labels on the same work/recording/audio/features block dataset loading | Real review unavailable |
| Holdout exposure | Separate table; trainer never reads labels; shared one-access ledger created before evaluation | Synthetic regression verified |
| Preprocessing | Pipeline fit inside every fold; strict 518 audio columns, std74 selection internal | Regression verified |
| Source nuisance | Development-only grouped model using source/codec/duration/year/sample quality; high performance blocks test | No real score |
| Label permutation | Fixed within-group shuffle preserving group prevalence; inapplicable cases flagged | No real score |
| Human chorus truth | Separate review status; fallback and automatic excerpt not certified chorus | 0 real annotations |
| All original assets | Preservation SHA manifest and existing validators/tests | Locally checked; see verification logs |

The perceptual signature is a conservative review/grouping aid, not proof that every remix, cover,
mastering or near-duplicate was detected. Real fingerprint/identity review remains indispensable.
Artist-connected holdout assessment remains blocked if group support is insufficient. Secondary
album-disjoint evaluation is deliberately not produced while the primary dataset does not exist.
No seed search or relaxation of identity isolation is used to make a split pass.

Local hashes detect unintended changes; they do not authenticate a provider or withstand a malicious
owner rewriting every manifest. Pickle/joblib bundles must come only from trusted local training.
The test ledger is a research-process guard; an owner must not delete it to bypass the protocol.
