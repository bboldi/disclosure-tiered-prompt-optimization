**Executor calls per trajectory by arm. Main and reference arms plan 576 search evaluations per trajectory (39 main trajectories in total); a candidate slot whose proposal failed is forfeited and recorded in the trajectory's decisions, so some trajectories spent fewer. GEPA runs report their own executor call counts. Selection and sealed evaluations are shared across arms through the sealed-panel cache.**

| Arm | Search schedule | Proposal evaluations | Incumbent re-evaluations | Search evaluations per trajectory | Selection (validation) | Sealed evaluations per selected prompt |
|---|---|---|---|---|---|---|
| Main (GLM-5.3, Tiers 1–3) | 8 rounds × 2 candidates × 24 profiles | 384 | 192 | 576 | shortlist × 96 | 384 |
| Reference optimizer (DeepSeek V4 Pro, Tier 2) | 8 rounds × 2 candidates × 24 profiles | 384 | 192 | 576 | shortlist × 96 | 384 |
| GEPA (Tier 2) | engine-controlled; capped by executor profile calls | — | — | 600 | shortlist × 96 | 384 |
