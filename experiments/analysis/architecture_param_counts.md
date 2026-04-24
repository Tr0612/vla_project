# Architecture Parameter Counts

Counts are computed by instantiating each config with current code.
`total_params` includes all weights; `trainable_params` reflects freeze/unfreeze policy.

## By Config

| Config | Architecture | Freeze Vision | Freeze Text | Unfreeze Vision Last N | Unfreeze Text Last N | Total Params | Trainable Params | Total (M) | Trainable (M) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| experiments/no_moe.yaml | mlp | 1 | 1 | 0 | 0 | 382,562,824 | 7,374,856 | 382.563M | 7.375M |
| experiments/no_moe_unfreeze.yaml | mlp | 1 | 1 | 2 | 2 | 382,562,824 | 35,726,344 | 382.563M | 35.726M |
| experiments/moe_text.yaml | moe | 1 | 1 | 0 | 0 | 383,395,864 | 8,207,896 | 383.396M | 8.208M |
| experiments/moe_text_unfreeze.yaml | moe | 1 | 1 | 2 | 2 | 383,395,864 | 36,559,384 | 383.396M | 36.559M |
| experiments/moe_full.yaml | moe | 1 | 1 | 0 | 0 | 383,395,960 | 8,207,992 | 383.396M | 8.208M |
| experiments/moe_full_unfreeze.yaml | moe | 1 | 1 | 2 | 2 | 383,395,960 | 36,559,480 | 383.396M | 36.559M |
| experiments/act.yaml | act | 1 | 1 | 0 | 0 | 382,829,576 | 7,641,608 | 382.830M | 7.642M |
| experiments/act_unfreeze.yaml | act | 1 | 1 | 2 | 2 | 382,829,576 | 35,993,096 | 382.830M | 35.993M |
| experiments/act_moe_unfreeze.yaml | act_moe | 1 | 1 | 2 | 2 | 384,538,636 | 37,702,156 | 384.539M | 37.702M |

## Aggregated (Architecture + Freeze Regime)

| Architecture | Freeze Vision | Freeze Text | Unfreeze Vision Last N | Unfreeze Text Last N | Total Params | Trainable Params |
|---|---:|---:|---:|---:|---:|---:|
| mlp | 1 | 1 | 0 | 0 | 382,562,824 (382.563M) | 7,374,856 (7.375M) |
| mlp | 1 | 1 | 2 | 2 | 382,562,824 (382.563M) | 35,726,344 (35.726M) |
| moe | 1 | 1 | 0 | 0 | 383,395,864 (383.396M) | 8,207,896 (8.208M) |
| moe | 1 | 1 | 2 | 2 | 383,395,864 (383.396M) | 36,559,384 (36.559M) |
| act | 1 | 1 | 0 | 0 | 382,829,576 (382.830M) | 7,641,608 (7.642M) |
| act | 1 | 1 | 2 | 2 | 382,829,576 (382.830M) | 35,993,096 (35.993M) |
| act_moe | 1 | 1 | 2 | 2 | 384,538,636 (384.539M) | 37,702,156 (37.702M) |
