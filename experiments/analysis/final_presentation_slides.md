# TinyVLA Final Presentation: Slide-by-Slide Talk Track

Use this as your speaking script. Each slide has:
- `Goal`: what the audience should remember
- `Show`: what to put on the slide
- `Talk track`: what to say

---

## Slide 1: Title + One-Line Problem
`Goal`: Frame the project in 20 seconds.

`Show`:
- Title: **Language-Conditioned Control with TinyVLA + MoE**
- Subtitle: *Can language-conditioned experts improve robot policy performance?*
- Your name, course, date

`Talk track`:
- "This project studies whether adding Mixture-of-Experts to a TinyVLA-style policy improves robotic control."
- "I compare four architectures: no-MoE baseline, text-conditioned MoE, full-feature MoE, and ACT."

---

## Slide 2: Motivation
`Goal`: Why this work matters.

`Show`:
- 3 bullets:
1. Robot tasks are diverse; one policy may underfit heterogeneous behaviors.
2. Language may help route behaviors to better specialists.
3. Offline training loss often differs from true closed-loop success.

`Talk track`:
- "My motivation was to test whether expert routing can help with task diversity."
- "I also wanted to measure generalization, not only training loss."

---

## Slide 3: Research Questions
`Goal`: Make evaluation criteria explicit.

`Show`:
1. Does MoE improve over a strong no-MoE baseline?
2. Is text-only routing enough, or do full features help?
3. Does ACT improve control quality in this setup?
4. How do seen-task and held-out-task results differ?

`Talk track`:
- "These questions guided the exact experiment design and final model selection."

---

## Slide 4: Architectures Compared
`Goal`: Show fair, controlled comparison.

`Show`:
- Table with models:
1. `no_moe`: MLP action head
2. `moe_text`: MoE, router conditioned on text
3. `moe_full`: MoE, router conditioned on action input (full features)
4. `act`: ACT chunked action head
- Shared setup:
  - `freeze_vision=true`, `freeze_text=true`
  - unfreeze last 2 layers for vision + text

`Talk track`:
- "Backbones and training setup are controlled; the main difference is action head/routing strategy."

---

## Slide 5: Data + Evaluation Protocol
`Goal`: Convince audience the evaluation is credible.

`Show`:
- Data: short-metaworld-vla
- Evaluation splits:
1. Seen-task evaluation (7 tasks)
2. Held-out task evaluation (`door-open-v3`, `peg-insert-side-v3`)
3. Peg-only focused training/eval (`peg-insert-side-v3`)
- Metrics:
  - Success rate
  - Avg max reward
  - Avg min object-to-target distance
  - Safety abort count

`Talk track`:
- "I intentionally report both binary success and dense metrics, because loss alone was misleading."

---

## Slide 6: Main Training Results (Core/Holdout)
`Goal`: Report supervised fit findings clearly.

`Show`:
- From `Output/presentation/run_20260419_161721/summary_models.csv`
- Core val loss:
  - `no_moe`: 0.004511
  - `moe_text`: 0.004545
  - `moe_full`: **0.004336** (best)
  - `act`: 0.005625
- Holdout-train val loss:
  - `no_moe`: 0.005605
  - `moe_text`: 0.005117
  - `moe_full`: **0.004488** (best)
  - `act`: 0.006512

`Talk track`:
- "On loss metrics, `moe_full` looks strongest."
- "But I do not pick model by loss only."

---

## Slide 7: Held-Out Closed-Loop Results
`Goal`: State the key limitation honestly.

`Show`:
- From `Output/presentation/run_20260419_161721/summary_rollouts.csv`
- All models: held-out success = **0.0** on both tasks
- Dense metric note:
  - On peg, ACT had best min distance (0.5412 vs 0.6010 for several others), but still no completion.

`Talk track`:
- "None of the models solved held-out tasks under this budget."
- "So this project becomes a strong analysis story on generalization gap."

---

## Slide 8: Seen-Task Results (Where Policy Actually Works)
`Goal`: Show learning is real, not completely broken.

`Show`:
- From `Output/presentation/seen_eval_20260420_201211/summary_seen_rollouts.csv`
- Mean success across 7 seen tasks:
  - `no_moe`: **0.6857**
  - `moe_full`: 0.5714
  - `act`: 0.4286
  - `moe_text`: 0.4000

`Talk track`:
- "Seen-task performance is clearly non-zero."
- "Best realized control is `no_moe`, even though `moe_full` had better val loss."

---

## Slide 9: Peg-Only Stress Test
`Goal`: Show you did targeted debugging for the hardest task.

`Show`:
- From `Output/presentation/peg_only_20260422_040829`
- Peg-only training val loss:
  - `no_moe`: 0.000776
  - `moe_text`: 0.000383
  - `moe_full`: **0.000374**
  - `act`: 0.009413
- Peg-only rollout success:
  - all models: **0.0**
- Dense peg metric best:
  - `moe_text` avg max reward = **2.7884**, avg min obj distance = **0.3051**

`Talk track`:
- "Even in peg-focused training, completion stayed zero, but MoE-text showed stronger dense progress signals."

---

## Slide 10: Safety + Failure Modes
`Goal`: Show maturity and practical awareness.

`Show`:
- Safety gate concept: verify object presence before acting; abort if absent
- Failure modes observed:
1. Peg insertion precision sensitivity
2. Distribution shift from seen to held-out tasks
3. Loss-performance mismatch

`Talk track`:
- "I added a safety check path to avoid unsafe blind actions."
- "Most failures are precision/generalization related, not just optimization."

---

## Slide 11: Final Conclusion
`Goal`: Make a clear, defensible final claim.

`Show`:
- **Final deployment choice:** `no_moe`  
  checkpoint: `checkpoints/checkpoints_stage2_no_moe_unfreeze/best.pt`
- Why:
1. Best seen-task closed-loop success
2. Most reliable task completion behavior
3. MoE best loss did not transfer to best success

`Talk track`:
- "If the criterion is real task completion, `no_moe` is the best current model."

---

## Slide 12: Future Work
`Goal`: End with a realistic and strong roadmap.

`Show`:
1. Improve held-out generalization (task/data coverage + curriculum)
2. ACT improvement: chunk execution/temporal ensembling (not only first action token)
3. Better policy class for long-horizon precision (diffusion/stronger autoregressive action decoding)
4. More robust evaluation suite (seen/held-out, safety, failure taxonomy)

`Talk track`:
- "The next step is not another random architecture swap; it is targeted generalization and control stabilization."

---

## Backup Slide A: Why MoE Did Not Win Yet
`Show`:
- Router entropy/gap evidence from MoE logs
- Hypothesis: expert specialization not yet aligned with control primitives

`Talk track`:
- "MoE capacity exists, but routing quality and policy objective may be misaligned with success signal."

## Backup Slide B: Reproducibility / Artifacts
`Show`:
- Run folders:
  - `run_20260419_161721`
  - `seen_eval_20260420_201211`
  - `peg_only_20260422_040829`
- CSVs, logs, videos, architecture notes

`Talk track`:
- "All claims are tied to saved artifacts and can be reproduced."
