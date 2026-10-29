# SignalLab follow-up work

Created by the 2026-10-23 autoplan review. Planning records only; no implementation or external writes authorised. The research-first direction is user-confirmed; the user approved the plan as-is on 2026-10-23.

## TensorFlow/Keras reproduction — P2, later learning milestone
- What: reproduce one declared baseline/transformer experiment using TensorFlow/Keras after the PyTorch study contract is stable.
- Why/pros: supports the user's original framework-learning goal and compares APIs against the same data/evaluation evidence.
- Cons: another production dependency requires explicit approval; duplicated training code and hardware/runtime validation add maintenance.
- Context: do not use different splits, model families or training budgets and call the result a framework effect. Start from NEXT_STEPS.md exact-input/run/protocol boundaries.
- Effort: human 2–4 days / AI-assisted 4–8h, plus measured training time.
- Depends on: run contracts, saved-model inference, bounded study and case study. Not in the next first-release patch.

## Broader synthetic-generation study — P2, pilot-gated
- What: scale reviewed paraphrases or compare generators after the <=50-parent local feasibility pilot.
- Why/pros: tests whether the result survives more data/methods rather than relying on one tiny run.
- Cons: review effort, CPU/memory budget and multiple-comparison risks; no improvement guaranteed.
- Context: FLAN-T5-small is a candidate, not a proven paraphraser. Preserve every attempted/rejected/ambiguous example and use common parent IDs for controls.
- Effort: human 2–5 days / AI-assisted 4–8h plus measured review/training time.
- Depends on: pilot resource/quality gates, final-cohort protocol fixed before generation.

## Group-aware / new-domain replication — P2, later research
- What: evaluate on independent product/customer groups or a new domain when lawful identifiers/data exist.
- Why/pros: stronger evidence beyond randomly partitioned public review rows.
- Cons: current data contract discards optional metadata; new schema/data provenance and grouping validation are needed.
- Context: current deduplication cannot prove absence of semantic copies or pretraining contamination. Document this limitation rather than treating hashes as a cure.
- Effort: human 2–4 days / AI-assisted 4–8h plus data review.
- Depends on: a concrete identified dataset, group fields and explicit scope approval.

## Calibration / abstention — P3, separate question
- What: assess calibration and a neutral/mixed/uncertain outcome only when separately labelled evaluation data exists.
- Why/pros: probabilities would be easier to interpret and mixed reviews better represented.
- Cons: changes binary product/label contract; threshold tuning consumes validation data and requires new tests.
- Context: current probabilities remain explicitly uncalibrated; do not add a cosmetic confidence threshold pretending to solve the problem.
- Effort: human 1–3 days / AI-assisted 2–6h plus labelling.
- Depends on: new evaluation question and approved label/data scope.

## Cross-platform fresh-machine walkthrough — P3
- What: measure cold setup and documented recipe on additional supported hardware/OS.
- Why/pros: turns portability claims into evidence.
- Cons: requires access to the actual machines and model download budget.
- Context: current evidence is local macOS ARM64; lockfile compatibility does not establish usability elsewhere.
- Effort: human 0.5–1 day / AI-assisted 1–2h per available environment, plus downloads.
- Depends on: implemented instructions and access to other machines; no CI workflow added.

## Public portfolio distribution / licensing — P3, user decision
- What: decide repository visibility, code license, data/model redistribution policy and presentation channel.
- Why/pros: other people could inspect/reproduce the portfolio.
- Cons: external publication and legal/data review are outside the current local scope.
- Context: no public repository or license currently established. Never publish raw uploaded reviews, model weights or dataset copies by default.
- Effort: human 0.5–1 day / AI-assisted 1–2h after decisions.
- Depends on: explicit user instruction for publication; no git/hosting/account action inferred.

## Controlled study consuming reviewed bundles — P2, after Synthetic Review Lab
- What: consume the validated fixed-partition reviewed bundle; compare real, duplicate, prefix and reviewed-paraphrase controls using common parent IDs and matched budgets across seeds 13/42/87.
- Why: distinguish generation feasibility from measured classification benefit without re-splitting synthetic rows into held-out partitions.
- Limits: reserve final evaluation under its frozen protocol; no improvement promised. No automatic training, model download or publication authorized by the plan approval.
- Depends on: all six Synthetic Review Lab implementation tasks and recorded resource/human-review gates.
- Planning reference: `.scratch/synthetic-review-plan/PLAN.md`, approved as-is on 2026-10-25. Existing CSV training commands cannot consume this handoff.
- Effort: estimate during its own review; real compute and human review remain additional.
