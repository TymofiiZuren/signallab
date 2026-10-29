# Synthetic Review Lab: validation evidence

The software workflow is implemented through T6. The pinned real FLAN-T5 preparation, real resource pilot and human semantic-quality study remain **not run**. Authored fixture acceptance and synthetic resource observations are not experimental results.

## Reproduce the software checks

```bash
uv run --locked python -m unittest discover -s tests -p test_synthetic_journey.py
uv run --locked python -m unittest discover -s tests
uv run --locked python -m compileall -q app.py signallab tests
```

The integrated journey has four tests. The recorded full suite passed **158 tests in 46.669 seconds**; compileall passed. Tests use private `.scratch/` roots, mock Hub downloads and never populate production study/pilot roots. Tiny local BERT/T5 fixtures exercise actual framework execution without establishing production model quality. The journey executes the Python example in the [learner walkthrough](synthetic-experiment-walkthrough.md), fits its TF-IDF classifier, checks validation-only metrics and fails if the consumer re-splits the source.

The acceptance IDs below refer to the approved `.scratch/synthetic-review-plan/TEST_PLAN.md`. Related checks are grouped for readability; this is an evidence map, not a measured code-coverage percentage or a claim that every possible fault has been injected.

| Acceptance IDs | Evidence and limits |
|---|---|
| U01–U07 | `tests/test_study.py`: legacy canonical keys, text-only final exclusions, input bounds/types, immutable/private publication, schema/digest/membership rejection, frozen partition reattachment and balanced training-only selection. Semantic cohort independence cannot be proved by fingerprints. |
| U08 | `tests/test_synthetic_cli.py`: real tiny tokenizer overflow and bounded generation; `tests/test_synthetic.py`: output/EOS limits. Real FLAN tokenization feasibility remains M01. |
| U09–U10 | `tests/test_synthetic.py`: original/final/candidate exclusions and frozen duplicate owner irrespective human edits; `tests/test_synthetic_handoff.py`: accepted final collision excluded from training. |
| U11 | `tests/test_synthetic.py`: literal text and bounded output publication; native UI displays text as data. No automatic semantic verification is claimed. |
| I01–I03 | `tests/test_synthetic_cli.py`: pinned/mock Hub preparation, allowlist/size/symlink checks, tiny real local T5 load/generation, incomplete weights, identity replacement, overflow and cache miss without download. These are not a real pinned Hub smoke test. |
| I04–I05 | Synthetic evidence and CLI suites: durable attempt intent, immutable terminal binding, bounded one-attempt worker protocol and malformed/dead worker failure. |
| I06–I10 | CLI suite: fake-worker wall/RSS guards, failed measurement, interruption, kill escalation, supervisor death, reaping and recovery without resuming; synthetic suite: inherited generation lock and interrupted evidence. Only test-owned processes are stopped. Sampled RSS cannot prove hard peak containment. |
| I11 | UI and journey suites: exact development attachment, missing/mismatched source removes pair/form while numeric evidence remains available. |
| I12–I15 | Synthetic suite: actual independent-process CAS, request/payload binding, preserved historical snapshots, publication failures and revision bounds. UI/journey suites: stale draft retained separately, explicit reload and context reset. Fault injection is representative, not an exhaustive operating-system failure matrix. |
| I16 | Handoff suite: 29/30 and per-class thresholds, all outputs reviewed, resource/generation/provenance gates, all-rejected valid negative result. Fixture decisions are not human quality evidence. |
| I17–I19 | Handoff suite: shared review lock with actual competing process, immutable seals, idempotent retry, review edits make old seals historical, tampered/unsupported/oversized records fail closed. Journey: damaged review evidence preserved and app/CLI blocked. |
| I20–I21 | Handoff/journey suites: accepted eligible descendants only, aligned lineage, exact ordered original partitions, independent returned frames, current-disk revalidation, stale seal/source rejection, baseline consuming the bundle without a splitter. |
| I22 | Study/synthetic/handoff/UI suites cover bounded CSV/records/revisions/seals and discovery limits, metadata-only history reading. This is bounded-contract evidence, not a load/performance benchmark. |
| A01–A02 | `tests/test_synthetic_ui.py`, CLI tests and main-app journey: early route before shared CSV gate, empty/error/running/stopped evidence, side-effect-free metadata and explicit commands. |
| A03–A04 | UI/journey suites and live fixture browser: explicit attachment, literal original/candidate, required verdict/reason/acknowledgement, Pending/All filters and explicit save. |
| A05–A06 | UI/journey suites: confirmed-save progression, uncertain acknowledgement retry, stale conflict/reload, source/navigation reset. Two live independent browser sessions reproduced stale rejection, preserved draft and explicit reload to the saved decision. |
| A07 | Handoff/UI suites and live browser/CLI: separate review/resource/qualification/currency status, historical versus current seal, no training claim; unsupported re-splitting trainers are excluded from the recipe. |
| A08 | Bounded manual accessibility checks: actual native Chrome 200% zoom and keyboard traversal through decision/reason/acknowledgement/save; native control labels/focus retained. Runtime reduced-motion emulation reported introduction animation `none`, sweep display `none`, sweep animation `none`; browser settings restored. Live DOM review radio/checkbox label targets measured at least 44px. A complete contrast inventory, screen-reader test and WCAG audit remain unperformed. |
| A09 | Live fixture measured at 1280/1100/768/390/375/320px: document width equals viewport, two columns at 1280, original-first stacking at all narrower tested widths; minimum review label target 44px. Prior T4/T5 visual checks complement these measurements. This does not exhaust all device/font/content combinations. |
| M01 | **Not run:** real pinned FLAN-T5-small preparation/loading with the locked dependencies. Tiny real T5 and mocked Hub preparation are software evidence only. |
| M02 | **Not run:** actual pinned model CPU resource pilot and real yield. Fake-worker measurements and authored fixture resource records do not establish feasibility. |
| M03 | **Not run:** human review of real generated candidates. Authored accept/reject decisions prove persistence mechanics only. No augmentation improvement is established. |

## Recorded integrated browser result

The private authored preview at port 8546 started with revision 29 and one pending decision. Session A saved revision 30; session B's stale proposal was refused and retained separately. Explicit reload showed A's actual saved Accept and reason. A seal at revision 30 became historical after a reason edit at revision 31. A new seal pinned revision 31. The actual CLI consumer rejected the historical seal with exit 1 and accepted the new seal with exit 0: original train/validation/test **72/24/24**, accepted synthetic **30**, augmented training **102**. No generator or classifier ran in that browser exercise.

Local evidence: `.scratch/synthetic-integration-implementation/` contains `full-tests.log`, `journey-tests.log`, `browser-handoff-check.json`, `responsive.json`, `reduced-motion-runtime.txt`, and browser screenshots. These are developer QA artifacts, not reusable research results. The preview banner explicitly labels authored text, decisions, hashes and resource values as fixtures.

## Next research gate

Follow the [learner walkthrough](synthetic-experiment-walkthrough.md) with legitimate development data and an independently reserved final cohort. Explicit preparation is the model-download action. Preserve failed or low-yield pilots rather than replenishing parents to pass a threshold. After the bounded real pilot and full human review, the later four-condition, matched-budget, multiple-seed study still needs implementation. One exploratory validation score is not evidence of augmentation benefit or final-cohort performance.
