# Run one inspectable synthetic-data experiment

This walkthrough connects SignalLab's existing protocol, CLI, human review, seals and fixed-partition consumer. It does **not** promise that paraphrases improve a classifier. The offline test rehearsal uses authored fixtures; your real experiment requires your own development data, a separately reserved final cohort, an explicitly prepared model and honest human review.

## Rehearse without a model download

From the project directory, in the existing locked environment:

```bash
uv run --locked python -m unittest discover -s tests -p test_synthetic_journey.py
uv run --locked python -m signallab.synthetic_cli --help
```

The journey tests use private roots under `.scratch/`, exercise two independent app sessions and the actual filesystem/CLI consumer, and fit a small TF-IDF classifier for the documentation example. They never prepare/download/run the Hugging Face generator and do not populate your research artifact roots. Fixture model hashes, generated text, resource observations and review decisions are authored inputs—not real feasibility results. Passing tests establish software behavior, not semantic quality.

## Freeze the question before generating

Keep an original development CSV and a different reserved-final CSV, both UTF-8 with `text,label` columns. Labels are binary negative/positive or 0/1. Do not place credentials or private review text in source descriptions. The original CSVs remain with you; retain them alongside compatible code/environment. The reserved-final cohort must stay outside prompts, review examples and development decisions. Software can reject known text overlap; it cannot prove that the cohort was never inspected or detect every semantic duplicate.

There must be enough real training parents for the question. A valid tiny CSV can still have too few training rows to reach 30 accepted outputs. The selection is at most 25 parents per class, in frozen order; skips, rejections and exclusions are never replenished. The authored demo is a workflow illustration, not a substitute for real evidence.

```bash
uv run --locked python -m signallab.synthetic_cli freeze --csv data/development.csv --final-csv data/reserved-final.csv --source 'Describe development origin and sampling' --final-source 'Describe separate reserved origin and sampling' --attest-final-reserved --max-parents 50 --threads 2
```

Replace example paths/descriptions with your actual inputs. Record the printed **PROTOCOL_ID**. The protocol pins input identities, ordered train/validation/test membership, source descriptions, final exclusion fingerprints, generation policy and feasibility question. It contains no raw final reviews or final labels. Existing records are never overwritten; a changed question/input/policy needs a new protocol, not edited JSON.

## Explicitly prepare, generate once, then inspect

The following commands contain placeholders. Replace `PROTOCOL_ID` and `PILOT_ID` with full printed IDs; do not type them literally.

```bash
uv run --locked python -m signallab.synthetic_cli prepare --protocol PROTOCOL_ID
uv run --locked python -m signallab.synthetic_cli generate --protocol PROTOCOL_ID --csv data/development.csv
uv run --locked python -m signallab.synthetic_cli inspect --pilot PILOT_ID --json
```

**Prepare is the download action.** Run it only when you intend to obtain the pinned public FLAN-T5-small assets. SignalLab has tested the loader with a tiny local T5 fixture; this repository does not yet establish the pinned real model's feasibility or quality. Generate is cache-only, CPU-only and supervised. A missing compatible preparation fails before a pilot starts; it does not silently download.

Keep the allocated pilot ID. The supervisor measures the full input including instructions/special tokens, skips inputs longer than 128 tokens intact, and permits one greedy output of at most 128 new tokens per selected training parent. It starts the wall clock before worker loading and samples worker RSS; the stop guard is 30 minutes or observed 4 GiB. This is sampled protection, not hard peak-memory containment. Unknown RSS cannot qualify. A failed/interrupted/stopped pilot remains evidence and is never automatically retried or resumed.

If you use root overrides, place them **before** each subcommand and keep the same roots throughout. The CLI's printed next commands preserve overrides.

## Review every generated output locally

Start the app on an unused local port; keep existing processes running:

```bash
uv run --locked streamlit run app.py --server.address 127.0.0.1 --server.port 8550
```

Open that address and choose **Synthetic Review Lab → Refresh local studies → protocol → pilot**. The sidebar dataset does not attach your research source. In **Development source**, upload the exact original development CSV; never upload the reserved final CSV. Maximum upload size is 10 MiB. A missing/mismatched source leaves numeric evidence readable but removes pairs and mutation controls.

Use Pending to inspect every generated candidate, including automatically excluded outputs. Compare the literal original training review against its candidate. The parent sentiment is read only. Choose:

- **Accept** only if meaning, polarity/negation, facts/numbers, English and meaningful rewording are preserved.
- **Reject** for a clear violation, refusal or instruction drift.
- **Ambiguous** when you cannot confidently establish preservation.

Give a specific reason, acknowledge the rubric and explicitly choose **Save decision and next pending**. There is no automatic semantic judgment or save. Negation, mixed praise/complaint, sarcasm, duration/price changes and decisive endings deserve particular attention. A human Accept never overrides blank/unchanged/duplicate/known-cohort/incomplete-output exclusions.

Synthetic artifacts persist generated text and reasons locally, including excluded outputs that copy original text. Exclusion is not redaction. No raw corpus upload/export is part of this workflow. Hashes identify evidence; they do not anonymise it.

## Seal the current qualified revision

Generation complete, human-review complete, qualified and currently sealed are different states. Qualification requires recorded supervised execution provenance, resource evidence passing the guard, all generated candidates reviewed, at least 30 accepted automatically eligible candidates and at least 10 per class. Legacy storage-only pilots remain inspectable but unqualified. Zero accepted candidates is an honest negative result, not an operational error or a reason to replenish parents.

With exact development data attached, inspect the status and choose **Seal accepted set**. If the review changed since the controls appeared, inspect it and explicitly **Reload seal context**. A seal binds one exact revision. Editing any review later makes it historical, even if the accepted set stays the same. No history is overwritten. The CLI alternative is:

```bash
uv run --locked python -m signallab.synthetic_cli seal --pilot PILOT_ID --csv data/development.csv --expected-revision REVIEW_REVISION
uv run --locked python -m signallab.synthetic_cli inspect-bundle --pilot PILOT_ID --seal SEAL_ID --csv data/development.csv --json
```

Retain the printed seal request ID. After an unconfirmed publication, only retry the unchanged request with `--seal-id SEAL_ID`; a changed revision needs a new identity. Inspection reports IDs, fixed partition/train/class counts and `no_training_has_run`, not review text or scores. It does not train. The training consumer rejects historical or inconsistent evidence.

## Optionally fit one exploratory classifier on the validated bundle

The following executable Python example uses full IDs and your exact source. Replace the placeholders and paths. It fits one TF-IDF/logistic-regression classifier on the bundle's augmented training frame, then evaluates only the frozen **validation** partition:

```python
import json
from signallab.study import load_study_csv
from signallab.synthetic import build_training_bundle
from signallab.core import train_baseline, evaluate

bundle = build_training_bundle(
    "artifacts/synthetic", "PILOT_ID", "SEAL_ID",
    load_study_csv("data/development.csv"),
    studies_root="artifacts/studies",
)
model = train_baseline(bundle.training)
validation = evaluate(model, bundle.splits.validation)
print(json.dumps({
    "protocol_id": bundle.protocol_id,
    "pilot_id": bundle.pilot_id,
    "seal_id": bundle.seal_id,
    "review_revision": bundle.review_revision,
    "training_rows": len(bundle.training),
    "validation": {key: validation[key] for key in ("n", "accuracy", "macro_f1")},
    "interpretation": "Exploratory single-condition validation; no augmentation benefit or final-evaluation claim.",
}, indent=2))
```

The original training/validation/test frames remain in `bundle.splits`; `bundle.training` is a separate augmented frame and `bundle.lineage` maps each row to its original parent and optional candidate. Never re-split it or pass it to existing raw-CSV `run_comparison`/transformer entry points. Keep test/final evaluation for the predeclared later study. This one-condition result cannot establish augmentation benefit; the matched-budget multi-condition study and multiple training seeds remain future work. The small unblinded review and model pretraining also limit interpretation.

## Recover without rewriting evidence

| Observation | What it means | Next step |
|---|---|---|
| Cache/preparation missing or incompatible | No generation should start. | Inspect environment/source identity; explicitly prepare a compatible snapshot if you intend to download. |
| Worker still running | Attempt counts are not a coherent snapshot yet. | Explicitly refresh; do not launch another owner or kill unrelated processes. |
| Interrupted owner and worker has exited | The pilot can only become interrupted evidence. | `uv run --locked python -m signallab.synthetic_cli recover --pilot PILOT_ID`; no resume or replenishment. |
| Resource status unknown/fail | Budget evidence cannot qualify. | Preserve the failure and explain it; do not relabel it complete or fabricate samples. |
| Source mismatch | Text/labels differ from frozen development input. | Reattach the retained original development CSV. Do not substitute final data. |
| Review conflict | Another session changed the revision; your proposal was not committed. | Copy the retained draft if needed, then Reload current decision. Reassess before explicit submission. |
| Save acknowledgement uncertain | Publication may have committed. | Retry the unchanged request with its original identity or explicitly reload current evidence. No success is assumed. |
| Historical seal | A later committed review exists. | Recheck qualification and explicitly reseal the current revision; old bytes remain intact. |
| Incomplete/corrupt/unsupported artifact | Safe readers cannot establish evidence integrity. | Preserve it for diagnosis. There is no automatic repair, pruning or supported manual hash-edit procedure. |
| Listing reaches 50-entry limit | The displayed list is bounded, not globally newest/complete. | Inspect a known full ID via CLI with the correct roots. |
| Review/seal allowance exhausted | Each pilot permits at most 200 review revisions and 200 seals. | Preserve history; do not prune it to continue. Reconsider the documented study design before any new pilot. |

For exact APIs, schemas, limits and failure boundaries see [Synthetic Review Lab contracts](synthetic-review.md). The [implementation validation map](synthetic-validation.md) separates tested software behavior from unvalidated real-model and human-quality gates.
