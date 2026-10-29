# SignalLab

A local machine-learning portfolio app for exploring customer reviews, training sentiment models and testing whether synthetic augmentation helps on held-out text.

## Run

Requires [uv](https://docs.astral.sh/uv/) and Python 3.11–3.13. Dependencies are locked in `uv.lock`.

```bash
cd /Users/thomas/Developer/Applications/signallab
uv sync --locked
uv run streamlit run app.py --server.address 127.0.0.1 --server.port 8537
```

Open http://127.0.0.1:8537. The authored 60-review demo works offline. It is a workflow demonstration, not a real-world benchmark. The app has Explore, Experiments, Synthetic Review Lab, Analysis, Saved runs, Case Study, Predict and Methodology workspaces. Predict offers the selected dataset's session-trained TF-IDF model or a saved local transformer. Switching datasets invalidates session baseline results; changing the augmentation slider marks old results as stale until rerun.

## Real reviews

```bash
uv run python -m signallab.fetch_data --limit 1000
```

This samples a balanced, deduplicated subset of the **training split** of [Amazon Polarity](https://huggingface.co/datasets/fancyzhx/amazon_polarity), using seed 42 and a streaming shuffle buffer of 10,000. The command preserves existing outputs; specify `--output data/another.csv` for another sample. The dashboard automatically offers `data/reviews.csv` as “Local review dataset.” Alternatively upload a UTF-8 CSV with `text,label` columns. Labels may be `0`/`negative` or `1`/`positive`. Use at least eight distinct examples of each class; neutral and missing labels are rejected. Optional extra columns are ignored.

The local learning app accepts at most 50,000 rows, 20,000 characters per review and 10 MB per upload. Uploaded files stay in memory and are not sent to hosted inference services. Downloaded datasets and weights contact Hugging Face. CSV exports include review text; numeric baseline/transformer reports exclude it. Synthetic pilot artifacts persist generated text and review reasons, including outputs that copy an original review. Review source dataset terms before redistributing public data.

## Experiment design

- Case/whitespace duplicates and known prefix-template variants are removed **before** splitting; conflicting duplicate labels are rejected.
- Fixed seed 42; stratified 60% train, 20% validation, 20% test, rounded to whole rows. Data ordering is canonicalised for reproducibility.
- Baseline: training-only TF-IDF word uni/bigrams + balanced logistic regression.
- Augmentation: prefix a reproducibly sampled training subset with “In my experience, ”. This intentionally weak template control adds examples, not independent semantic information. It is not LLM generation.
- Both runs share validation/test reviews. Report macro F1, accuracy, per-class precision/recall, confusion matrices and mistakes. Inspect validation for model development; test is an exploratory comparison, not an endlessly reusable final benchmark.
- Dataset hashes and split/configuration details travel with exported reports. A single seed, small sample or positive score difference does not establish significance. Semantic duplicates, repeated customers/products and pretrained-data contamination require further controls for publishable claims.

## Freeze a synthetic-study protocol

The protocol API freezes the exact development input, ordered 60/20/20 partitions, a deterministic training-only parent selection (up to 25 per class), and the approved generation/review/resource rules. This API does not download models or train anything. The generation CLI below consumes these protocols; the review page seals qualified evidence and the bundle consumer preserves the frozen partitions.

Supply two separate UTF-8 `text,label` CSVs, each at most 10 MiB, 50,000 rows and 20,000 characters per review, with at least eight distinct reviews per class. The second cohort must be separately sourced and declared reserved for final evaluation. The existing exploratory test partition is part of development, not a fresh final cohort. Canonical text overlap is rejected across cohorts even when labels differ or known prefix variants are added. Hashes cannot establish semantic independence, absence of pretraining contamination or whether someone previously inspected a cohort.

```python
from signallab.study import freeze_protocol, read_protocol, attach_development, load_study_csv, STUDIES_ROOT

protocol = freeze_protocol(
    "data/development.csv", "data/reserved-final.csv",
    source="Describe development source/version and sampling",
    final_source="Describe the separate reserved source/version and sampling",
    attest_final_reserved=True,  # Operator declaration; software cannot prove this.
)
stored = read_protocol(STUDIES_ROOT, protocol["protocol_id"])
splits = attach_development(stored, load_study_csv("data/development.csv"))
```

Run this using the project's locked environment (`uv run --locked python`). Provide your actual CSV paths and source descriptions; the example files are not supplied automatically. `attach_development` requires exact validated text and labels, restores the recorded order by ID without re-splitting, and returns independent frames. Small training pools can have fewer than 50 selected parents; they are not replenished and the feasibility threshold stays 30 accepted eligible examples including 10 per class.

Protocols are saved once under `artifacts/studies/<protocol-id>/protocol.json`, with private directory/file permissions on the local POSIX filesystem. They contain development labels/membership, exclusion hashes for both cohorts, source descriptions, settings and environment/source identities; original review text and final-cohort labels are not copied into the protocol. Hashes are identifiers, not anonymisation. Retain both original CSVs and the matching source/environment separately. Never place credentials or review text in source descriptions. Invalid input creates no study output. Completed protocols are never overwritten; failed publication may leave a preserved pending file/directory. Readers reject symlinks, malformed/oversized JSON, inconsistent identities and unsupported schemas. Preserve damaged files and create a new protocol rather than editing evidence. New protocol settings require a new identity; reading old evidence does not claim its source code or model assets were archived.

## Record synthetic pilot evidence

The next Python building block, `signallab.synthetic`, records fixed-order attempts, literal candidate outputs, automatic exclusions and an immutable terminal pilot manifest. Human reviews require the exact original development data and use revision checking, payload-bound retry IDs and append-only snapshots. A human Accept cannot override a duplicate/leakage exclusion; rejected or ambiguous decisions remain auditable. Abandoned pilots can only be finalized as interrupted, never resumed. Qualified seals and the fixed-partition handoff are available; controlled training remains later work.

These storage APIs do not load a generator or measure resources. The CLI below validates prepared assets, generates outputs, monitors its worker and records resource evidence. The native review page supports explicit human decisions and qualified sealing. See [the evidence API and recovery guide](docs/synthetic-review.md) for the actual interfaces and boundaries.

Unlike numeric baseline snapshots, `artifacts/synthetic/<pilot-id>/` persists **generated text and review reasons** locally. Treat these private files as data-bearing research artifacts. Original source rows and final-cohort labels are not separately copied. Generated outputs, including unchanged or known-cohort collisions, can reproduce original review text; exclusion is not redaction. Publication errors preserve pending evidence and never report success; incomplete review commits block further use until explicit recovery. No automatic repair, raw-corpus export or public upload is provided.

For the complete learner workflow, see [one inspectable synthetic-data experiment](docs/synthetic-experiment-walkthrough.md) and the [validation evidence and remaining research gates](docs/synthetic-validation.md). Synthetic Review Lab hands off a validated bundle; it does not automatically feed the existing Experiments CSV comparison.

## Synthetic generation CLI

Validated on local macOS with the existing locked dependencies. Preparation needs bandwidth/disk and generation uses CPU; no hosted inference or account is required for the configured public model. The application never starts these commands automatically. Model-preparation readiness has been tested with a tiny locally created T5 checkpoint; the pinned FLAN-T5 candidate and its paraphrase quality still need a separately authorized real pilot.

1. Supply your actual development and separately reserved final CSVs, then freeze their identities and source descriptions:

   ```bash
   uv run --locked python -m signallab.synthetic_cli freeze --csv data/development.csv --final-csv data/reserved-final.csv --source 'Describe development source and sampling' --final-source 'Describe separately reserved source and sampling' --attest-final-reserved
   ```

2. Run the printed **prepare** command when you intend to download. This is the only command that contacts Hugging Face. It requests the protocol-pinned `google/flan-t5-small` revision, copies only allowlisted fast-tokenizer/config/safetensors files to a new private snapshot, then runs a bounded local-only loading smoke test. Files alone do not mean ready. A missing/unsupported fast tokenizer or incompatible weights fails without installing dependencies or falling back to `.bin`/remote code. Incomplete preparations are preserved; an explicit retry creates a new preparation identity.
3. Run the printed **generate** command with your original development CSV path. It is cache-only, uses one CPU model worker, measures the complete prompt including special tokens, skips inputs over 128 tokens intact, and allows one greedy output of at most 128 new tokens per selected training parent. Selection/skips/rejections are not replenished. The pilot ID is printed before execution; loading and processed-parent messages contain no review text.
4. Inspect a saved pilot without model loading. Replace `PILOT_ID` with its full printed ID:

   ```bash
   uv run --locked python -m signallab.synthetic_cli inspect --pilot PILOT_ID
   uv run --locked python -m signallab.synthetic_cli inspect --pilot PILOT_ID --json
   ```

   Inspection reports operational status, selected/skipped/generated counts, resource observations and review counts, without candidate text or reasons. Generation completion is separate from human review/quality qualification. No training has run. Human review currently uses the Python evidence API; the Streamlit review workspace, qualified seals, `inspect-bundle` and controlled training follow in later tasks.
5. For an unfinished pilot after its worker has exited, use `recover --pilot PILOT_ID`. It acquires the generation lock and finalizes interrupted evidence with unknown resources; it never resumes generation or repairs review snapshots. An active inherited worker lock prevents recovery. Preserve failed artifacts; do not edit hashes or overwrite a completed pilot.

The sampled stop guard covers worker launch, model loading and generation: 30 minutes or an observed worker RSS of 4 GiB. It is not an OS hard-memory limit or a precise real-time guarantee. Missing RSS stops the pilot; earlier observations remain recorded and unknown measurement never qualifies. A watchdog exits the worker on supervisor-lifetime EOF; cleanup terminates/reaps only its owned child, escalating after two seconds if necessary. Downloads are outside the pilot clock. Snapshots and pilots bind asset checksums, source/runtime identities and protocol settings; modified assets or a changed preparation environment require an explicit new readiness check.

Use `--help` for each command. Optional `--studies-root`, `--prepared-root` and `--pilots-root` go **before** the subcommand; printed next commands preserve custom roots. Generation can select an exact `--prepared` snapshot when discovery is ambiguous or exceeds its 50-entry bound. Exit 0 means the command completed/was inspected, not that the feasibility threshold passed; generation failures/budget stops return 1, argument usage errors 2, and interruption before a managed pilot returns 130. A managed interrupted pilot returns 1 with its preserved terminal status. Existing baseline/transformer CSV trainers do not consume these synthetic corpora.

## Error analysis

After running a baseline comparison, open **Analysis**. Validation is selected by default. Compare model disagreements, inspect mistakes, filter by actual sentiment or literal text, and view accuracy with review counts by sentiment and fixed word-count bands (≤20, 21–80, >80). Small slices are descriptive, not significance tests. Filtered CSV downloads contain review text.

New baseline and transformer evaluation reports add `evidence_version: 1` and ordered `evidence` records containing exact SHA-256 review IDs, true labels and predictions. IDs hash the validated text and label; they are separate from the normalised dataset hash. Analysis joins by exact IDs and rejects mismatched partitions. Legacy transformer reports remain available as aggregate results but cannot support detailed analysis. New compatible transformer results at `artifacts/transformer/report.json` join the baseline comparison when the dataset matches. Other output directories are not discovered yet. Current-session analysis remains available; saved baseline comparisons can now be reopened after restarting the app. Report import, transformer history discovery and the broader research protocol remain roadmap work.

## Review synthetic candidates

Open **Synthetic Review Lab** in the sidebar, choose **Refresh local studies**, then a protocol and pilot. Summaries work without a CSV. Attach the exact original development CSV on this page to inspect literal training-parent/paraphrase pairs; never attach the reserved final source. Review every generated candidate, including automatic exclusions. Each explicit save needs a verdict, reason and rubric acknowledgement; it records an immutable revision and advances to the next pending candidate. Conflicts retain your draft and require **Reload current decision**.

Human acceptance and automatic eligibility are separate. Qualified sealing and the validated training bundle are available; this workspace does not train a model or establish performance gains. Generated text and review reasons are persisted locally. See [review workflow and recovery](docs/synthetic-review.md#streamlit-review-workspace-t4) for attachment limits, filters, concurrency and privacy details.

## Seal a qualified pilot and inspect its handoff

A qualified pilot needs completed supervised generation with recorded execution provenance, observed resources within the guard, every generated candidate reviewed, and at least 30 human-accepted automatically eligible outputs including 10 per class. Accept cannot override automatic exclusions. Older storage-only pilots remain inspectable but cannot qualify. These gates describe recorded pilot evidence; software cannot certify a reviewer's semantic judgment or prove classifier improvement.

After attaching the exact development source in **Synthetic Review Lab**, use **Seal accepted set**. The action pins the review revision you saw; concurrent edits require **Reload seal context**. A later review revision makes previous seals historical and the training builder rejects them. History is never overwritten. You can also seal from the terminal, replacing placeholders with actual IDs, revision and the original development path:

```bash
uv run --locked python -m signallab.synthetic_cli inspect --pilot PILOT_ID
uv run --locked python -m signallab.synthetic_cli seal --pilot PILOT_ID --csv data/development.csv --expected-revision REVIEW_REVISION
uv run --locked python -m signallab.synthetic_cli inspect-bundle --pilot PILOT_ID --seal SEAL_ID --csv data/development.csv --json
```

Seal requests print their ID before publication; retain it and pass `--seal-id SEAL_ID` only to retry an unchanged request after an unconfirmed response. A changed revision needs a new seal. Root overrides precede the subcommand. Inspection prints identities/counts, never review text, reasons or a flat augmented CSV. Model assets are not needed to seal or inspect existing evidence; no download or training runs in these commands.

The supported Python consumer returns independent original partitions, an augmented training frame and row-aligned lineage. It always re-reads disk evidence under lock and requires a current qualified seal:

```python
from signallab.study import load_study_csv
from signallab.synthetic import build_training_bundle
from signallab.core import train_baseline, evaluate

bundle = build_training_bundle(
    "artifacts/synthetic", "PILOT_ID", "SEAL_ID",
    load_study_csv("data/development.csv"),
)
# Optional explicit experiment: fit only this training frame.
model = train_baseline(bundle.training)
validation_report = evaluate(model, bundle.splits.validation)
```

This example is an explicit single-condition validation experiment, not the matched-budget controlled study or a final evaluation. Keep test reviews for their planned use; the separately reserved final CSV never enters the bundle. Do not pass `bundle.training` to `run_comparison` or the existing transformer CSV entry point: those functions re-split raw data and do not consume this contract. See [seals and fixed-partition handoff](docs/synthetic-review.md#qualified-seals-and-fixed-partition-handoff-t5) for currency, bounds and recovery details.

## Save and reopen an experiment

1. In **Experiments**, run a baseline comparison and choose **Save experiment**. Saving records the completed ratio, even if the slider has since changed.
2. Open **Saved runs** and choose **Refresh saved runs**. Numeric results and JSON downloads are available without a CSV upload. The table identifies Ready, Incomplete, Invalid, Legacy and Incompatible entries.
3. Choose **Inspect in Analysis**, then select the exact original dataset in the sidebar. Validation is the default partition. You can also select **Analysis → Saved experiment** and refresh there. An input/split mismatch blocks text-level analysis with a recovery message.

Completed baseline snapshots live at `artifacts/runs/<run-id>/report.json`. Every Save creates a new ID; completed records are never overwritten by the save API. Publication uses an exclusively created directory, a flushed/fsynced pending file and an atomic rename. A failed save leaves session results usable; a partial directory is shown as Incomplete on refresh. There is no delete or automatic migration action. An unsaved session report from an earlier app version needs a new baseline comparison before it can be saved. This is a local macOS/POSIX filesystem implementation; other platforms have not been validated.

Saved schema version 2 records include the dataset source label (older snapshots explicitly show Source not recorded), exact validated-input identity, ordered train/validation/test IDs and a split digest, preprocessing and split-policy versions, baseline parameters, dependency/Python versions and source hashes captured before model fitting. Git provenance is explicitly unknown. On read, metrics are recomputed from the evidence and checked for consistency; exact IDs bind review text only when the original dataset is reattached. These checks detect inconsistencies, not deliberate forgery by someone who can rewrite local files. A snapshot is not a frozen final-cohort protocol or a statistically fair augmentation study.

Snapshots persist labels, predictions and review hashes, **not review text or serialised models**. Hashes are identifiers rather than anonymisation; protect experiment files as local research data. No pickle/joblib model is loaded. Reopening a snapshot does not restore a Predict model: run a current-session comparison for new baseline predictions, or choose a separately saved transformer. Existing transformer reports and weights remain untouched; this history feature currently saves baseline comparisons only.

Discovery runs only after an explicit refresh. It checks at most 50 immediate directory entries and reads at most 16 MiB per report; a warning identifies a truncated listing. Entries are sorted by save time within that bounded listing, not advertised as the globally newest 50. Symlink roots, run directories and report files are not followed. Invalid/unsupported records are preserved. Future schema versions require their compatible app version; legacy numeric transformer results retain their existing aggregate view.

## Read and export a case study

1. Open **Case Study** and choose **Refresh saved runs**. If there are no completed snapshots, first run a baseline comparison in **Experiments**, then explicitly **Save experiment**. Nothing trains or saves automatically.
2. Select one saved baseline comparison. Validation is the default; test results are explicitly exploratory. Read the recorded source, paired results, signed macro-F1 difference (augmented minus real-only), accuracy change in percentage points, and counts of corrected, regressed, both-wrong and both-correct reviews. A tie or negative result is a valid outcome, not a failed workflow.
3. Optionally expand **Attach original reviews to inspect examples**, select the original source and enable attachment. Exact original CSV text and labels are required; the sidebar dataset alone does not attach anything. The app shows at most two examples per corrected/regressed/both-wrong category, ordered by exact review ID. They are illustrative, not a representative sample. Missing/mismatched data blocks previews while numeric evidence remains available.
4. Choose **Inspect this partition in Analysis** to retain the selected saved run and partition; select/upload the exact dataset there again if needed. Upload widgets do not promise persistence across workspaces.
5. After reading the limitations, choose **Download Markdown**. The <=1 MiB UTF-8 file contains the displayed snapshot's numeric results, provenance, content digest, warnings and recorded method/environment. Review text is always excluded, including when previews are attached. Downloading does not publish, create a draft or rewrite a saved run.

Case Study uses validated schema-2 baseline snapshots only. Legacy transformer aggregates remain in their existing views; they are not presented as directly comparable case-study evidence. Demo and unknown-source warnings remain visible and travel with the export. Hashes are identifiers, not anonymisation. Source hashes do not archive code; single-split descriptive differences do not prove significance or causality. The wider research protocol and synthetic pilot remain roadmap work.

Numeric stories work in a fresh session without a CSV, model weights or training. Selected records are re-read and validated on reruns; missing, replaced or corrupt records never reuse an old story/download. Preserve damaged files and save a new comparison, then refresh. Discovery keeps its existing 50-entry/16-MiB limits; it is not a global newest-run query.

To reproduce a baseline story, provide its exact original dataset, match the recorded environment/source and ratio, and follow **Experiments → Run baseline comparison → Save experiment**. Original upload locations were not recorded; the export cannot recover them. The `signallab.train` command below trains transformers, not the TF-IDF baseline. Warm/cold setup and study timing depend on your machine; no universal setup-time claim is made. Reuse a running local server or choose an unused `--server.port` (for example 8538); do not stop unrelated processes to free a port.

## PyTorch + Hugging Face

```bash
uv run python -m signallab.train --csv data/reviews.csv --epochs 2 --ratio 0.5
```

Downloads the base `distilbert/distilbert-base-uncased` tokenizer and safetensors weights, then fine-tunes **two separate classifiers** on CPU: real-only and template-augmented. The explicit PyTorch loop includes tokenisation/padding, AdamW, cross-entropy loss, gradient clipping, evaluation mode and best-checkpoint selection using validation macro F1. Both runs reset seed 42 before loading the model, so their initial classification heads match. Maximum sequence length is 128 tokens, batch size 8, learning rate 2e-5. `--batch-size` accepts integers 1–128; Python entry points and evaluation share the same bound, checked before output creation/model loading/training. CPU training can take several minutes; the UI never pretends that a run has completed.

Models/tokenizers are saved under `artifacts/transformer/{real,augmented}/`; `report.json` appears only after both runs complete. The dashboard shows this report only when its dataset hash matches the active data. Outputs are preserved: use `--output artifacts/another-run` for a new run. A failed run may leave incomplete checkpoints; retry with a new output directory. No Hub upload or external account is needed. Saved weights are available in Predict through the Saved transformer source.

## Predict with a saved transformer

Open **Predict → Saved transformer**, choose **Real-only transformer** or **Prefix-augmented transformer**, enter a review and choose **Analyse review**. This works in a fresh session without baseline training or a CSV upload. The model uses its own training dataset; changing the sidebar dataset does not change the saved classifier. Its dataset hash, seed, base checkpoint and experiment ratio are shown. Existing schema-1 reports are explicitly labelled as legacy with incomplete revision/input provenance.

The current form reads the completed report and `real`/`augmented` model directories under `artifacts/transformer`. Other `--output` directories are preserved but are not yet selectable in Predict. Missing, malformed or incompatible configuration, safetensors and tokenizer files produce a recoverable error; do not overwrite an existing experiment to repair it. Saved baseline history still contains numeric evidence rather than models.

Loading is lazy, CPU-only and local: `local_files_only=True`, `trust_remote_code=False`, safetensors and a fast tokenizer. Supported classifiers are binary BERT/DistilBERT with explicit NEGATIVE=0/POSITIVE=1 label maps. Incomplete weight loads are rejected instead of silently using a new random classifier head. File/config bounds and symlink checks apply before loading. Only one loaded classifier is cached; a lock covers cache replacement and inference. Artifact identity includes file names, inode, size and modification time; changes invalidate reuse. These are trusted local research artifacts, not an arbitrary model-upload sandbox.

Evaluation and inference share tokenisation, truncation at 128 tokens including special tokens, evaluation mode and softmax class ordering. Longer reviews show a truncation notice. Probabilities remain uncalibrated, and binary predictions cannot represent neutral/mixed sentiment. Prediction context is reconciled on every render, including navigation away and back, baseline retraining, source/model switches and artifact changes; these clear old results; failed or empty submissions do not retain the previous prediction. No model/data download is triggered by saved-model inference.

## Validate

```bash
uv run python -m unittest discover -s tests -v
uv run python -m compileall -q app.py signallab tests
```

Tests exercise validation, conflicting labels, deduplication/split isolation, deterministic augmentation, metrics, and the Streamlit explore/train/predict flow. The transformer integration test actually optimises, saves and evaluates a tiny local BERT offline; it verifies the training contract, not DistilBERT quality. Temporary test artifacts live under `.scratch/`.

Study contract tests also exercise cross-cohort text-only exclusions, frozen partition reattachment, deterministic balanced selection, schema/byte limits, private immutable publication, symlink rejection and concurrent publishers. They use authored fixtures and prove neither final-cohort independence nor real generator quality.

Synthetic evidence tests cover attempt/terminal binding, automatic exclusions, exact-source review, historical snapshots, idempotent retries, publication failures, two-process stale-write protection and an inherited generation lock held until child exit. They do not execute real generation or establish model/resource feasibility.

Generator tests additionally create a tiny local safetensors T5/fast tokenizer, run actual readiness and generation in a supervised offline child, reject incomplete/replaced weights, and verify genuine overlength inputs are skipped. Fake owned workers exercise wall/RSS stops, failed probes, malformed responses, interruption, kill escalation and supervisor death. Hub download calls are mocked; no production model download or semantic-quality claim is part of the suite.

Integrated journey tests exercise two independent app sessions, stale-save recovery, seal currency, source removal, preserved damaged evidence, and the executable learner-guide classifier example. See [validation evidence](docs/synthetic-validation.md) for the limits of those checks.

## Structure

- `app.py`: Streamlit workspace and dataset-scoped session state.
- `signallab/core.py`: validation, splits, augmentation, baseline and shared metrics.
- `signallab/study.py`: immutable synthetic-study protocols, text-only exclusions, fixed parent selection and exact frozen-partition reattachment.
- `signallab/synthetic.py`: bounded pilot/candidate evidence, immutable terminal manifests and atomic human-review history, qualified seals and current fixed-partition training bundles; no model execution.
- `signallab/synthetic_cli.py`: explicit freeze/prepare/generate/recover/inspect/seal/inspect-bundle commands, local asset checks and owned-worker supervision.
- `signallab/synthetic_ui.py`: native exact-source review, conflict recovery and sealing controls.
- `signallab/synthetic_worker.py`: bounded worker IPC and lifetime watchdog started before ML imports.
- `signallab/runs.py`: immutable baseline snapshots, bounded discovery and exact dataset reattachment.
- `signallab/case_study.py`: pure numeric stories, paired mistakes, separate exact-text previews and text-free Markdown.
- `signallab/case_study_ui.py`: native Case Study selection, optional attachment and Analysis handoff.
- `signallab/analysis.py`: exact-ID alignment and descriptive performance slices.
- `signallab/inference.py`: shared CPU tokenisation, evaluation probabilities and bounded local classifier loading/cache.
- `signallab/prediction_ui.py`: saved transformer source, provenance and prediction states.
- `signallab/train.py`: explicit PyTorch fine-tuning and experiment reports.
- `signallab/fetch_data.py`: bounded public review sampling.
- `signallab/demo.py`: clearly labelled authored fixtures.

TensorFlow is intentionally deferred until there is a specific framework-comparison question. Next useful experiments are meaningful training-only paraphrases with label review, multiple seeds with confidence intervals, and a separately reserved real evaluation set.
