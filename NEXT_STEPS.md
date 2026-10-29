<!-- /autoplan restore point: /Users/thomas/.gstack/projects/signallab/unknown-autoplan-restore-20261023-141354.md -->
# SignalLab: next steps

Status: APPROVED by user on 2026-10-23. Early case study and bounded local generator pilot accepted. Implementation remains separately scoped.
Date: 2026-10-23. Project: /Users/thomas/Developer/Applications/signallab.
Mode proposed by autoplan: SELECTIVE EXPANSION. This document is planning only.


## Read this first: reviewed recommendation

Build the next release around **traceable experiments and the actual saved transformer**. First freeze the research/evaluation protocol, then record compatible runs, enable saved-model prediction and publish a short negative-result case study. Follow with a bounded reviewed paraphrase pilot, then a fair repeated-seed study and a final case study. TensorFlow remains a later same-contract learning exercise.

The immediate work is stages 0–2; this is a complete useful release, not a promise to run the whole study at once. Full research requirements, deliberate exclusions, failure registries, all review sections and passes, test coverage and tasks appear below. Application code, dependency lockfile and model/data artifacts have not been modified during planning. Status: **APPROVED by user on 2026-10-23**.

## Original intent
Build an analytics and machine-learning learning portfolio using synthetic analysis, PyTorch, TensorFlow and Hugging Face. The user approved the current Python stack and built the customer-review sentiment direction. This milestone should make the learning demonstrable through reproducible experiments and understandable model behaviour.

## Confirmed premises
1. The main outcome remains a learning portfolio that explains the builder's ML/data-science reasoning; a commercial SaaS product is not yet requested.
2. Customer-review sentiment remains the domain, and the local Streamlit app remains the delivery format for this milestone.
3. Research quality and usable PyTorch/Hugging Face models come before adding more frameworks. TensorFlow stays on the roadmap rather than being silently removed.
4. Success means a visitor can inspect data quality, compare recorded runs, try the actual saved transformer, and understand a result including failure. Better scores are not promised.

## Verified current evidence
- app.py: Explore, Experiments, Predict and Methodology. The prediction form uses session-only baseline models, not saved transformers.
- signallab/core.py: fixed SEED=42, deterministic stratified 60/20/20 splits, training-only prefix augmentation, TF-IDF/logistic regression, per-class metrics and predictions.
- signallab/train.py: explicit CPU PyTorch loop, validation-based checkpoint selection, separate saved real/augmented classifiers. The same epoch count gives augmented training more optimiser updates; augmentation benefit is therefore confounded with training exposure.
- app.py reads only artifacts/transformer/report.json. CLI --output supports other directories that the dashboard cannot discover.
- Dataset hash and split sizes are stored, but explicit split membership, dependency/code/checkpoint revisions and training budgets are not a complete run contract.
- Saved one-epoch 200-review experiment: real-only DistilBERT validation/test macro F1 0.7333/0.7333; augmented 0.4813/0.3866. These are small exploratory samples, not benchmark claims.
- Eleven existing tests cover core data and Streamlit happy/error flows plus real offline tiny-BERT training. Further contract and CLI/report-corruption coverage needs review.
- No Git repository exists here. Remote, branch, git history and base diff are unavailable. Do not initialise Git, stage, commit or push as part of planning. Base branch fallback is main for skill bookkeeping only, not an existing branch.
- PRODUCT.md and DESIGN.md exist. No existing feature plan, CLAUDE.md or TODOS.md was found.

## Intake roadmap (historical draft; reviewed delivery order below is authoritative)

### Milestone 1: trustworthy recorded experiments
Unify baseline and transformer report metadata with a versioned manifest: run ID, source/provenance, dataset identity, split membership hashes, seed, model and checkpoint identity, augmentation method, training budget and dependency versions. Preserve old artifacts; read legacy reports as legacy rather than inventing missing fields. Publish completed reports only after validation, and distinguish incomplete, corrupt, incompatible and completed runs. Discover local run directories so CLI output and UI selection agree. Persist numeric baseline reports without unsafe model pickle loading.
Acceptance: same inputs produce traceable partitions; wrong-data/wrong-split reports cannot appear as directly comparable; partial/corrupt runs never masquerade as completed; the existing saved run remains readable as legacy.

### Milestone 2: use the saved transformer
Extend Predict with an explicit model/run selector and local safetensors/tokenizer loading. Keep baseline prediction available, label source dataset and run, and show uncalibrated probabilities honestly. Specify loading, missing checkpoint, incompatible labels, empty input and changed-dataset states. Cache by validated run identity, not just display name. Do not deserialize arbitrary pickle or execute remote model code.
Acceptance: saved-model predictions match evaluation/inference results for the same text; dataset/model switches cannot reuse stale predictions; offline inference works after artifacts exist.

### Milestone 3: meaningful controlled synthetic augmentation
Keep the prefix method as an explicit weak control. Add duplicate oversampling as a control and evaluate a bounded, local Hugging Face paraphrase candidate from training data only. Store parent lineage, generator identity, method and review status. Human-review generated examples for sentiment preservation; unreviewed or rejected examples must not enter accepted training experiments. Prevent source/descendant split leakage and duplicate test contamination. Decide generator/checkpoint/resource budget during review; no new dependencies or downloads are authorised by this plan.
Acceptance: training-only lineage is enforced; rejected/ambiguous examples stay excluded; real-only, duplicate, prefix and reviewed-paraphrase conditions can be compared without claiming synthetic benefit in advance.

### Milestone 4: credible evaluation
Make seeds configurable across baseline, split, augmentation, model initialisation and training. Use the same per-seed partition and declared training-budget policy across methods. Report optimiser updates/examples seen and matched-update comparisons so more data does not silently mean more training. Select settings on validation, predeclare primary metric, and freeze an additional untouched real evaluation cohort before final claims. Show per-seed results and paired uncertainty intervals with a precise explanation of what variation they capture; do not treat dependent split scores as independent observations. Start small and estimate CPU cost before larger runs.
Acceptance: reproducible seeds, auditable budgets, no model selection on final evaluation text, explicit inconclusive results, and failure examples linked to the correct evaluated model.

### Milestone 5: an explainable portfolio case study
Add a guided reading path and a local Markdown case study: question, dataset/provenance, baseline, augmentation hypothesis, controls, measured outcomes, mistakes and learning. Explain the observed negative augmentation result without asserting an untested cause. Document reproducible commands, hardware/budget and limitations. Design the later TensorFlow/Keras comparison around the same declared experimental contract; do not add a framework merely to enlarge the dependency list.
Acceptance: a new visitor can understand the question and evidence without reading code; a developer can rerun a bounded experiment from documented commands. Cold download and training time are measured separately from app-start time.

## Alternatives to review
A. Minimal continuation: saved-transformer prediction and a written case study. Lowest effort; leaves experiment tracking and synthetic-data evidence weak.
B. Research-first continuation (recommended): trustworthy runs, saved-model inference, reviewed synthetic controls, reproducible evaluation, case study. Reuses the current stack; exposes real ML reasoning; needs more experiment time.
C. Framework-first continuation: add TensorFlow/Keras next and compare implementations before expanding experiments. Broadens framework practice sooner; duplicates training work while current evaluation gaps remain.

## Scope constraints
Preserve current Streamlit visual language and local privacy boundary. No SaaS accounts, hosted generation, queues, database services, deployment, CI/CD, paid APIs, new production dependencies, Git changes or unrelated rewrite. Planning may identify future choices but does not authorise implementation or external writes.

## Review progress
- Intake and source inspection complete.
- Premise confirmation: CONFIRMED by user on 2026-10-23.
- CEO review: COMPLETE (single-model; planned remedies only).
- Design review: COMPLETE (plan/source review, not visual QA).
- Engineering review: COMPLETE; 22 coverage groups and test artifact written.
- Developer-experience review: COMPLETE; cold onboarding unmeasured.
- Final approval: APPROVED AS-IS by user on 2026-10-23.

## Decision Audit Trail
| # | Phase | Decision | Classification | Principle | Rationale | Rejected |
|---|---|---|---|---|---|---|
| 1 | Intake | Plan in the existing Python/Streamlit project | Mechanical | DRY | Reuse the implementation the user approved | Rebuild in a different stack |
| 2 | Intake | Include UI and developer-experience reviews | Mechanical | Completeness | Run selection, prediction states, CLI and reproducibility instructions all change | Backend-only review |
| 3 | Intake | Preserve implementation and existing artifacts | Mechanical | Explicit over clever | User requested next-step design, not implementation | Silent code changes or artifact replacement |
| 4 | Intake | Research-first ordering confirmed by user | User-confirmed | Pragmatic | User explicitly selected research-first on 2026-10-23 | Framework breadth first |

## Phase 1: CEO review

### Premise confirmation and system audit
The user confirmed the research-first portfolio direction on 2026-10-23: retain review sentiment, local Streamlit and PyTorch/Hugging Face; follow with TensorFlow. The actual outcome is demonstrating experimental reasoning through a usable application, rather than collecting technologies or guaranteeing a score increase. Doing nothing leaves a working toy whose transformer cannot be tried in the UI and whose augmentation result mixes data changes with extra optimiser updates.

Read PRODUCT.md, DESIGN.md, README.md, app.py, core.py, train.py, fetch_data.py and all three test files. There is no Git repository, so history, stash, remote platform, base diff, PR state and churn analysis are unavailable, not clean. No CLAUDE.md or pre-existing TODOS.md was found; no invented architecture or test commands are substituted. Existing style references are the sklearn pipeline, the shared prediction_report function and explicit PyTorch loop; avoid duplicating report assumptions in app.py or collapsing CLI failures into a generic message.

### 0A: premise challenge
P1, portfolio purpose: confirmed and supported by PRODUCT.md. P2, local review domain: confirmed, and the current code/data make it cheaper to improve evidence than to switch domain. P3, synthetic augmentation should improve performance: rejected as a premise; it is a hypothesis, and the saved negative result is valid portfolio material. P4, more frameworks demonstrate stronger ML knowledge: the user explicitly chose research-first ordering, so keep TensorFlow later without deleting that learning goal. P5, repeating a small held-out test validates generalisation: rejected; reserve a fresh cohort and describe remaining contamination/grouping limits.

### 0B: what already exists
| Subproblem | Owning code | Reuse decision |
|---|---|---|
| CSV/text/label contract and duplicate checks | core.validate_reviews, load_csv | Extend lineage-aware validation, keep one owner |
| Splitting and sampling | core.make_splits, augment_training | Accept explicit seed/protocol; preserve legacy behaviour |
| Baseline learning | core.train_baseline, run_comparison | Reuse pipeline and metrics; add report persistence |
| PyTorch learning and validation selection | train.fit_model, score | Parameterise budget/seed, reuse inference path |
| Public review sampling | fetch_data.fetch_reviews | Add provenance and predeclared independent cohort protocol |
| UI navigation and state invalidation | app.py sidebar and dataset_id boundary | Retain four workspaces; add local run selection |
| Offline model contract test | tests/test_transformer.py | Extend with save/load parity and publication failure |
| Error/prediction app flow | tests/test_app.py | Add stale-run and transformer states, not a new test framework |

### 0C: dream state and delta
```text
CURRENT                         THIS PLAN                         12-MONTH IDEAL
one exploratory split   --->    recorded, inspectable study --->   repeatable domain studies
prefix-only control             actual transformer inference      same protocol across frameworks
session baseline prediction     reviewed synthetic lineage        independently evaluated claims
```
The plan supplies a credible experiment and a readable story, not a general experiment-management platform. The twelve-month ideal adds replication across domains and TensorFlow only after the first protocol can be reproduced. Public pretraining contamination and absent customer/product grouping remain limitations even in that ideal; hashing does not solve them.

### 0C-bis: alternatives
| Approach | Completeness | Human / AI-assisted estimate | Risk | Pros | Cons | Reuse |
|---|---|---|---|---|---|---|
| A: prediction + early case study only | 6/10 | 1–2 days / 2–4h | Low | Visible result quickly; tiny change surface | Weak augmentation evidence; no durable comparisons | app/train/README |
| B: staged research-first plan | 9/10 | 5–8 days / 8–16h plus measured experiment time | Medium | Full hypothesis/control/evaluation story; no new service | Review effort and CPU cost; requires explicit study design | All current learning code |
| C: TensorFlow comparison first | 6/10 for confirmed research goal | 3–5 days / 5–8h plus training | Medium | Framework learning sooner; reproducible API comparison later | Duplicates unresolved protocol; new dependency approval | Data/metrics only |
Research-first B is the confirmed direction. Estimates are planning ranges, not timing measurements or promised AI speedups. B does not mean an all-at-once implementation: every stage must yield a usable local app and preserve old outputs.

### 0D: selective expansion and scope decisions
The first release is milestones 1–2 plus a short exploratory case study. Preserve the full later synthetic/evaluation work; this changes sequencing rather than cancelling features. Before any paraphrase work, write the final-cohort selection protocol and freeze its source/revision/selection; the old 200-review data and exposed scores remain development evidence.

Five adjacent improvements were examined: show why runs cannot be compared (accept), copyable reproduction commands (accept), an early negative-result narrative (accept), evidence-linked review progress (accept with augmentation), and measured CPU budget summaries (accept). MLflow was examined as the standard experiment-tracking alternative: its official documentation covers parameters, code versions, metrics and artifacts. For this app a small local immutable report contract reuses existing JSON without a new dependency/server; this is intentionally narrower, not an attempt to recreate MLflow.

A practical paraphrase pilot is capped at 50 training parents and one candidate per parent. Review all candidates with a sentiment/meaning rubric before acceptance; record rejected and ambiguous counts. Do not expand beyond the pilot until elapsed time, peak memory, acceptance and label-flip rates are measured. A failed pilot is a complete research result, not a reason to silently substitute the weak prefix method.

### 0E: temporal interrogation
| Human stage | AI-assisted work estimate | Decision resolved before implementation |
|---|---|---|
| Hour 1: foundations | 15–30m | Fixed protocol and run identity; legacy read policy; final cohort selection |
| Hours 2–3: core | 30–90m | Atomic report publication; strict compatibility; model-label semantics |
| Hours 4–5: integration | 30–90m | Run picker uses the same registry as CLI; cached model identity and stale prediction rules |
| Hour 6+: tests and writing | 1–2h | Offline fixtures, malformed reports, failure injection, case-study limitations |
These rows describe implementation coordination, not model-training speed. Generator and CPU training wall times are unknown until measured. 0F mode is SELECTIVE EXPANSION under autoplan; intermediate implementation choices follow its six principles.

### Section 1: architecture
The current small Python modules are sufficient; add one local runs module rather than a DB, queue or separate service. App, baseline and transformer code must use the same manifest validation instead of implementing subtly different compatibility checks. The first bottleneck is model memory/training time, not a database or web tier; local process failure must leave a visible incomplete run rather than false success.
```text
CSV/demo -> core validation -> frozen protocol/splits -> baseline or train.py
                                  |                         |
                                  v                         v
                              lineage IDs            runs.py publication
                                                            |
                                                      local artifacts
                                                            |
                                    app run picker -> validation -> Predict / Compare
```

### Section 2: error and rescue map
The CLI currently hides useful causes behind ValueError/OSError/RuntimeError, and UI report parsing is only lightly structured. Add named domain errors with safe, actionable context and exception chaining; never log raw reviews, credentials, environment values or unrestricted paths. Registry below defines each new failure contract; tests trigger it at the owning boundary.

| Codepath | Trigger / exception | Current coverage | Planned rescue and user sees |
|---|---|---|---|
| Manifest read | absent / FileNotFoundError | Missing fixed report has UI text | No runs yet; show the next command |
| Manifest parsing | JSONDecodeError, invalid field / ReportValidationError | Broad parser catch only | Mark corrupt/incompatible row, exclude comparison, preserve file |
| Run identity | duplicate ID / FileExistsError | Transformer output collision tested | Refuse overwrite; print new output suggestion |
| Artifact publication | disk-full/permission / OSError | Not tested | Non-zero exit; incomplete status; no complete report |
| Model resolve | missing weights/tokenizer / FileNotFoundError or OSError | Not tested | Disable unavailable model; name required local artifact |
| Model semantics | wrong head/label map / ModelCompatibilityError | Not tested | Reject model, never guess label order |
| Path access | escaped/symlinked artifact / UnsafeArtifactPath | No run browser yet | Reject path outside configured artifact root |
| Generator | unavailable checkpoint / OSError | New | Explicit download/setup recovery; preserve pilot progress |
| Generator output | empty/unchanged/duplicate/oversize / CandidateValidationError | New | Record rejected reason; never train automatically |
| Review save | conflicting row version / ReviewConflictError | New | Preserve accepted state; refresh candidate before retry |
| Protocol validation | parent in held-out partition / LineageLeakageError | New | Block experiment; show offending IDs only |
| Training | non-finite loss/OOM / RuntimeError | Non-finite check exists | Failed run with stage, safe cause, smaller pilot recovery |
| Statistical report | unpaired IDs or constant delta / ComparisonError | New | No interval for mismatched rows; constant intervals labelled correctly |

### Section 3: security and threat model
This remains a single-user local app bound to loopback; there is no new account/role boundary. Main risks are loading untrusted artifacts, resolving arbitrary paths and publishing review text in exported evidence. Likelihood/impact: path escape medium/high; unsafe model deserialisation medium/high; private review leakage medium/high; malformed numeric reports medium/medium.

Only local safetensors and supported tokenizer/config assets may be loaded, with trust_remote_code=False and local_files_only=True in prediction. Validate resolved paths stay inside the configured artifact root, including symlinks; do not accept arbitrary UI paths. Candidate text is data, not instructions to the application or permission to contact a hosted service; reports omit raw reviews by default, while explicit dataset exports disclose that they contain text.

### Section 4: data and interaction edges
```text
input -> validate -> train/generate/review -> publish -> list/select -> compare/predict
 |          |                 |               |             |                |
 missing    invalid type      upstream error  conflict      empty/stale      wrong run
 -> help    -> field error    -> failed stage -> preserve   -> refresh       -> clear result
 empty      conflicting label empty candidate disk full     partial         model missing
 -> reject  -> reject         -> review reject -> incomplete -> disabled     -> recover
```
Double-submit must not overwrite or launch duplicate runs with the same identity; disable the owning action while in flight. Navigation away does not cancel CLI training and the UI must say so; browser refresh may lose session-only baselines but durable reports remain. Dataset changes, run changes and model changes invalidate predictions before rendering any result; an incompatible run stays inspectable without enabling scientifically invalid comparisons.

### Section 5: code quality
Reuse core.prediction_report for all metrics and one inference batch function for evaluation and interactive transformer prediction. Avoid a generic experiment framework, abstract model interface hierarchy or duplicated UI schema parser. Separate small owning concerns in app.py only when introducing run selection; do not reformat or redesign existing unrelated sections.

### Section 6: tests
Existing tests exercise real tiny-BERT training but only assert output existence/shape; they do not prove saved-model inference matches evaluation. Add save/reload logits/probability parity, immutable publication with injected failure, legacy report handling, split/lineage mismatch rejection and the complete new run-selection/prediction flow. Generator quality cannot be established by a mocked unit test: use hand-labelled challenge cases and inspect the bounded real pilot separately from structural tests.

### Section 7: performance
The slow paths are generation, repeated transformer training and first model load; no measured p99 exists, so none is invented. Restrict the first pilot to 50 parents, one loaded model at a time, 128-token classification inputs, bounded batches and limited local run scanning. Cache inference by artifact content identity with max_entries=1; loading all four method models simultaneously is not acceptable for a CPU learning app.

### Section 8: observability
Record run ID, stage, seed, method, examples seen, optimiser updates, elapsed time and peak-memory measurement source. Persist started/completed/failed state and safe failure codes locally, not hosted traces or alerts. Display a link between a result and its reproduction command so a failed experiment can be understood weeks later without reading private text logs.

### Section 9: rollout and rollback
Roll out readers before new writers, then inference, then augmentation, then the study runner. Preserve v1 reports; show legacy provenance limitations rather than synthesising v2 metadata from guesswork. Rollback means selecting the old supported app version and legacy outputs, not rewriting artifacts; no Git operation, deployment pipeline or database migration is part of this plan.
```text
legacy reader -> v2 reader -> new writer -> Predict -> augmentation -> study
rollback: stop using new features -> retain artifacts -> select supported legacy outputs
```

### Section 10: long-term trajectory
A versioned local manifest is a reusable boundary for later TensorFlow experiments without binding them to a specific trainer. Local-only design is reversible (4/5); a particular hypothesis/final cohort is intentionally not reusable after exposure, so a new study requires a new protocol. Knowledge concentration is reduced through an early and final case study, with exact methods and unsuccessful results written down.

### Section 11: product design
Retain Explore, Experiments, Predict and Methodology; the run list belongs inside Experiments rather than creating another navigation system. Visitors should see the question and dataset first, selected run and comparability second, then metrics and mistakes. Design must specify loading/empty/invalid/partial states and mobile keyboard behaviour before implementation; the next phase covers those in detail.
```text
Explore -> Experiments (select compatible runs) -> Predict (same saved model)
                  |                                  |
                  +-> inspect mistakes -> Methodology/case study
```

### NOT in scope
TensorFlow now (user confirmed later); hosted generation (privacy boundary); MLflow/W&B (unapproved dependencies/service); worker queues/database (no demonstrated need); deployment/CI (not requested); model-quality guarantees (not supported); semantic deduplication claims (no validated method); distributed/GPU training (no local requirement); public data/model publication (external write); Git initialisation/commits (not requested).

### Failure modes registry
| Codepath | Failure | Rescue planned? | Test requirement | User-visible? | Local log |
|---|---|---|---|---|---|
| Report reader | Unknown schema/corrupt | Yes, preserve/disable | test_runs invalid cases | Explicit status | Error code + run ID |
| Writer | Partial/disk full/duplicate | Yes, no complete report | Inject failures/collision | Failed or incomplete | Stage |
| Inference | Wrong labels/stale weights | Yes, reject/invalidate | Save/load + model switch | Disabled/result cleared | Artifact hash |
| Generation | Label flip/empty/duplicate | Yes, review quarantine | Challenge eval + lineage | Reject/ambiguous | Review counts |
| Training | Extra update confound | Yes, matched-budget protocol | Count steps/examples | Budget alongside result | Counts |
| Evaluation | Final cohort exposed early | Yes, protocol before generation | Access-path tests | Development vs final label | Evaluation event |
| Statistics | Nonpaired/degenerate | Yes, reject or label | Known delta/zero difference | Inconclusive when needed | Cohort ID |
All seven failure categories are planned, not implemented. No silent critical gap is accepted in the plan; these tests block the stage that adds the corresponding behaviour.

### CEO completion summary
| Item | Result |
|---|---|
| Mode / system audit | SELECTIVE EXPANSION; no Git baseline; current code inspected |
| Step 0 | Premises confirmed; B selected; preserve learning purpose |
| Architecture | 2 issues: duplicate report contracts and fixed output discovery |
| Errors | 13 codepaths mapped; recovery and tests specified |
| Security | 4 risks; 3 high-impact; no added service/auth surface |
| Data/UX | 6 interaction categories mapped; missing states assigned to Design |
| Quality | 2 reuse decisions; no generic platform abstraction |
| Tests | 6 high-value coverage categories; expanded in Eng artifact |
| Performance | 3 slow paths; bounded pilot; no fabricated latency |
| Observability | 2 gaps: lineage/provenance and stage/resource evidence |
| Rollout | Reader-before-writer and artifact-preserving rollback |
| Future | Reversibility 4/5; final cohort consumption explicit |
| Design | 3 hierarchy/state/navigation gaps for next phase |
| Required artifacts | Leverage map, dream delta, 13-row rescue registry, 7-row failure registry, 4 diagrams |
| Scope scan | 5 adjacent improvements accepted; 10 exclusions; deferred items recorded later in TODOS.md |
| Completeness | All 11 sections examined; selected complete boundary/error paths |
| Voices | Added below; Claude unavailable, no cross-model consensus claim |

### CEO independent voices
**CODEX SUBAGENT (strategic independence):** Six dimensions reviewed. Findings: freeze the final cohort before generation (high); make the hypothesis and early case study explicit (medium); bound generator feasibility (medium); release in stages and differentiate through evidence (medium). Integrated as protocol-first sequencing, an early short case study, and a bounded pilot. These refine the confirmed direction rather than remove requested features.

**CLAUDE SUBAGENT:** unavailable. `claude auth status` returned loggedIn=false. The available subagent is Codex, not a substitute Claude model.

**CODEX SAYS (CEO CLI):** unavailable. The CLI auth preflight passed, but the actual read-only invocation exited 1 because its configured gpt-6.1-sol model is unsupported by the signed-in ChatGPT account. Do not change model/account settings or repeatedly retry the same infrastructure failure. Later phases use independent Codex subagents and primary review, labelled single-model.

| Dimension | Claude | Codex CLI | Independent Codex subagent | Cross-model consensus |
|---|---|---|---|---|
| Premises valid | N/A | N/A | Sound, user confirmed | N/A |
| Right problem | N/A | N/A | Sharpen hypothesis | N/A |
| Scope calibration | N/A | N/A | Stage releases | N/A |
| Alternatives | N/A | N/A | B fits confirmed goal | N/A |
| Competitive risk | N/A | N/A | Differentiate through evidence | N/A |
| Six-month trajectory | N/A | N/A | Freeze final cohort first | N/A |
Confirmed cross-model dimensions: 0/6; missing voices are not agreement. No user challenge claimed. Same-model reviewer findings are advisory evidence, not a mandate to change the user's direction.

### Decision 5: Freeze final-cohort protocol before generation

| # | Phase | Decision | Classification | Principle | Rationale | Rejected |
|---|---|---|---|---|---|---|
| 5 | CEO | Freeze final-cohort protocol before generation | Mechanical | Completeness | Prevents the later study from consuming its final evaluation while developing synthetic methods | Reserve it only at milestone 4 |

### Decision 6: Publish a short exploratory case study in the first release

| # | Phase | Decision | Classification | Principle | Rationale | Rejected |
|---|---|---|---|---|---|---|
| 6 | CEO | Publish a short exploratory case study in the first release | Taste | Bias toward action | Makes learning visible while preserving the complete later study | Wait for all infrastructure |

### Decision 7: Limit initial generation to 50 parents and one candidate each

| # | Phase | Decision | Classification | Principle | Rationale | Rejected |
|---|---|---|---|---|---|---|
| 7 | CEO | Limit initial generation to 50 parents and one candidate each | Taste | Pragmatic | CPU feasibility and review quality need evidence before scaling | Run the complete matrix immediately |

### Decision 8: Use local immutable reports, not a tracking service

| # | Phase | Decision | Classification | Principle | Rationale | Rejected |
|---|---|---|---|---|---|---|
| 8 | CEO | Use local immutable reports, not a tracking service | Mechanical | DRY | Existing JSON and local privacy boundary cover current needs | Add MLflow dependency/server |

### Decision 9: Report same-model review degradation explicitly

| # | Phase | Decision | Classification | Principle | Rationale | Rejected |
|---|---|---|---|---|---|---|
| 9 | CEO | Report same-model review degradation explicitly | Mechanical | Explicit over clever | Claude is logged out and Codex CLI model is rejected | Fabricate cross-model consensus |

## Phase 2: design review

### Scope, incumbent system and scoring
This is an operational Streamlit workspace, not a landing-page redesign. DESIGN.md and .streamlit/config.toml are authoritative: native controls, four sidebar workspaces, restrained green/red chart series with text labels, and shared spacing. Initial draft design completeness was 5/10: features were named but run ownership, review flow and small-screen behaviour were not specified. After the decisions below the written specification is 8.6/10; that rates plan completeness, not rendered UI quality or accessibility compliance.

The configured gstack design binary path is absent. A separate installation has a binary, but no image-generation call is needed to redesign an existing native interface; the plan uses an explicit text wireframe and state tables. No mockups were generated or approved. A visual and keyboard pass on the implemented app remains required before claiming the UI works.

### Pass 1: information architecture, 5 → 9
Preserve the existing navigation and put durable run browsing inside Experiments. Its first three information priorities are the research question, selected run/cohort/budget identity, and the comparison outcome with limitations; configuration details go in an expander. Predict must not depend on an active upload being valid: saved models own their training context, and deliberate inference on new review text is valid even without the original CSV.

```text
SIDEBAR: SignalLab | Explore | Experiments | Predict | Methodology
Explore:     selected data -> quality summary -> filters -> review table
Experiments: question -> saved runs table -> compare eligibility -> metrics -> mistakes
             [New baseline experiment]        [Training command]
Predict:     model/run picker -> trained-on label -> review form -> result/probabilities
Methodology: study question -> exploratory case study -> protocol/limits -> reproduction

Within Experiments, augmentation review:
review queue -> parent + candidate pair -> accept / reject / ambiguous -> progress
```
The saved-run table shows run ID, model, method, status, cohort ID, budget and created time. Default to the latest compatible completed run only when that selection is labelled; never automatically compare two merely because their dataset hashes match. Legacy/incomplete/incompatible runs remain inspectable, but unsupported comparisons and unavailable inference actions are disabled with a precise reason.

### Pass 2: interaction states, 5 → 9
| Feature | Loading | Empty | Error | Success | Partial / stale |
|---|---|---|---|---|---|
| Saved-run list | “Reading local experiments…” | “No saved experiments. Run a baseline or copy the training command.” | Per-run corrupt/unsupported status; valid rows remain | Selectable run summary | Incomplete row lists last stage; Refresh action |
| Comparison | “Checking cohort and budget…” | “Select two compatible runs.” | Explain exact mismatch, no computed delta | Paired metrics, budget, mistakes | Legacy scores shown separately; comparison unavailable |
| Transformer load | “Loading selected local model…” | “No saved transformer. Train one locally.” | Missing artifact, unsafe path or wrong label map + recovery | Model/run and trained-on identity visible | Changed file invalidates selection/result before use |
| Prediction | Submit disabled while evaluating | Visible label and example prompt, no fake result | Empty/oversize input or model failure; result cleared | Label, per-class probabilities, submitted model identity | Pending edits say “Result is for the last submitted review” |
| Candidate review | “Loading candidate…” | “No candidates yet” or “All candidates reviewed” | Conflict/rejected output reason, preserve earlier decisions | Accepted/rejected/ambiguous count and Next | Save failure leaves draft visible but not accepted for training |
| Training command | CLI prints stage/run ID | Valid command and offline prerequisites | Non-zero exit, code/cause/recovery | Published report and output path | Existing incomplete directory preserved; new run required |
| Final evaluation | Clearly separate protocol action | “No frozen study protocol” | Refuse unfrozen/changed recipe or wrong cohort | Final report marked consumed/exposed | Later changes require a new cohort, not relabelled final scores |

Use native forms for prediction and candidate decisions so reruns do not accidentally submit partially edited input. Double-clicks must be idempotent at the run/review owner even if the button is disabled. CLI training survives navigating away because it is a separate process; do not show a browser cancel action that cannot cancel it.

### Pass 3: journey and emotional arc, 5 → 8
| Stage | Visitor action | Intended feeling | Specific support |
|---|---|---|---|
| First five seconds | Read the experiment question | Oriented | One question and explicit demo/real label |
| First minute | Inspect one saved run | Trust | Dataset/cohort/method/budget beside metrics |
| Five minutes | Compare methods and inspect errors | Curious, able to challenge | Incompatible-run explanation; no “winner” on inconclusive evidence |
| Try it | Submit a mixed review | Understand model limits | Actual saved model identity and uncalibrated probabilities |
| Learn | Read negative-result case study | Confidence in the reasoning | Hypothesis, controls, failed method and what remains unknown |
| Return | Reopen app and choose run | Continuity | Durable reports, predictable selection, no phantom session model |

The early case study gives a complete path before synthetic generation is ready. Long-term learning is reflected in a documented change of hypothesis or method, not decorative progress badges. A real novice usability check is still needed; the emotional responses above are design intent rather than observed user feedback.

### Pass 4: specificity and generic-pattern risk, 8 → 9
No new hero, card grid, icon system or ornamental chart is planned. Use run rows, explicit labels, native expanders and a side-by-side parent/candidate reading task; each section has one job. The inherited native font and theme take precedence over the review skill's generic preference for custom fonts because the user required the existing visual language and no redesign.

Litmus: product identity yes (SignalLab/sidebar), strong anchor yes (the comparison table), headings scannable yes, one section/one task yes, decorative cards necessary no, motion necessary no, usable without shadows yes. These are specification assessments, not screenshot approval. The only “positive” colouring denotes sentiment; augmentation improvements/declines also carry numeric signs and words.

### Pass 5: design-system alignment, 9 → 9
Reuse theme primary #246657, background #FAFAF7, secondary #EEF1EC and text #202C28; existing negative series #A13E37 retains an explicit Negative label. Use st.dataframe, st.form, st.selectbox, st.info/warning/error and native expanders before introducing components. Shared surfaces remain closed units; parent columns/gap handle layout rather than CSS overrides.

No design-system changes are proposed. Report statuses need text (“Legacy”, “Incomplete”, “Incompatible”), never colour alone. Charts have an adjacent data table or text equivalent so exact values remain accessible.

### Pass 6: responsive and accessibility, 2 → 8
At 1280px, show the table as the main workspace, with confusion/per-class details in two columns; keep the sidebar navigation. At 768px, prioritise run/model/method/status in the table and move provenance detail into a labelled expander. At 375px, use native collapsible navigation, one-column review pairs, sequential matrix/metrics sections, and horizontally scrollable tables contained inside the content area; no horizontal document scrolling or truncated primary action.

All inputs retain visible labels. Test tab order from workspace to dataset/run selector to form to submit; Enter submits only the active form and visible errors can be reached by keyboard. Verify body contrast ≥4.5:1, visible focus, 44px touch targets, 200% zoom, headings, radio groups and status announcements on the implemented native controls; do not assume Streamlit alone guarantees compliance. If a native primitive falls short, use its semantic API first and document a small local remedy rather than restyling the app.

### Pass 7: remaining decisions, 4 → 8
Saved-run browsing is decoupled from active dataset validation; data preparation errors should block new training, not reading a valid saved model. Final-cohort results stay outside exploratory comparison until explicit protocol completion, with an irreversible-in-study “already exposed” label rather than claiming a filesystem security boundary. Parent and candidate text are shown together for review; selection records the candidate hash and revision so a changed text cannot inherit an old approval.

The user approved both scope/taste recommendations on 2026-10-23: publish a short first-release case study before the full study, and gate expanded generation on the 50-parent local pilot. No decision is delegated vaguely to the implementer. Generator model performance, responsive rendering and novice comprehension remain empirical uncertainties to measure, not unresolved CSS choices.

### Design completion summary
| Pass | Initial → plan score | What would make it 10 |
|---|---|---|
| Information architecture | 5 → 9 | Test run selection with an unfamiliar visitor |
| State coverage | 5 → 9 | Verify all states in AppTest and live app |
| Journey | 5 → 8 | Observe learner comprehension |
| Specificity | 8 → 9 | Confirm full implementation uses native patterns |
| Design system | 9 → 9 | Render at intended sizes and inspect |
| Responsive/accessibility | 2 → 8 | Keyboard, zoom, 375/768/1280px verification |
| Decisions | 4 → 8 | User approved taste choices; pilot feasibility remains to be measured |
NOT in scope: visual redesign, custom font/assets, new navigation framework, image-generated mockup approval, invented user research. Existing leverage: four workspaces, native form/feedback components, theme and charts. Overall written plan 8.6/10, not visual QA; 0 approved mockups. Detailed run/model/state requirements are implementation tasks, not completed features.

### Decision 10: Keep four native workspaces and put run browsing inside Experiments

| # | Phase | Decision | Classification | Principle | Rationale | Rejected |
|---|---|---|---|---|---|---|
| 10 | Design | Keep four native workspaces and put run browsing inside Experiments | Mechanical | DRY | Existing navigation fits the learning journey | New dashboard shell |

### Decision 11: Decouple saved-model use from active CSV validation

| # | Phase | Decision | Classification | Principle | Rationale | Rejected |
|---|---|---|---|---|---|---|
| 11 | Design | Decouple saved-model use from active CSV validation | Mechanical | Completeness | Missing original data must not block valid local inference | Gate all workspaces on dataset upload |

### Decision 12: Invalidate prediction on model/run/artifact changes

| # | Phase | Decision | Classification | Principle | Rationale | Rejected |
|---|---|---|---|---|---|---|
| 12 | Design | Invalidate prediction on model/run/artifact changes | Mechanical | Explicit over clever | Prevent stale text or stale-model results | Keep last result without identity |

### Decision 13: Specify 375/768/1280px and keyboard states

| # | Phase | Decision | Classification | Principle | Rationale | Rejected |
|---|---|---|---|---|---|---|
| 13 | Design | Specify 375/768/1280px and keyboard states | Mechanical | Completeness | Native widgets still require verification | Assume responsive accessibility |

### Design independent review and resolutions
The independent Codex design reviewer identified seven findings: three high (dataset gating, run discoverability/capabilities, early orientation) and four medium (submitted-text identity, contextual recovery, review flow, runtime accessibility). The state table and information hierarchy above resolve them in the specification. Further refinement: represent report-readable, comparable and prediction-available as separate validated capabilities, not one overloaded status. A result includes a short excerpt of the submitted review in session-only UI state; never export it silently in the run manifest.

Run selection is page-local in Experiments and Predict, with an immutable shared run ID only when deliberately carrying the selection between them. Candidate review lives within Experiments. Recovery instructions name the selected run and preserve the old artifact rather than suggesting training overwrites or repairs it.

| Design litmus | Claude | Codex CLI | Independent Codex subagent / primary resolution | Consensus |
|---|---|---|---|---|
| Product identity | N/A | N/A | Preserve existing SignalLab/sidebar | N/A |
| Strong anchor | N/A | N/A | Question and comparison table | N/A |
| Scannable headings | N/A | N/A | Question/outcome before technical metadata | N/A |
| One job per section | N/A | N/A | Page-local selection, review inside Experiments | N/A |
| Cards justified | N/A | N/A | No new decorative cards | N/A |
| Motion useful | N/A | N/A | No added motion needed for this task | N/A |
| No shadow dependency | N/A | N/A | Native table/form hierarchy | N/A |
Claude remains unavailable; CLI failure is the established unsupported-model condition. Cross-model confirmed 0/7, not a two-model design approval. No independent reviewer has performed rendered UI QA.

### Decision 14: Separate readable, comparable and inference-ready capabilities

| # | Phase | Decision | Classification | Principle | Rationale | Rejected |
|---|---|---|---|---|---|---|
| 14 | Design | Separate readable, comparable and inference-ready capabilities | Mechanical | Explicit over clever | A run may have valid metrics but missing weights or unverifiable splits | Single overloaded completed flag |

## Phase 3: engineering review

### Step 0: scope and reuse challenge
Read core.py, train.py, fetch_data.py, app.py and all tests again against the proposed boundaries. The current code has one global SEED and normalized dataset identity, position-only prediction lists, a fixed UI report path, epoch-based validation, and no generator/review workflow. Reuse sklearn's Pipeline, torch DataLoader/inference_mode, existing metric code and Streamlit's native state/caching; do not add an experiment service or trainer hierarchy.

The entire roadmap exceeds eight files, so it must not become one unreviewable patch. This is not a scope cut: sequence milestone 1 (identity/report/protocol), milestone 2 (inference/early case study), milestone 3 (synthesis/review), then milestone 4 (controlled study/final case study). Each patch has one contract change, both producer and consumer, its tests and documentation; no stage is called done while its directly coupled UI or CLI is inconsistent. No new runtime classes/services are required beyond small explicit dataclasses and named exceptions.

### Section 1: architecture and precise contracts
```text
                                   [new protocol + immutable run metadata]
CSV/demo -> core.validate_reviews -> core.make_splits -> baseline / train.fit_model
               |                         |                    |
      exact row IDs + dedup groups       |              shared prediction_report
               |                         |                    |
               +-> synthesis.py ---------+-> reviewed lineage  |
                         |                                    v
                   review decisions                    runs.py publish/read
                                                              |
                                      artifacts/runs/<run-id> + legacy reader
                                                              |
                                app.py: list -> validate capabilities -> compare
                                                     |
                                               inference.py -> Predict
```
New modules: runs.py owns local artifact schema/publication/discovery; inference.py owns safe saved classifier loading and inference; synthesis.py owns candidate generation, lineage and review eligibility. Keep methods flat, avoid wrappers around every existing function. Later evaluation orchestration may be a small study.py CLI using these same contracts. No background job system, HTTP API, authentication subsystem, DB, network inference or deployment is introduced.

**Identity contract.** Keep normalized dedup groups separate from exact model-input identity. A v2 sample_id hashes exact validated text and label plus a preprocessing version, not casefolded text. The run's input fingerprint hashes canonical ordered exact sample records; retain source file digest and optional Git commit, or source-file digests when Git is unavailable. Dataset reorder does not change canonical partition membership; case/whitespace changes remain traceable even if the chosen model happens to tokenize them identically.

**Split/evidence contract.** Store ordered train/validation/development-test sample IDs, truth/prediction pairs and protocol ID. Direct comparison requires identical ordered evaluation ID/truth records and compatible protocol; never attach a stored vector to whichever active frame happens to have the same length. Join mistake text only by exact sample ID with the matching source dataset. If the source CSV is absent, reports still show counts/matrices/IDs and explain how to recover text; do not reconstruct or invent reviews.

**Schema contract.** A v2 run includes schema_version, immutable run_id, timestamps, safe state/error code, data provenance, exact input ID, split protocol, preprocessing/tokenizer/checkpoint identifiers and resolved revisions, code/dependency digests, model label semantics, augmentation method/parents/review revision, random seeds, training budget/schedule, per-partition evidence, metrics and model-file checksums. Truth and predictions must be binary; scores finite/in range; confusion matrix totals must reconcile. Any unavailable field is explicit, not silently filled from the current environment.

**Compatibility contract.** Readable, directly comparable and inference-ready are independent booleans with reasons. Across model families show shared-cohort benchmarks, without implying identical optimisation recipes; a controlled augmentation comparison within a model family requires identical protocol, initial checkpoint, preprocessing, training budget/validation opportunities and approved parent subset except the declared augmentation variable. Legacy reports are readable and may support safe inference from a validated local artifact, but cannot acquire v2 comparison claims from missing metadata.

**Paths/publication contract.** Default root is artifacts; legacy artifacts/transformer remains discoverable and new runs use artifacts/runs/<run-id>. Preserve --output compatibility, but an output outside the configured artifact root is not silently discovered; the command prints that fact and a root-local example. Additional roots require an explicit local configuration, never arbitrary manifest paths or filesystem-wide scans. Resolve all artifact paths relative to the run directory and verify containment, rejecting traversal, symlink escapes and invalid run IDs.

Create a fresh exclusive run directory, record local stage state, write models, compute checksums and validate all evidence, then atomically publish report.json within that owned directory. Only a validated published report is complete; failure never overwrites an existing report. Readers see incomplete until publication. This is crash-safe publication, not automatic resume or a claim of cross-filesystem transactional behaviour.

**Inference contract.** Load supported binary sequence-classification configs with trust_remote_code=False, use_safetensors=True and local_files_only=True. Verify head dimensions and label mapping (not merely index 0/1), tokenizer revision/max-length and artifact checksums before exposing Predict. Factor a single logits-to-probabilities/prediction path used by score and interactive inference; lock cached inference if shared globally. Cache at most one model keyed by complete artifact identity; do not cache user review text globally.

### Controlled study and generation protocol
Write the study question before synthesis: “Does reviewed paraphrasing improve binary review sentiment beyond repeated exposure under a declared training budget?” Fix development partitions once with split seed 42; vary training/augmentation seeds [13,42,87] within the fixed training pool, not by repeatedly redefining the final cohort. The already exposed 200-review dataset remains development evidence. Reserve a distinct, predeclared real cohort before generation, record source revision and sampling rules, and keep its content/labels out of generation, review, tuning, UI search and model selection until the explicit final evaluation command.

Use a primary matched-update budget, with the same batch size, maximum updates, validation-every-K-updates schedule, final validation step and tie rule (earliest maximal validation macro F1) across augmentation conditions. Record all evaluated update numbers and the selected step, actual examples seen, padding/truncation, elapsed time and peak-memory method. Equal-epoch runs may be a clearly labelled secondary analysis; do not interpret their differences as augmentation benefit separated from extra training. Logistic regression comparisons record solver/configuration rather than pretending their convergence iterations equal transformer update units.

Candidate generator for the bounded local pilot: google/flan-t5-small with a pinned resolved revision, fast tokenizer and safetensors, subject to model-card review and compatibility with existing approved dependencies. Use AutoModelForSeq2SeqLM on CPU, no device_map/accelerate or hosted API. No new package is silently installed: if the saved fast tokenizer cannot load with the current lockfile, stop with an actionable dependency decision or choose a compatible local candidate during implementation. The model card supports CPU use; this is a proposed generator, not a verified paraphrase-quality claim.

Pilot: at most 50 training parents, one candidate each, bounded 128-token inputs including instruction and 128 new output tokens, deterministic decoding/config recorded. Skip parents that cannot fit intact; never silently truncate away the sentiment-bearing clause. Initial stop budget is 30 minutes elapsed or 4 GiB model-process peak RSS, whichever comes first, measured on the actual machine; these are proposed caps, not guarantees that Python can recover from an OS kill. Candidate review requires preserved polarity, unchanged factual claims, readable English and meaningful rewording; accept/reject/ambiguous with a reason and immutable candidate hash. Unknown/refusal/instruction-like/empty/unchanged/duplicate outputs are rejected rather than repaired into training data automatically.

A pilot proceeds to the controlled study only if at least 30 candidates are accepted, at least 10 per class, and resource caps held. If these criteria are not met, record a completed feasibility result and stop expansion; changing generator or rubric is a new development experiment, not an excuse to edit rejected evidence. For duplicate/prefix/paraphrase controls use the same accepted parent IDs and counts to separate parent selection from transformation; publish attempt/acceptance/skip counts so selection bias is visible. These feasibility thresholds do not imply statistical power or quality significance.

For uncertainty, report seed-to-seed spread descriptively. On the one frozen final cohort use paired bootstrap resampling of the same sample IDs for the predeclared primary model pair (for example 1000 resamples, interval seed fixed in the protocol); call it a conditional interval for these trained models/cohort, not independent proof across training seeds or domains. Respect groups if available; when product/customer group IDs are absent, disclose that dependence is unresolved. Reject mismatched IDs, missing predictions, NaNs and invalid labels; handle zero differences/degenerate resamples explicitly. Predeclare the pair/metric before final evaluation and do not search across many pairs for a winner.

### Section 2: code quality
One shared run reader/validator must serve both app and CLIs; model-family-specific behaviour stays in current training/inference functions rather than a generic experiment framework. Extend parameters in core.make_splits, augment_training, train_baseline and train.seed_everything together so a report never advertises a seed that a producer ignores. Keep the v1 hash implementation readable for legacy identity, but new writers use versioned exact-input identity; a single ambiguous dataset_id function must not pretend both semantics are interchangeable.

Replace collapsed CLI errors with safe named codes and actionable messages; chain internal exceptions for debugging without dumping uploaded review rows or environment variables. Strictly validate reports before pandas allocation and model configs before model construction. No broad exception fallback should turn an invalid model/report into a successful prediction or a default label.

### Section 3: test review and diagram
Framework detected from tests and README: standard-library unittest, Streamlit AppTest and a real tiny offline Hugging Face/PyTorch fixture. Rerun during this review: `python -m unittest discover -s tests -v`, all 11 passed. Those tests establish current behaviour only; none proves planned run discovery, paraphrase quality or final-cohort isolation is implemented.

```text
NEW ENTRY / EXECUTION BRANCH                      COVERAGE OWNER / REQUIRED SIGNAL
core: exact IDs vs dedup groups ----------------- test_core: case/whitespace/reorder/conflict
core: explicit protocol/seed splits ------------- test_core: disjoint IDs, seed plumbing
runs: absent/legacy/v2/unknown/corrupt ------------ test_runs: capability + schema branch matrix
runs: exact ordered evidence -------------------- test_runs: permutations/missing/duplicate IDs
runs: publish success/collision/fail ------------- test_runs: inject disk error before publication
runs: resolve/custom root/escape ---------------- test_runs: traversal/symlink/outside policy
runs: size/NaN/config bounds --------------------- test_runs: reject BEFORE parser/allocation/load
inference: safe supported load ------------------ test_transformer: save/reload logits parity
inference: labels/checksum/missing artifact ------- test_transformer: wrong mapping/changed weights
app: browse without CSV / invalid upload -------- AppTest: saved report/Predict still reachable
app: select run/model / edit+submit -------------- AppTest: stale cleared; submitted excerpt/model
app: empty/loading/error/partial ----------------- AppTest: exact state and recovery for table rows
app: narrow / keyboard / zoom ------------------- live 375/768/1280px + focus/200% zoom check
synthesis: parent eligibility/bounded generation - test_synthesis: no heldout/truncation path
synthesis: candidate validation ----------------- test_synthesis: empty/duplicate/refusal/oversize
review: accept/reject/ambiguous/conflict ---------- test_synthesis: approved hash/revision only
review -> training eligibility ------------------ integration: rejected/changed descendant blocked
study: equal updates / selection cadence -------- tiny model integration: same evaluated steps
study: final access policy ---------------------- integration: no final text in dev entrypoints
study: paired intervals/seed spread -------------- unit: known delta, zero diff, nonpaired rejection
CLI -> output -> app -> Predict ----------------- end-to-end offline tiny-model roundtrip
real generator quality / classifier performance - labelled pilot EVAL; separate from unit tests
```
All 22 rows are required new coverage groups; they are gaps relative to today's suite. Highest-priority Friday-release test is the offline end-to-end run publication, discovery, exact selected-model prediction and legacy fallback. Hostile tests permute evidence IDs without changing array length, swap class labels, modify artifacts behind a cache key and inject symlink escapes. Chaos tests interrupt publication or fail the second model after the first saved checkpoint; neither may publish a complete aggregate report.

Generator eval cases include negation, mixed sentiment, delivery complaints, factual numbers, sarcasm, long reviews with decisive final clauses, and instruction-like source text. Compare no-augmentation, duplicate exposure, prefix and approved paraphrases using common accepted parents; review on training data only, never a final-cohort example. No CLAUDE prompt/tool-definition patterns exist here, so no agent eval suite is inferred; this is a text-generation quality eval with a written human rubric.

Use many deterministic unit cases, a few real tiny-model integrations and one full app-flow test; network and real large-model quality tests are explicit manual research checks. Do not assert exact large-model numeric metrics across hardware or convert a fixed random seed into a claim of universal determinism. Write each new owning-layer test first, observe the intended failure, implement the smallest coherent change, then rerun the coupled producer/consumer checks.

### Section 4: performance and resource bounds
Before parsing a manifest, cap report size at 10 MiB, predictions at the app's existing 50,000-row maximum, nesting/schema structure and finite numbers. Scan only configured bounded roots, at most 1000 run directories to depth two, and paginate visible run rows; explain when results are truncated. Load config/tokenizer files with 1 MiB/10 MiB limits and validate supported dimensions before model construction; initial supported checkpoints are the existing DistilBERT classifier family and tiny test model, not every arbitrary Hub model.

Bound classifier weight files to 512 MiB each/768 MiB total for this local milestone, reject unreasonable config dimensions and set explicit batch/token limits before loading. These caps are application constraints selected for the existing small model, not universal “safe file” guarantees; extending supported families requires a deliberate config/resource review. Do not parse a malicious size header or allocate an enormous DataFrame before checking bounds.

A single loaded classifier plus optimiser state already has material memory cost in train.fit_model, including the deep-copied best_state. Measure peak RSS for the pilot; discard models between conditions and do not share mutable training objects with prediction. Model-cache eviction is bounded, inference is serialized for a shared cached resource, and raw review inputs stay session-local. There are no database traversals, indexes, connection pools or N+1 queries to optimise; filesystem enumeration, model loading and repeated training are the relevant slow paths.

### Engineering failure registry and rollout
| Failure | Owner | Critical if silent? | Planned blocking test / recovery |
|---|---|---|---|
| Same normalized ID, different exact inputs | core/runs | Yes | Exact-input case-change fixture; v2 ID changes |
| Same membership, wrong prediction order | runs/app | Yes | Permuted/missing IDs rejected or correctly joined |
| More checkpoint-selection opportunities | train/study | Yes | Equal evaluated steps and tie-rule fixture |
| New output not discoverable | train/runs/app | No, if explicit | Root policy + contextual output message |
| Oversized/corrupt metadata/config | runs/inference | Yes | Bound-before-load tests; disabled capability |
| Rejected synthetic text trains anyway | synthesis/core | Yes | Hash/review lineage eligibility integration |
| Final cohort used during development | protocol/study | Yes | Entry-point access tests; separate evaluation command |
| Incomplete run presented as completed | runs/app | Yes | Failure injection; no published report |
All eight are requirements, not existing fixes. Reader-before-writer rollout, immutable outputs and retained v1 reader cover compatibility. Rollback never converts v2 metadata to guessed v1 or rewrites original artifacts.

### Engineering completion summary
Scope accepted as a staged roadmap, no planned feature removed. Architecture: 5 material contract findings resolved in the specification; code quality: 3 ownership/error/seed findings; test diagram: 22 new coverage groups with owners, plus existing 11 green tests. Performance: 4 explicit resource-bound decisions; 0 silent critical gaps accepted, though all new capabilities await implementation and test-first verification. What already exists and exclusions are recorded in CEO and repeated by reference here: no new service, CI, deployment, dependency, repository mutation or framework migration. Parallelisation: one implementation lane for shared contracts, then one for inference/UI, then one for generation/study; stages are sequential because each depends on earlier contracts, not isolated worktrees. Test-plan artifact is written separately under ~/.gstack/projects/signallab/.

### Decision 15: Separate exact input identity from normalized dedup grouping

| # | Phase | Decision | Classification | Principle | Rationale | Rejected |
|---|---|---|---|---|---|---|
| 15 | Eng | Separate exact input identity from normalized dedup grouping | Mechanical | Completeness | Normalized data hashes cannot identify every model input | Use current casefolded hash as complete evidence |

### Decision 16: Persist ordered sample/truth/prediction records

| # | Phase | Decision | Classification | Principle | Rationale | Rejected |
|---|---|---|---|---|---|---|
| 16 | Eng | Persist ordered sample/truth/prediction records | Mechanical | Explicit over clever | Prevents wrong positional joins and supports reports without source text | Bare prediction vectors |

### Decision 17: Match validation opportunities as well as optimiser updates

| # | Phase | Decision | Classification | Principle | Rationale | Rejected |
|---|---|---|---|---|---|---|
| 17 | Eng | Match validation opportunities as well as optimiser updates | Mechanical | Completeness | Checkpoint selection cadence otherwise confounds augmentation | Equal epochs or steps alone |

### Decision 18: Keep outside-root CLI outputs explicit, never silently scanned

| # | Phase | Decision | Classification | Principle | Rationale | Rejected |
|---|---|---|---|---|---|---|
| 18 | Eng | Keep outside-root CLI outputs explicit, never silently scanned | Mechanical | Pragmatic | Preserves output compatibility while containing UI file access | Scan arbitrary filesystem paths |

### Decision 19: Validate byte, numeric and model-config bounds before allocation

| # | Phase | Decision | Classification | Principle | Rationale | Rejected |
|---|---|---|---|---|---|---|
| 19 | Eng | Validate byte, numeric and model-config bounds before allocation | Mechanical | Completeness | One malformed artifact can exhaust the local process | Only bound run count |

### Decision 20: Pilot FLAN-T5-small locally with fixed feasibility gates

| # | Phase | Decision | Classification | Principle | Rationale | Rejected |
|---|---|---|---|---|---|---|
| 20 | Eng | Pilot FLAN-T5-small locally with fixed feasibility gates | Taste | Pragmatic | A concrete small checkpoint and bounded rubric make next research action executable | Unbounded generator exploration |

### Decision 21: Use a fixed final cohort and a predeclared paired bootstrap

| # | Phase | Decision | Classification | Principle | Rationale | Rejected |
|---|---|---|---|---|---|---|
| 21 | Eng | Use a fixed final cohort and a predeclared paired bootstrap | Mechanical | Explicit over clever | Avoids treating dependent seed/split scores as independent evidence | Search pairs on the final set |

### Engineering independent review
Independent Codex subagent reported five material gaps: exact input identity (high), ordered prediction joins (high), output-root policy (medium), validation/checkpoint opportunity (medium), and preallocation bounds (medium). All five are now explicit in the engineering contracts and corresponding required tests. This does not mean the implementation is fixed; only the plan has been corrected.

| Dimension | Claude | Codex CLI | Independent Codex subagent | Final primary disposition |
|---|---|---|---|---|
| Architecture | N/A | N/A | Close identity/join contracts | Specified exact IDs and ordered records |
| Tests | N/A | N/A | Add five adversarial groups | Added to 22-row coverage map |
| Performance | N/A | N/A | Bound single artifacts before load | Byte/config/row/model limits specified |
| Security | N/A | N/A | Resolve output/root discrepancy | Explicit configured roots; no arbitrary scan |
| Errors | N/A | N/A | Validate corrupt and partial evidence | Named recovery + publication failure tests |
| Rollout | N/A | N/A | Reader-first preservation sound | Preserve legacy artifacts; no guessed conversion |
Cross-model confirmed 0/6; independent same-model review only. No new feasibility blocker remains unspecified, but the generator's quality/resource gate is deliberately empirical and can stop the later study.

## Phase 3.5: developer-experience review

### Persona, mode and evidence
Mode: DX POLISH. Primary developer persona is a learner or technical portfolio reviewer comfortable opening a terminal, but not expected to understand Hugging Face caches or run schemas before seeing value. They expect a working local demo, a small reproducible experiment and enough source clarity to explain the method. This is a Python learning app, not a public SDK, enterprise integration or agent skill.

Read README, pyproject.toml, .python-version, both argparse CLIs and the planned producer/consumer boundaries. Executed both existing --help commands; they expose flags but omit individual descriptions/defaults. No cold installation, fresh-machine walkthrough or Windows/Linux validation was performed. All time targets below are acceptance targets, not observed timings.

### Developer perspective
I open the README and see a customer-review experiment rather than another generic sentiment demo. Under Run, I find an absolute directory on Thomas's computer. I can replace it with my own project folder, but I have to infer that; a portable instruction would remove that first doubt. The next two commands are clear: synchronise the lockfile and run Streamlit. I want to see the authored demo first, before downloading customer reviews or waiting for training.

The demo warning tells me not to treat its scores as a benchmark, which helps me trust the project. I run the baseline comparison and want to understand a mistake, then try the actual transformer. Today Predict only offers the baseline; the proposed saved-model selector makes that distinction useful rather than hidden. When I move to the terminal, train --help lists --ratio and --epochs, but does not tell me their ranges, CPU cost or what happens when output already exists. I need one small command sequence with the expected report path and a visible confirmation in the app.

If I disconnect the network, I expect the authored demo to work after dependencies are installed. I do not assume that also proves transformer assets are complete. If a tokenizer is missing, tell me exactly which preparation step is needed and preserve the typed review. I want to rerun the experiment and understand its limitations, not learn a tracking platform before I can inspect one result.

This narrative is a source-grounded persona hypothesis, not a transcript of observed user research.

### Comparable onboarding benchmarks
| Tool | Official documented path | Time evidence | Implication |
|---|---|---|---|
| Streamlit | Install package, run hello | No measured cold-start time verified | Demo-first local success is conventional |
| Hugging Face Transformers | Load explicit pretrained assets, then inference | Downloads/hardware affect time; not benchmarked here | Separate asset acquisition from cached inference |
| MLflow | Install package, log a run, inspect it | Documentation says quickstart; no timing measurement here | Run metadata/results inspection is familiar; dependency not required here |
| SignalLab today | cd project; uv sync; Streamlit app; run baseline | Existing environment works; first-install time unknown | Remove machine-specific path and teach one golden path |

References inspected: [Streamlit installation](https://docs.streamlit.io/get-started/installation), [Transformers quickstart](https://huggingface.co/docs/transformers/quicktour), [MLflow tracking quickstart](https://www.mlflow.org/docs/latest/ml/getting-started/quickstart/). These are workflow comparisons, not a competitor latency ranking.

### First success and concrete reproduction recipe
The first success is seeing the authored reviews and one measured baseline result without an account, data download or GPU. On a machine with uv, supported Python and dependencies already installed, target at most 3 minutes from opening the README to that result; measure cold dependency installation separately and publish actual hardware/network context. Do not promise under-five-minute cold ML setup without measurements.

The portable launch sequence remains three steps, with the project-directory step explicitly adapted to the user's checkout:
```text
cd <your-signallab-directory>
uv sync --locked
uv run streamlit run app.py --server.address 127.0.0.1 --server.port 8537
```
Use a free isolated port if occupied; never kill another process to free it. The expected first state is Explore with “Authored demo” selected; then Experiments -> Run baseline comparison -> measured table. Preparing the environment is distinct from acquiring real data or model weights.

After the new protocol/CLI features are implemented, document and test this exact small research recipe; these flags/subcommands are PROPOSED and must not be presented as working today:
```text
uv run python -m signallab.fetch_data --limit 200 --output data/research-pilot-200.csv
uv run python -m signallab.study prepare --csv data/research-pilot-200.csv --split-seed 42 --output artifacts/protocols/research-pilot.json
uv run python -m signallab.train --csv data/research-pilot-200.csv --protocol artifacts/protocols/research-pilot.json --seed 42 --checkpoint distilbert/distilbert-base-uncased --max-steps 15 --eval-every 5 --batch-size 8 --ratio 0.5 --output artifacts/runs/research-pilot-42
```
`study prepare` freezes development roles and records the reserved-final-cohort selection policy without reading final labels into development. Source and model revisions resolve once into the protocol/run and reproduction instructions use those resolved revisions thereafter; a second execution checks expected digests rather than silently resolving a newer model/data revision. Target here is a bounded smoke experiment: both training conditions get 15 updates and validations at steps 5, 10, 15. The expected result is a completed report in the named new directory, discoverable in Experiments, and both saved classifiers selectable in Predict. The scores are not a quality threshold.

The proposed --checkpoint-revision and --dataset-revision/expected-digest options must be documented alongside the golden path and emitted reproduction command. A real replication uses those values from the published manifest. If an output exists, the developer chooses a fresh path; the tool does not overwrite it. The later [13,42,87] seed study and <=50-parent synthesis pilot are separate recipes, not hidden work behind the first command.

### Nine-stage developer journey
| Stage | Actual touchpoint | Friction | Planned resolution |
|---|---|---|---|
| 1 Discover | README purpose | Outcome buried after features | Question, exploratory finding and screenshot first |
| 2 Install | Run section/uv.lock | Absolute local path; large cold ML setup | Portable path, supported Python, separate cold timing |
| 3 Hello world | Streamlit/Explore | Learner may start downloads too early | Authored offline-after-install demo first |
| 4 First experiment | Experiments baseline button | Session result vs durable run unclear | Save/show run ID and exact output |
| 5 Real usage | fetch/train CLIs | Flags lack help; no budget recipe | Descriptions, valid ranges, bounded recipe |
| 6 Debug | Parser failures/model load | Generic exception type without cause | Safe code, cause, next command and docs anchor |
| 7 Reproduce | Saved report and local assets | Revision/schema/output assumptions | Manifest-generated pinned command and capability checks |
| 8 Upgrade | Existing v1 report | New reader may reject/guess metadata | Legacy read-only display; non-destructive new v2 runs |
| 9 Extend/share learning | Code and case study | No public license/distribution settled | Local extension guide; publication/licensing remains a later user decision |

### First-time confusion report
These are simulated timestamps for source-based roleplay, not measured elapsed time. T+0:00: learner opens README and replaces a personal path (fix portable instructions). T+0:30: learner wonders if Hugging Face login is mandatory (state no login needed for the existing public demo/model workflow). T+1:00: learner sees generic flags without defaults (add argparse descriptions). T+2:00: learner trains into an outside-root output and cannot find it (print discovery policy/recovery). T+3:00: learner wants to try the saved model offline but lacks tokenizer files (list prerequisites and exact missing asset). All five are in scope as onboarding/help acceptance refinements.

### Pass 1: getting started, 6 → 8
The existing Run section provides a valid local command but hardcodes a machine path and mixes installation with a ready environment. Keep the demo as the first meaningful interaction, requiring no credentials, and teach real training only after it. A 10 would require measured novice cold/warm walkthroughs; the plan instead records a <=3-minute warm target and honestly leaves cold timing unknown.

### Pass 2: CLI design, 5 → 8
Use the existing module-command style rather than introducing an umbrella CLI framework. Add help text for every flag: units, defaults, range, network requirements, CPU cost, preserve-output behaviour and discovery root. Validate incompatible flags before downloading models or creating run directories, and expose a --validate-only option for the study/train protocol checks if it reduces expensive failed runs; its output must report that training did not occur.

Keep seed for training randomness separate from protocol split seed. --max-steps and --eval-every are explicit update units, while legacy --epochs remains supported with a labelled equal-epoch budget; reject mutually ambiguous combinations instead of guessing. A 10 would include a clean external learner walkthrough of the implemented help and recipe.

### Pass 3: errors and debugging, 4 → 9
Three traced current paths require different messages. Output collision already suggests a new path, but should name the run/output and link docs. Invalid epochs currently collapses into “Training failed (ValueError)” even though the cause is known; name the invalid flag/range before model acquisition. Missing tokenizer/model assets should say which required asset is absent, that no network fetch occurred during offline inference, and how to run the explicit preparation step.

Target format: stable safe error code + problem + cause + recovery + local docs anchor. Example: `OUTPUT_EXISTS: artifacts/runs/research-pilot-42 already exists. Existing results were preserved. Choose a new --output directory. See README#saved-runs.` Debug details preserve exception chaining locally without printing raw reviews, secrets or full environment state. Error contracts are tested for the message and non-zero status, not only that something raised.

### Pass 4: documentation and learning, 6 → 8
Organise README as quick demo, one small experiment, saved-model prediction, troubleshooting and research method; put full schema/flags in reference sections and the case study in docs/case-study.md. The generator pilot and final study examples must say whether prerequisites/assets are already present, give a bounded budget and expected output path, and mark all unpublished planned commands as such. A 10 requires executing every published example from an existing environment and asking a learner to explain the result without reading implementation code.

### Pass 5: upgrade and migration, 6 → 9
Write v2 reports into new exclusive directories and keep a tested v1 read path. Show “Legacy: split/input provenance unverified” and distinguish whether metrics are readable, safe model inference is available and controlled comparisons are permitted. No codemod or bulk mutation is needed because raw artifacts are not rewritten; verify old and new outputs side by side after each stage.

### Pass 6: environment and tooling, 7 → 8
Keep uv.lock, Python 3.11–3.13 and standard unittest/AppTest. Document exact tested environment and separate unsupported platforms from proven ones; macOS ARM64 evidence does not prove Windows/Linux operation. Tests and bounded CLIs remain noninteractive and can be run manually; no CI/CD is added. Runtime use is offline only after dependencies and the necessary tokenizer/config/safetensors assets are installed.

### Pass 7: community and ecosystem, 5 → 6
This is currently a local portfolio, with no confirmed public repository, license or support channel. Do not invent an open-source license, community, adoption figures or hosting commitment to raise a score. Provide local extension pointers and source citations now; public distribution and contribution policy remain a P3 user decision if the portfolio is later shared.

### Pass 8: measurement and feedback, 3 → 8
Measure warm demo success, cold sync time, asset download time and actual training time separately; one number would conceal very different sources of waiting. Record where a learner needed help in a local walkthrough, without hosted telemetry or analytics. Recheck the walkthrough after implementation and compare with the <=3-minute warm target; do not automatically schedule monitors, automations or surveys.

### DX scorecard and implementation checklist
| Dimension | Current evidence score | Reviewed plan score | Remaining evidence needed |
|---|---|---|---|
| Getting started | 6 | 8 | Warm/cold learner timing |
| CLI/API | 5 | 8 | Implemented help/invalid-flag tests |
| Errors | 4 | 9 | Trigger specific failures and recovery |
| Documentation | 6 | 8 | Execute published bounded recipe |
| Upgrade | 6 | 9 | v1/v2 roundtrip fixtures |
| Environment | 7 | 8 | Offline assets and platform limits verified |
| Community | 5 | 6 | Public distribution remains undecided |
| Measurement | 3 | 8 | Local walkthrough data |
Overall plan score 8.0/10 versus current evidence assessment 5.25/10. These are reviewer judgments, not measured DX outcomes. TTHW target <=3 minutes warm; current/cold unknown. Competitive workflow target is a local demo followed by one small reproducible run, not a measured latency rank. The magical moment is saved-model output linked to the exact experiment, delivered through the existing app and copyable CLI recipe.

- [ ] Portable three-step demo instructions; expected first screen and baseline result.
- [ ] Help describes every flag/range/unit/default, CPU/network boundary and preservation policy.
- [ ] One bounded real-data recipe, pinned reproduction command and expected local run output.
- [ ] Explicit offline prerequisites and missing-asset recovery; no inference-time downloads.
- [ ] Stable errors with cause/fix/docs link and non-zero status; no sensitive text logs.
- [ ] v1 readable beside v2; no automatic artifact rewrite.
- [ ] Existing-environment README -> help -> small run -> discovery -> prediction walkthrough.
- [ ] Warm/cold/download/training timings recorded separately; no invented claims.
- [ ] Case study and extension pointers accurate; license/publication intentionally undecided.

Principles covered: first success before credentials/downloads, learning by doing, explicit recovery, defaults with documented overrides, commands in context and saved-model traceability. No TypeScript SDK, editor extension, hosted sandbox, CI workflow, billing, support service or public distribution is added. The product is not a Claude Code skill, so the skill-specific appendix was checked and is inapplicable by product type.

### Decision 22: Use portable demo-first instructions and measure cold setup separately

| # | Phase | Decision | Classification | Principle | Rationale | Rejected |
|---|---|---|---|---|---|---|
| 22 | DX | Use portable demo-first instructions and measure cold setup separately | Mechanical | Explicit over clever | Personal paths and mixed timings hide first-run costs | Promise fast cold ML install |

### Decision 23: Preserve module CLIs and add full help plus one bounded recipe

| # | Phase | Decision | Classification | Principle | Rationale | Rejected |
|---|---|---|---|---|---|---|
| 23 | DX | Preserve module CLIs and add full help plus one bounded recipe | Mechanical | DRY | Existing argparse style works; missing context is the problem | Introduce umbrella CLI framework |

### Decision 24: Make offline asset prerequisites explicit

| # | Phase | Decision | Classification | Principle | Rationale | Rejected |
|---|---|---|---|---|---|---|
| 24 | DX | Make offline asset prerequisites explicit | Mechanical | Completeness | Offline demo does not establish cached transformer readiness | Fetch weights silently during prediction |

### Decision 25: Keep community/publication claims out until actually decided

| # | Phase | Decision | Classification | Principle | Rationale | Rejected |
|---|---|---|---|---|---|---|
| 25 | DX | Keep community/publication claims out until actually decided | Mechanical | Explicit over clever | Local project has no confirmed public repo or license | Invent open-source and adoption story |

### DX independent review
The independent Codex reviewer found five remaining refinements: portable onboarding, full flag help, one bounded reproduction recipe, offline setup boundaries, and a docs walkthrough. All are incorporated above; they refine the confirmed research-first scope, with no new hosting/dependency request.

| Dimension | Claude | Codex CLI | Independent Codex subagent | Primary resolution |
|---|---|---|---|---|
| Getting started <5min | N/A | N/A | Conditional, cold unmeasured | <=3min warm target; separate cold timing |
| CLI naming/defaults | N/A | N/A | Help incomplete | Exact help/flag contract |
| Errors actionable | N/A | N/A | Verify distinct recovery | Specific codes and failure tests |
| Docs complete/findable | N/A | N/A | Bounded recipe missing | Written proposed recipe + walkthrough |
| Upgrade safe | N/A | N/A | Reader-first sound | Preserve/verify legacy capabilities |
| Environment clear | N/A | N/A | Clarify acquisition/offline | Explicit prerequisite and asset checklist |
Cross-model confirmed 0/6. Same-model review is complete; no fresh-machine timing, cross-platform or real new-feature validation is claimed.

## Cross-phase themes

1. **Evidence identity is the product's trust boundary.** CEO called for reproducible records, Design required run-owned context, and Eng found normalized IDs and positional predictions insufficient. The final plan uses exact input fingerprints, ordered evidence and separate readable/comparable/inference capabilities.
2. **Honest research requires budget and access controls.** CEO moved cohort policy before synthesis; Eng matched validation opportunities as well as steps; DX requires those choices in reproduction commands. Final-cohort exposure and development scores are explicitly different states, not labels applied after tuning.
3. **Learning value should arrive before infrastructure is complete.** Independent strategy/design/DX reviews all supported a short early case study, saved-model interaction and a bounded first experiment. This is repeated same-model signal, not independent cross-model confirmation.
4. **Recovery must preserve evidence.** All phases converge on legacy-readable artifacts, no overwrite, contextual errors and clear offline prerequisites. A failed run remains inspectable and never acquires a complete report by fallback.

No user challenges were asserted: the user-confirmed research-first scope remains intact. All refinements are planned, not implemented.

## Final choices approved by user on 2026-10-23

**Choice A: early case study.** Recommend publishing a short local exploratory narrative with the first run/inference release, then extending it after the controlled study. Waiting until the final study is also viable, but delays visible evidence of learning. This is a sequencing/taste choice, not a scientific requirement.

**Choice B: bounded local generator pilot.** Recommend FLAN-T5-small as a concrete candidate, <=50 parents, all candidates human-reviewed, proposed 30-minute/4-GiB cap and stated acceptance thresholds. A manually authored/imported synthetic pilot is also viable and could be faster, but demonstrates less local text-generation integration. These are proposed operational caps; approve them as a starting experiment, not a performance promise or an installation authorisation.

The research-first direction, existing stack/domain, TensorFlow-later ordering, no secrets/external writes and evidence-preserving constraints are already settled. The recorded plan approval does not authorise a new production dependency, paid service, publication, Git mutation, deployment or implementation.

## Implementation order and verification gates

| Stage | Deliverable | Primary signal before moving on |
|---|---|---|
| 0: protocol | Hypothesis, development roles, final-cohort selection/access policy | Inspect protocol before generation; exposed data remains development |
| 1: run evidence | Shared v2 reader/writer, exact identity, ordered predictions, discovery, legacy capabilities | Test-first publication/corruption/compatibility/path fixtures; coupled UI and CLI agree |
| 2: usable learning demo | Saved-transformer Predict + early case study + contextual recovery | Offline save/load prediction parity and browse/compare/predict app flow |
| 3: synthetic pilot | Bounded generator, parent lineage, hash-bound review decisions | Actual reviewed quality/resource evidence; rejected/unreviewed cannot train |
| 4: controlled study | Shared parent controls, matched updates/cadence, seeds, predeclared final pair/interval | Budget/access/paired-statistics fixtures plus measured real experiment |
| 5: finished case study | Reproduction recipe, results, mistakes, limitations, TensorFlow follow-up | Learner can explain/reproduce bounded result; no unsupported gain claims |

First complete release: stages 0–2. Later stages stay in scope but depend on measured pilot results. If a pilot fails, preserve and explain that outcome; do not hide it or substitute a different success criterion.

## Implementation Tasks (aggregated across phases)

Scoped to this review invocation because Git branch/commit metadata is unavailable. Exact-match deduplication only; related documentation/state tasks retain their distinct owning concerns. Estimates are planning ranges, not measured model-running times.

- [ ] **C1 (P1, human: 4h / AI-assisted: 1h) — study protocol** — Freeze evaluation roles and predeclare hypothesis before synthesis
  - Source: ceo-review — CEO sections 0A, 0D, 4: currently exposed holdout cannot become fresh final evidence
  - Files: signallab/core.py, signallab/fetch_data.py, tests/test_core.py
- [ ] **D1 (P1, human: 3h / AI-assisted: 45m) — workspace context** — Separate saved-run context from active dataset preparation
  - Source: design-review — Design passes 1 and 7: current top-level dataset validation blocks all pages
  - Files: app.py, tests/test_app.py
- [ ] **E1 (P1, human: 1–2d / AI-assisted: 2–4h) — run contracts** — Implement versioned exact-input identity, ordered evidence and immutable publication
  - Source: eng-review — Eng architecture: normalized IDs and positional predictions cannot prove compatibility
  - Files: signallab/core.py, signallab/runs.py, signallab/train.py, app.py, tests/test_runs.py
- [ ] **E2 (P1, human: 1d / AI-assisted: 2h) — saved inference** — Load supported local safetensors safely and verify save/load prediction parity
  - Source: eng-review — Eng inference contract: saved transformer is not currently usable in Predict
  - Files: signallab/inference.py, signallab/train.py, app.py, tests/test_transformer.py
- [ ] **E3 (P1, human: 1d / AI-assisted: 2h) — study fairness** — Match optimiser budget and validation schedule and protect final evaluation roles
  - Source: eng-review — Eng controlled study: equal epochs and validation-per-epoch confound the comparison
  - Files: signallab/train.py, signallab/core.py, signallab/study.py, tests/test_study.py
- [ ] **C2 (P2, human: 2h / AI-assisted: 30m) — case study** — Write the exploratory negative-result case study early
  - Source: ceo-review — CEO section 0D and independent strategy review: portfolio value should not wait for infrastructure
  - Files: docs/case-study.md, README.md
- [ ] **D2 (P2, human: 4h / AI-assisted: 1h) — interaction states** — Implement the run, inference and review state tables with native controls
  - Source: design-review — Design pass 2: partial/corrupt/incompatible/model-missing states need distinct actions
  - Files: app.py, tests/test_app.py
- [ ] **E4 (P2, human: 1–2d / AI-assisted: 2–4h plus measured pilot/review time) — synthetic lineage** — Run the bounded local generator and require hash-bound human review
  - Source: eng-review — Eng generation protocol: label preservation and feasible CPU cost are unmeasured
  - Files: signallab/synthesis.py, app.py, tests/test_synthesis.py
- [ ] **X1 (P2, human: 4h / AI-assisted: 1h) — learning path** — Document portable demo, pinned bounded run and offline asset recovery
  - Source: devex-review — DX 1, 3, 4: hardcoded path, missing bounded reproduction recipe and offline prerequisites
  - Files: README.md, docs/case-study.md, signallab/train.py, signallab/fetch_data.py
- [ ] **X2 (P2, human: 3h / AI-assisted: 45m) — CLI contracts** — Add argument help, preflight validation and specific actionable error tests
  - Source: devex-review — DX 2 and 3: --help lacks descriptions and errors collapse distinct causes
  - Files: signallab/train.py, signallab/fetch_data.py, tests/test_cli.py

## Consolidated Decision Audit Trail

| # | Phase | Decision | Classification | Principle | Rationale | Rejected |
|---|---|---|---|---|---|---|
| 1 | Intake | Plan in the existing Python/Streamlit project | Mechanical | DRY | Reuse the implementation the user approved | Rebuild in a different stack |
| 2 | Intake | Include UI and developer-experience reviews | Mechanical | Completeness | Run selection, prediction states, CLI and reproducibility instructions all change | Backend-only review |
| 3 | Intake | Preserve implementation and existing artifacts | Mechanical | Explicit over clever | User requested next-step design, not implementation | Silent code changes or artifact replacement |
| 4 | Intake | Research-first ordering confirmed by user | User-confirmed | Pragmatic | User explicitly selected research-first on 2026-10-23 | Framework breadth first |
| 5 | CEO | Freeze final-cohort protocol before generation | Mechanical | Completeness | Prevents the later study from consuming its final evaluation while developing synthetic methods | Reserve it only at milestone 4 |
| 6 | CEO | Publish a short exploratory case study in the first release | Taste | Bias toward action | Makes learning visible while preserving the complete later study | Wait for all infrastructure |
| 7 | CEO | Limit initial generation to 50 parents and one candidate each | Taste | Pragmatic | CPU feasibility and review quality need evidence before scaling | Run the complete matrix immediately |
| 8 | CEO | Use local immutable reports, not a tracking service | Mechanical | DRY | Existing JSON and local privacy boundary cover current needs | Add MLflow dependency/server |
| 9 | CEO | Report same-model review degradation explicitly | Mechanical | Explicit over clever | Claude is logged out and Codex CLI model is rejected | Fabricate cross-model consensus |
| 10 | Design | Keep four native workspaces and put run browsing inside Experiments | Mechanical | DRY | Existing navigation fits the learning journey | New dashboard shell |
| 11 | Design | Decouple saved-model use from active CSV validation | Mechanical | Completeness | Missing original data must not block valid local inference | Gate all workspaces on dataset upload |
| 12 | Design | Invalidate prediction on model/run/artifact changes | Mechanical | Explicit over clever | Prevent stale text or stale-model results | Keep last result without identity |
| 13 | Design | Specify 375/768/1280px and keyboard states | Mechanical | Completeness | Native widgets still require verification | Assume responsive accessibility |
| 14 | Design | Separate readable, comparable and inference-ready capabilities | Mechanical | Explicit over clever | A run may have valid metrics but missing weights or unverifiable splits | Single overloaded completed flag |
| 15 | Eng | Separate exact input identity from normalized dedup grouping | Mechanical | Completeness | Normalized data hashes cannot identify every model input | Use current casefolded hash as complete evidence |
| 16 | Eng | Persist ordered sample/truth/prediction records | Mechanical | Explicit over clever | Prevents wrong positional joins and supports reports without source text | Bare prediction vectors |
| 17 | Eng | Match validation opportunities as well as optimiser updates | Mechanical | Completeness | Checkpoint selection cadence otherwise confounds augmentation | Equal epochs or steps alone |
| 18 | Eng | Keep outside-root CLI outputs explicit, never silently scanned | Mechanical | Pragmatic | Preserves output compatibility while containing UI file access | Scan arbitrary filesystem paths |
| 19 | Eng | Validate byte, numeric and model-config bounds before allocation | Mechanical | Completeness | One malformed artifact can exhaust the local process | Only bound run count |
| 20 | Eng | Pilot FLAN-T5-small locally with fixed feasibility gates | Taste | Pragmatic | A concrete small checkpoint and bounded rubric make next research action executable | Unbounded generator exploration |
| 21 | Eng | Use a fixed final cohort and a predeclared paired bootstrap | Mechanical | Explicit over clever | Avoids treating dependent seed/split scores as independent evidence | Search pairs on the final set |
| 22 | DX | Use portable demo-first instructions and measure cold setup separately | Mechanical | Explicit over clever | Personal paths and mixed timings hide first-run costs | Promise fast cold ML install |
| 23 | DX | Preserve module CLIs and add full help plus one bounded recipe | Mechanical | DRY | Existing argparse style works; missing context is the problem | Introduce umbrella CLI framework |
| 24 | DX | Make offline asset prerequisites explicit | Mechanical | Completeness | Offline demo does not establish cached transformer readiness | Fetch weights silently during prediction |
| 25 | DX | Keep community/publication claims out until actually decided | Mechanical | Explicit over clever | Local project has no confirmed public repo or license | Invent open-source and adoption story |

## Pre-gate verification

- CEO: specific premises, leverage/alternatives/dream/temporal analysis, all 11 sections, 13-path rescue table, seven-category failure table, exclusions, summary and independent review recorded.
- Design: all seven passes scored, run capability/state tables, wireframe/journey, system alignment and explicit responsive/keyboard acceptance criteria recorded. No generated/approved mockup or visual QA claimed.
- Engineering: actual code/tests read, dependencies and contracts diagrammed, 22-row new-coverage map, resource/error/rollout requirements, independent review and separate QA test-plan artifact written.
- DX: persona, first-person narrative, official workflow comparisons, nine-stage journey, simulated confusion log, eight scored passes, target/measurement limits, checklist and independent review recorded.
- Four phase task artifacts written using jq JSON serialization, current-invocation aggregation complete. Cross-phase themes and decision audit recorded. Deferred work is in TODOS.md.
- No Git repository: platform/base history/diff/stash/commit-window unavailable; main fallback is bookkeeping only. No repository mutation performed.
- Review limitations: Claude unauthenticated; Codex CLI invocation failed unsupported configured model. Four independent Codex subagent reviews completed; cross-model agreement is N/A in every phase. No repeated retry or settings change.
- Existing validation rerun: 11 unittest/AppTest/tiny-model tests passed. New-feature tests, actual generator quality, runtime responsive/accessibility, fresh-machine timing and new-study results remain future implementation evidence.

## GSTACK REVIEW REPORT

| Review | Trigger | Why | Runs | Status | Findings |
|---|---|---|---|---|---|
| CEO | autoplan | Research scope and sequencing | 1 | APPROVED; implementation unverified | Protocol before synthesis; early case study; bounded pilot |
| Design | autoplan | Run ownership and interaction clarity | 1 | REVIEWED; implementation unverified | 7 passes; plan score 8.6/10; capability/state contracts |
| Engineering | autoplan | Exact evidence and safe experiment execution | 1 | REVIEWED; implementation unverified | 5 material independent findings specified; 22 future coverage groups |
| DX | autoplan | Reproduction, CLI and learning flow | 1 | REVIEWED; implementation unverified | 8 passes; plan score 8.0/10; warm <=3min target, cold unknown |
| Independent voices | autoplan | Challenge primary assumptions | 4 subagent reviews; 1 CLI attempt | SINGLE-MODEL DEGRADATION | Claude logged out; CLI model rejected; no cross-model consensus |

**VERDICT:** Planning and review artifacts complete with declared limitations. User approved the plan as-is on 2026-10-23, including both taste recommendations. No research implementation, release or benefit claim is cleared by review scores alone. The separately requested Cloudflare-inspired redesign supersedes earlier visual assumptions only.

NO UNRESOLVED DECISIONS
