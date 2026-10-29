"""Pure, text-free evidence stories. Review previews have a separate contract."""
from dataclasses import dataclass
import json
import re

from signallab.analysis import analysis_rows
from signallab.runs import attach_reviews, digest, validate_record

MAX_MARKDOWN_BYTES = 1024 * 1024
QUESTION = 'Does training-only prefix augmentation change sentiment classification on the same held-out reviews?'
LIMITATIONS = (
    'A small exploratory partition and a single seed do not establish generalization, significance or a causal augmentation benefit.',
    'The prefix adds examples, not independent semantic information. This is a weak control, not LLM-generated paraphrasing.',
    'No frozen final evaluation cohort or matched-budget repeated-seed study is recorded here.',
    'Review hashes identify evidence; they are not anonymisation. Source labels are recorded declarations, not verified licenses.',
    'Semantic duplicates, product/customer grouping and pretrained-data contamination remain uncontrolled.',
    'Binary sentiment cannot represent neutral or mixed reviews; model probabilities are uncalibrated.',
    'Recorded source hashes and environment describe this run; they do not archive its code or prove a fresh reproduction.',
)


@dataclass(frozen=True)
class Condition:
    name: str
    training_rows: int
    macro_f1: float
    accuracy: float


@dataclass(frozen=True)
class CaseStudy:
    run_id: str
    content_id: str
    input_id: str
    dataset_id: str
    split_digest: str
    created_at: str
    partition: str
    source: str
    seed: int
    ratio: float
    support: int
    conditions: tuple
    paired_counts: tuple
    warnings: tuple
    recorded_environment: str

    @property
    def f1_delta(self):
        return self.conditions[1].macro_f1 - self.conditions[0].macro_f1

    @property
    def accuracy_delta_pp(self):
        return 100 * (self.conditions[1].accuracy - self.conditions[0].accuracy)

    @property
    def interpretation(self):
        return describe_change(self.f1_delta)


@dataclass(frozen=True)
class Example:
    category: str
    review_id: str
    text: str
    actual: str
    real_prediction: str
    augmented_prediction: str


def describe_change(delta):
    if delta == 0:
        return 'No measured change in macro F1 on this partition.'
    if round(delta, 4) == 0:
        return 'The macro F1 difference is smaller than the displayed precision; this is not an exact tie.'
    return f'Prefix-augmented macro F1 is {"higher" if delta > 0 else "lower"} on this partition. This is a descriptive difference, not evidence of a reliable effect.'


def _category(real, augmented):
    a = real['prediction'] == real['label']
    b = augmented['prediction'] == augmented['label']
    if a and b:
        return 'Both correct'
    if b:
        return 'Corrected'
    return 'Regressed' if a else 'Both wrong'


def build_case_study(record, partition='validation'):
    validate_record(record)
    if partition not in ('validation', 'test'):
        raise ValueError('Choose validation or test for the case study.')
    manifest = record['manifest']
    source = manifest.get('data_source', 'Source not recorded')
    warnings = ['Exploratory baseline comparison; this is not a final benchmark.']
    if source == 'Authored demo':
        warnings.append('Authored demo illustrates the workflow; these are not real customer performance results.')
    elif source == 'Source not recorded':
        warnings.append('Source not recorded. Check original dataset provenance before treating these scores as research evidence.')
    if partition == 'test':
        warnings.append('Repeated inspection makes these test results developmental. Reserve fresh reviews for final claims.')
    first, second = [run[partition] for run in record['runs']]
    counts = dict.fromkeys(('Corrected', 'Regressed', 'Both wrong', 'Both correct'), 0)
    for a, b in zip(first['evidence'], second['evidence']):
        counts[_category(a, b)] += 1
    environment = {key: manifest[key] for key in (
        'preprocessing', 'split_policy', 'parameters', 'dependencies', 'python', 'git_commit', 'source_hashes')}
    environment['split_sizes'] = record['split_sizes']
    environment['augmentation'] = record['augmentation']
    return CaseStudy(
        run_id=record['run_id'], content_id=digest(record), input_id=manifest['input_id'],
        dataset_id=record['dataset_id'], split_digest=manifest['split_digest'],
        created_at=record['created_at'], partition=partition, source=source,
        seed=record['seed'], ratio=record['augmentation']['ratio'], support=first['n'],
        conditions=tuple(Condition(run['name'], run['training_rows'], run[partition]['macro_f1'], run[partition]['accuracy']) for run in record['runs']),
        paired_counts=tuple(counts.items()), warnings=tuple(warnings),
        recorded_environment=json.dumps(environment, ensure_ascii=False, sort_keys=True, indent=2),
    )


def case_study_examples(record, frame, partition='validation'):
    """Exact attachment and deterministic preview; never supplied to Markdown."""
    if partition not in ('validation', 'test'):
        raise ValueError('Choose validation or test for examples.')
    splits = attach_reviews(record, frame)
    evaluated = getattr(splits, partition)
    rows = analysis_rows(record['runs'][0][partition], evaluated).set_index('review_id')
    second = analysis_rows(record['runs'][1][partition], evaluated).set_index('review_id')
    pairs = sorted(zip(record['runs'][0][partition]['evidence'], record['runs'][1][partition]['evidence']), key=lambda pair: pair[0]['review_id'])
    examples = []
    for category in ('Corrected', 'Regressed', 'Both wrong'):
        selected = [a['review_id'] for a,b in pairs if _category(a,b) == category][:2]
        for review_id in selected:
            row, other = rows.loc[review_id], second.loc[review_id]
            examples.append(Example(category, review_id, row.text, row.Actual, row.Predicted, other.Predicted))
    return tuple(examples)


def case_study_markdown(study):
    """Export only the numeric story type; no frame or Example can enter it."""
    if not isinstance(study, CaseStudy):
        raise ValueError('Prepare a validated numeric case study before exporting.')
    lines = ['# SignalLab case study', '', QUESTION, '', '## Evidence',
             f'- Run: {study.run_id}', f'- Saved at: {study.created_at}',
             f'- Record content digest: {study.content_id}', f'- Exact input: {study.input_id}',
             f'- Normalised dataset: {study.dataset_id}', f'- Split digest: {study.split_digest}',
             f'- Partition: {study.partition} ({study.support} reviews)', f'- Recorded source: {study.source}',
             f'- Seed: {study.seed}', f'- Completed prefix ratio: {study.ratio!r} synthetic examples per real training example', '',
             *[f'> {warning}' for warning in study.warnings], '', '## Method',
             'TF-IDF word uni/bigrams and balanced logistic regression. Both conditions use the same held-out reviews; augmentation adds a prefix to sampled training reviews only.', '',
             '## Measured results', '| Condition | Training rows | Macro F1 | Accuracy |',
             '|---|---:|---:|---:|']
    for condition in study.conditions:
        lines.append(f'| {condition.name} | {condition.training_rows} | {condition.macro_f1:.4f} | {condition.accuracy:.2%} |')
    f1 = 0.0 if round(study.f1_delta, 4) == 0 else study.f1_delta
    pp = 0.0 if round(study.accuracy_delta_pp, 2) == 0 else study.accuracy_delta_pp
    lines += ['', f'Macro F1 change: {f1:+.4f}. Accuracy change: {pp:+.2f} percentage points. Direction: augmented minus real-only.', study.interpretation, '', '## Paired mistakes',
              *[f'- {category}: {count}' for category,count in study.paired_counts], '',
              'Illustrative on-screen mistakes use at most two reviews per corrected, regressed and both-wrong category, ordered by exact review ID. They are not a representative sample. Review text is excluded from this export.', '',
              '## Limitations', *[f'- {item}' for item in LIMITATIONS], '', '## Recorded method and environment']
    # Recorded version strings are untrusted. JSON escapes newlines; a longer
    # fence prevents even embedded backticks from breaking into active Markdown.
    longest = max((len(match) for match in re.findall(r'`+', study.recorded_environment)), default=0)
    fence = '`' * max(3, longest+1)
    lines += [fence+'json', study.recorded_environment, fence, '', '## Reproduce and investigate',
              '1. Use the exact original dataset identified above. Original uploaded-file locations were not recorded; supply your own original CSV. Authored demo is usable only when its exact identity matches.',
              f'2. In Experiments select the original dataset and use the exact recorded ratio {study.ratio!r}, then run the baseline comparison and Save experiment. If the slider cannot select that exact value, use the existing Python API `signallab.core.run_comparison(frame, ratio={study.ratio!r})` with the validated original frame. Check recorded environment/source hashes before expecting identical output.',
              f'3. Open Case Study, refresh and select the new comparison and {study.partition}. Inspect the same run and partition in Analysis; reattach original reviews there if needed.',
              'Next research step: freeze an evaluation protocol, add reviewed training-only synthetic controls, and compare matched budgets across seeds before making final claims.', '']
    data = '\n'.join(lines).encode('utf-8')
    if len(data) > MAX_MARKDOWN_BYTES:
        raise ValueError('Case study exceeds 1 MiB. No download was prepared; check recorded metadata.')
    return data
