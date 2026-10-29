"""Immutable, text-free baseline snapshots for the local learning workspace.

A snapshot is evidence, not a serialised model or proof of a final evaluation.
Only immediate children of the configured local root are discoverable.
"""
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import re
import stat
import uuid

from signallab.core import BASELINE_PARAMETERS, PREPROCESSING, SPLIT_POLICY, dataset_id, exact_dataset_id, make_splits, prediction_report, review_id, validate_reviews

ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = ROOT / 'artifacts/runs'
MAX_REPORT_BYTES = 16 * 1024 * 1024
MAX_ENTRIES = 50
PARTITIONS = ('train', 'validation', 'test')
BASELINE_NAMES = ('TF-IDF · real only', 'TF-IDF · augmented')
HEX_ID = re.compile(r'^[0-9a-f]{64}$')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()


def build_record(report, frame, splits):
    """Copy only the supported baseline contract, never arbitrary report fields."""
    clean = validate_reviews(frame)
    _require(report.get("input_id") == exact_dataset_id(clean) and report.get("dataset_id") == dataset_id(clean),
             "This report was trained on different exact input. Rerun the baseline comparison before saving.")
    split_ids = {key: [review_id(row.text, row.label) for row in getattr(splits, key).itertuples()] for key in PARTITIONS}
    record = {
        'schema_version': 2, 'run_id': uuid.uuid4().hex,
        'created_at': datetime.now(timezone.utc).isoformat(),
        'dataset_id': dataset_id(clean), 'seed': report['seed'],
        'augmentation': dict(report['augmentation']),
        'split_sizes': dict(report['split_sizes']),
        'runs': json.loads(json.dumps(report['runs'], allow_nan=False)),
        'manifest': {
            'kind': 'baseline-comparison', 'input_id': exact_dataset_id(clean),
            'data_source': report.get('source', 'Source not recorded'),
            'split_ids': split_ids, 'split_digest': digest(split_ids),
            **json.loads(json.dumps(report['provenance'], allow_nan=False)),
        },
    }
    validate_record(record)
    attach_reviews(record, clean)
    return record


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _keys(value, expected):
    _require(isinstance(value, dict) and set(value) == set(expected), 'Record fields do not match the saved-run contract.')


def _hash(value):
    return isinstance(value, str) and HEX_ID.fullmatch(value) is not None


def _finite_tree(value, depth=0):
    _require(depth <= 12, 'Record nesting exceeds the supported limit.')
    if isinstance(value, dict):
        for key, child in value.items():
            _require(isinstance(key, str) and len(key) <= 100, 'Invalid record key.')
            _finite_tree(child, depth + 1)
    elif isinstance(value, list):
        _require(len(value) <= 50000, 'Record list exceeds 50,000 entries.')
        for child in value:
            _finite_tree(child, depth + 1)
    elif isinstance(value, str):
        _require(len(value) <= 256, 'Record string exceeds the supported limit.')
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        _require(-1e9 <= value <= 1e9 and math.isfinite(value), 'Record contains a non-finite or out-of-bounds number.')
    else:
        _require(value is None or isinstance(value, bool), 'Unsupported record value.')


def _validate_evaluation(evaluation, ids):
    _keys(evaluation, ('n', 'accuracy', 'macro_f1', 'classes', 'confusion_matrix', 'predictions', 'evidence_version', 'evidence'))
    _require(evaluation['evidence_version'] == 1 and isinstance(evaluation['evidence'], list), 'Missing exact review evidence.')
    evidence = evaluation['evidence']
    _require(len(evidence) == len(ids), 'Evidence count does not match its partition.')
    for item in evidence:
        _keys(item, ('review_id', 'label', 'prediction'))
        _require(type(item['label']) is int and item['label'] in (0, 1) and
                 type(item['prediction']) is int and item['prediction'] in (0, 1), 'Evidence needs integer binary labels.')
    _require([item['review_id'] for item in evidence] == ids, 'Evidence IDs do not match ordered split membership.')
    expected = prediction_report([item['label'] for item in evidence], [item['prediction'] for item in evidence])
    for key, value in expected.items():
        _require(evaluation[key] == value, f'Stored {key} disagrees with the recorded evidence.')


def validate_record(record):
    """Validate all consumers' assumptions before the record reaches the UI."""
    _finite_tree(record)
    _keys(record, ('schema_version', 'run_id', 'created_at', 'dataset_id', 'seed', 'augmentation', 'split_sizes', 'runs', 'manifest'))
    _require(type(record['schema_version']) is int and record['schema_version'] == 2, 'Unsupported saved-run schema. Use a compatible SignalLab version.')
    _require(isinstance(record['run_id'], str) and re.fullmatch('[0-9a-f]{32}', record['run_id']) is not None, 'Invalid run identity.')
    try:
        created = datetime.fromisoformat(record['created_at'])
        _require(created.utcoffset() is not None, 'Saved run timestamp needs a timezone.')
    except (TypeError, ValueError) as exc:
        raise ValueError('Invalid saved run timestamp.') from exc
    _require(_hash(record['dataset_id']) and type(record['seed']) is int and record['seed'] == 42, 'Unsupported dataset identity or seed.')
    _keys(record['augmentation'], ('method', 'ratio'))
    ratio = record['augmentation']['ratio']
    _require(type(ratio) in (int, float) and 0 <= ratio <= 1 and
             record['augmentation']['method'] == 'training-only prefix template', 'Unsupported augmentation settings.')
    manifest = record['manifest']
    fields = ('kind', 'input_id', 'preprocessing', 'split_policy', 'split_ids', 'split_digest', 'parameters', 'dependencies', 'python', 'git_commit', 'source_hashes')
    _keys(manifest, (*fields, 'data_source') if isinstance(manifest, dict) and 'data_source' in manifest else fields)
    _require(manifest.get('data_source', 'Source not recorded') in ('Authored demo', 'Upload CSV', 'Local review dataset', 'Source not recorded'), 'Unsupported dataset source label.')
    _require(manifest['kind'] == 'baseline-comparison' and manifest['preprocessing'] == PREPROCESSING and
             manifest['split_policy'] == SPLIT_POLICY, 'Unsupported preprocessing or split policy. Use the original app version.')
    _keys(manifest['split_ids'], PARTITIONS)
    _keys(record['split_sizes'], PARTITIONS)
    all_ids = []
    for key in PARTITIONS:
        ids = manifest['split_ids'][key]
        _require(isinstance(ids, list) and 1 <= len(ids) <= 50000 and all(_hash(item) for item in ids), 'Invalid split membership.')
        _require(type(record['split_sizes'][key]) is int and record['split_sizes'][key] == len(ids), 'Split size disagrees with membership.')
        all_ids.extend(ids)
    _require(16 <= len(all_ids) <= 50000 and len(set(all_ids)) == len(all_ids), 'Partition overlap or invalid dataset size.')
    _require(manifest['input_id'] == digest(sorted(all_ids)) and manifest['split_digest'] == digest(manifest['split_ids']), 'Dataset or split digest disagrees with membership.')
    _keys(manifest['parameters'], ('vectorizer', 'classifier'))
    _keys(manifest['parameters']['vectorizer'], ('ngram_range', 'min_df', 'max_features', 'sublinear_tf'))
    _keys(manifest['parameters']['classifier'], ('max_iter', 'random_state', 'class_weight'))
    _require(manifest['parameters'] == json.loads(json.dumps(BASELINE_PARAMETERS)), 'Unsupported baseline configuration.')
    _keys(manifest['dependencies'], ('pandas', 'numpy', 'scikit-learn'))
    _require(all(isinstance(v, str) and v for v in manifest['dependencies'].values()) and
             isinstance(manifest['python'], str) and manifest['python'] and manifest['git_commit'] is None, 'Invalid runtime provenance.')
    _keys(manifest['source_hashes'], ('core.py', 'analysis.py', 'runs.py'))
    _require(all(_hash(v) for v in manifest['source_hashes'].values()), 'Invalid source provenance.')
    _require(isinstance(record['runs'], list) and len(record['runs']) == 2, 'A baseline comparison needs both conditions.')
    for index, run in enumerate(record['runs']):
        _keys(run, ('name', 'training_rows', 'validation', 'test'))
        expected_rows = record['split_sizes']['train'] + (int(record['split_sizes']['train'] * ratio) if index else 0)
        _require(run['name'] == BASELINE_NAMES[index] and type(run['training_rows']) is int and run['training_rows'] == expected_rows, 'Baseline condition or training count is invalid.')
        for key in ('validation', 'test'):
            _validate_evaluation(run[key], manifest['split_ids'][key])
    for key in ('validation', 'test'):
        _require([e['label'] for e in record['runs'][0][key]['evidence']] ==
                 [e['label'] for e in record['runs'][1][key]['evidence']], 'Conditions disagree about true labels.')
    return record


def attach_reviews(record, frame):
    validate_record(record)
    clean = validate_reviews(frame)
    _require(exact_dataset_id(clean) == record['manifest']['input_id'] and dataset_id(clean) == record['dataset_id'],
             'This saved run needs its exact original dataset. Select it or upload the same CSV to inspect review text.')
    splits = make_splits(clean)
    actual = {key: [review_id(row.text, row.label) for row in getattr(splits, key).itertuples()] for key in PARTITIONS}
    _require(actual == record['manifest']['split_ids'], 'Current split membership differs. Use the original app version or run a new experiment.')
    return splits


def save_record(record, root=None):
    validate_record(record)
    encoded = json.dumps(record, ensure_ascii=False, allow_nan=False, indent=2).encode('utf-8')
    _require(len(encoded) <= MAX_REPORT_BYTES, 'Report size exceeds 16 MiB. Use a smaller dataset.')
    root = Path(root) if root is not None else RUNS_ROOT
    _require(not root.is_symlink(), 'Saved-run root must not be a symlink.')
    root.mkdir(parents=True, exist_ok=True)
    destination = root / record['run_id']
    destination.mkdir(exist_ok=False)
    temporary = destination / 'report.pending'
    with temporary.open('xb') as file:
        file.write(encoded)
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary, destination / 'report.json')
    return record['run_id']


def _unique_keys(pairs):
    result = {}
    for key, value in pairs:
        _require(key not in result, 'Duplicate JSON keys are not supported.')
        result[key] = value
    return result


def _read_document(root, run_id):
    _require(isinstance(run_id, str) and re.fullmatch(r'[A-Za-z0-9_-]{1,80}', run_id) is not None, 'Invalid run directory name.')
    # Directory descriptors and O_NOFOLLOW also close the directory-symlink race.
    flags = os.O_RDONLY | os.O_NOFOLLOW
    root_fd = os.open(root, flags | os.O_DIRECTORY)
    try:
        run_fd = os.open(run_id, flags | os.O_DIRECTORY, dir_fd=root_fd)
        try:
            report_fd = os.open('report.json', flags | os.O_NONBLOCK, dir_fd=run_fd)
            with os.fdopen(report_fd, 'rb') as file:
                info = os.fstat(file.fileno())
                _require(stat.S_ISREG(info.st_mode) and info.st_size <= MAX_REPORT_BYTES, 'Report size/type is unsupported (maximum 16 MiB).')
                content = file.read(MAX_REPORT_BYTES + 1)
                _require(len(content) <= MAX_REPORT_BYTES, 'Report size exceeds 16 MiB.')
        finally:
            os.close(run_fd)
    finally:
        os.close(root_fd)
    try:
        return json.loads(content, object_pairs_hook=_unique_keys,
                          parse_constant=lambda value: (_ for _ in ()).throw(ValueError('Non-finite JSON number.')))
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ValueError('Report JSON is unreadable. Preserve it and save a new experiment.') from exc


def read_record(root, run_id):
    try:
        record = _read_document(Path(root), run_id)
    except OSError as exc:
        if isinstance(exc, FileNotFoundError):
            raise
        raise ValueError('Cannot safely open this saved run. Check permissions and avoid symlinks.') from exc
    validate_record(record)
    _require(record['run_id'] == run_id, 'Record identity does not match its directory.')
    return record


def discover_runs(root=None, limit=MAX_ENTRIES):
    """Explicit bounded refresh; summaries only, never model deserialisation."""
    root = Path(root) if root is not None else RUNS_ROOT
    _require(type(limit) is int and 1 <= limit <= MAX_ENTRIES, 'Use a discovery limit between 1 and 50.')
    _require(not root.is_symlink(), 'Saved-run root must not be a symlink.')
    if not root.exists():
        return [], False
    with os.scandir(root) as iterator:
        candidates = list(itertools.islice(iterator, limit + 1))
    truncated = len(candidates) > limit
    entries = []
    for candidate in candidates[:limit]:
        if not candidate.is_dir(follow_symlinks=False) or candidate.is_symlink():
            continue
        entry = {'run_id': candidate.name, 'status': 'Invalid', 'created_at': '', 'input_id': '', 'detail': ''}
        try:
            document = _read_document(root, candidate.name)
            if isinstance(document, dict) and document.get('schema_version') == 1:
                entry.update(status='Legacy', detail='No complete saved-run provenance. Existing aggregate views remain available; save a new baseline for analysis.')
            elif isinstance(document, dict) and document.get('schema_version') != 2:
                entry.update(status='Incompatible', detail='Unsupported schema. Open with its original SignalLab version.')
            else:
                record = validate_record(document)
                _require(record['run_id'] == candidate.name, 'Record identity does not match its directory.')
                entry.update(status='Ready', created_at=record['created_at'], input_id=record['manifest']['input_id'], detail='Numeric results available; attach the exact dataset for review text.')
        except FileNotFoundError:
            entry.update(status='Incomplete', detail='No completed report was published. Preserve this directory and save a new experiment.')
        except (OSError, ValueError, TypeError, KeyError, OverflowError) as exc:
            entry['detail'] = f'{type(exc).__name__}: report failed validation. Preserve it and save a new experiment.'
        entries.append(entry)
    return sorted(entries, key=lambda entry: (entry['created_at'], entry['run_id']), reverse=True), truncated
