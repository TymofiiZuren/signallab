"""Frozen local research protocols, not proof of unseen or independent data.

Only development membership/labels and text-only exclusion hashes are persisted.
Final-cohort text and labels never leave construction; callers retain that CSV.
"""
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import io
import json
import os
from pathlib import Path
import platform
import re
import stat
import uuid

import pandas as pd

from signallab.core import (
    PREPROCESSING, SEED, SPLIT_POLICY, Splits, canonical_review_key,
    exact_dataset_id, load_csv, make_splits, review_id, validate_reviews,
)

STUDIES_ROOT = Path(__file__).resolve().parents[1] / 'artifacts/studies'
MAX_CSV_BYTES = 10 * 1024 * 1024
MAX_PROTOCOL_BYTES = 16 * 1024 * 1024
PARTITIONS = ('train', 'validation', 'test')
KIND = 'synthetic-study-protocol'
DECLARATION = 'Operator declares this final cohort reserved and not inspected for model selection.'
QUESTION = ('Can the pinned local generator produce at least 30 meaning- and polarity-preserving '
            'human-accepted paraphrases, including at least 10 per class, from at most 50 fixed '
            'training parents within the recorded resource guard?')
PROMPT = ('Paraphrase the customer review below in English. Preserve its meaning, sentiment, facts '
          'and numbers. Return only the paraphrase.\nReview: {text}\nParaphrase:')
POLICY = {
    'question': QUESTION,
    'later_study_metric': 'macro_f1',
    'selection': 'sha256-selection-v1-seed42-balanced-interleave-0-1-no-replenishment',
    'generator': {'model': 'google/flan-t5-small',
                  'revision': '0fc9ddf78a1e988dac52e2dac162b0ede4fd74ab',
                  'device': 'cpu', 'dtype': 'float32', 'seed': 42,
                  'fast_tokenizer': True, 'safetensors': True, 'trust_remote_code': False,
                  'local_files_only': True, 'max_input_tokens': 128, 'truncate_input': False,
                  'do_sample': False, 'num_beams': 1, 'num_return_sequences': 1,
                  'max_new_tokens': 128, 'calls_per_parent': 1,
                  'prompt_template': PROMPT, 'prompt_digest': hashlib.sha256(PROMPT.encode()).hexdigest()},
    'resources': {'wall_seconds': 1800, 'rss_bytes': 4 * 1024**3,
                  'rss_sample_seconds': 0.25, 'termination_grace_seconds': 2,
                  'measurement': 'sampled-worker-rss-not-hard-memory-containment',
                  'unknown_resource_qualifies': False},
    'feasibility': {'accepted_min': 30, 'accepted_per_class_min': 10,
                    'generation_complete': True, 'all_generated_reviewed': True},
    'review': {'verdicts': ['accept', 'reject', 'ambiguous'], 'reason_required': True,
               'rubric': ['meaning', 'polarity', 'facts_and_numbers', 'english', 'meaningful_rewording'],
               'automatic_exclusion_overridable': False, 'max_revisions': 200},
}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def text_fingerprint(text):
    """Label-independent exact canonical collision key; not semantic deduplication."""
    return hashlib.sha256(canonical_review_key(text).encode('utf-8')).hexdigest()


def _hash(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


def _protocol_id(value):
    _require(isinstance(value, str) and re.fullmatch('[0-9a-f]{32}', value) is not None,
             'Invalid protocol ID. Use the full ID returned when freezing the study.')
    return value


def _keys(value, names):
    _require(isinstance(value, dict) and set(value) == set(names), 'Protocol fields do not match schema version 1.')


def _source(value):
    _require(isinstance(value, str) and bool(value.strip()) and len(value) <= 2000,
             'Record a non-empty source description of at most 2,000 characters for each cohort.')
    return value.strip()


def _settings(max_parents, threads):
    _require(type(max_parents) is int and 1 <= max_parents <= 50, 'max_parents must be an integer 1–50.')
    _require(type(threads) is int and 1 <= threads <= 8, 'threads must be an integer 1–8.')


def _select(split_ids, rows, maximum):
    by_id = {row['review_id']: row for row in rows}
    choices = []
    for label in (0, 1):
        ids = [rid for rid in split_ids['train'] if by_id[rid]['label'] == label]
        ids.sort(key=lambda rid: hashlib.sha256(f'selection-v1|42|{rid}'.encode()).hexdigest())
        choices.append(ids[:(maximum + (1 if label == 0 else 0)) // 2])
    # Short classes are not replenished from the other class.
    return [group[i] for i in range(max(map(len, choices), default=0)) for group in choices if i < len(group)]


def build_protocol(development, final_cohort, *, source, final_source, attest_final_reserved,
                   max_parents=50, threads=2):
    """Validate and freeze rules in memory; no model load, download or disk write."""
    _settings(max_parents, threads)
    source, final_source = _source(source), _source(final_source)
    _require(attest_final_reserved is True, 'Explicit final-cohort reservation declaration is required.')
    _require(isinstance(development, pd.DataFrame) and isinstance(final_cohort, pd.DataFrame),
             'Supply development and reserved final reviews as pandas DataFrames, or use freeze_protocol with CSV paths.')
    dev, final = validate_reviews(development), validate_reviews(final_cohort)
    rows = [{'review_id': review_id(row.text, row.label), 'label': int(row.label),
             'fingerprint': text_fingerprint(row.text)} for row in dev.itertuples()]
    rows.sort(key=lambda row: row['review_id'])
    final_hashes = sorted(text_fingerprint(text) for text in final.text)
    overlap = {row['fingerprint'] for row in rows} & set(final_hashes)
    _require(not overlap, f'Development/final text overlap detected ({len(overlap)} canonical reviews). '
                         'Choose a disjoint reserved cohort; no protocol was created.')
    splits = make_splits(dev)
    ids = {name: [review_id(row.text, row.label) for row in getattr(splits, name).itertuples()]
           for name in PARTITIONS}
    record = {
        'schema_version': 1, 'kind': KIND, 'protocol_id': uuid.uuid4().hex,
        'created_at': datetime.now(timezone.utc).isoformat(),
        'preprocessing': PREPROCESSING, 'split_policy': SPLIT_POLICY, 'seed': SEED,
        'development': {'input_id': exact_dataset_id(dev), 'source': source,
                        'rows': rows, 'split_ids': ids, 'split_digest': _digest(ids)},
        'final_cohort': {'input_id': exact_dataset_id(final), 'fingerprints': final_hashes,
                         'count': len(final), 'source': final_source, 'declaration': DECLARATION},
        'selection': {'max_parents': max_parents, 'parent_ids': _select(ids, rows, max_parents)},
        'threads': threads, 'policy': json.loads(json.dumps(POLICY)),
        'provenance': {'python': platform.python_version(), 'git_commit': None,
                       'dependencies': {name: version(name) for name in ('pandas', 'scikit-learn', 'torch', 'transformers', 'tokenizers')},
                       'source_hashes': {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                                         for name in ('core.py', 'study.py')}},
    }
    record['content_digest'] = _digest(record)
    validate_protocol(record)
    return record


def _bounded_tree(value, depth=0):
    _require(depth <= 12, 'Protocol nesting exceeds 12 levels.')
    if isinstance(value, dict):
        _require(len(value) <= 100, 'Protocol object has too many fields.')
        for key, child in value.items():
            _require(isinstance(key, str) and len(key) <= 100, 'Invalid protocol field name.')
            _bounded_tree(child, depth + 1)
    elif isinstance(value, list):
        _require(len(value) <= 50000, 'Protocol list exceeds 50,000 entries.')
        for child in value:
            _bounded_tree(child, depth + 1)
    elif isinstance(value, str):
        _require(len(value) <= 2000, 'Protocol string exceeds 2,000 characters.')
    else:
        _require(value is None or type(value) is bool or
                 (type(value) in (int, float) and -1e10 <= value <= 1e10),
                 'Protocol contains a non-finite or unsupported value.')


def validate_protocol(record):
    """Reject inconsistent or unsupported evidence before any consumer sees it.

    Hashes detect inconsistencies, not deliberate rewriting by the machine owner.
    """
    _bounded_tree(record)
    _keys(record, ('schema_version', 'kind', 'protocol_id', 'created_at', 'preprocessing',
                   'split_policy', 'seed', 'development', 'final_cohort', 'selection',
                   'threads', 'policy', 'provenance', 'content_digest'))
    _require(type(record['schema_version']) is int and record['schema_version'] == 1 and record['kind'] == KIND,
             'Unsupported study protocol kind/version. Use a compatible SignalLab reader; preserve the file.')
    _protocol_id(record['protocol_id'])
    try:
        created = datetime.fromisoformat(record['created_at'])
        _require(created.utcoffset() is not None, 'Protocol timestamp requires a timezone.')
    except (TypeError, ValueError) as exc:
        raise ValueError('Protocol timestamp is invalid.') from exc
    _require(record['preprocessing'] == PREPROCESSING and record['split_policy'] == SPLIT_POLICY and
             type(record['seed']) is int and record['seed'] == 42, 'Unsupported preprocessing or split policy.')
    _require(_hash(record['content_digest']) and record['content_digest'] ==
             _digest({k: v for k, v in record.items() if k != 'content_digest'}), 'Protocol content digest does not match.')
    dev, final, selection = record['development'], record['final_cohort'], record['selection']
    _keys(dev, ('input_id', 'source', 'rows', 'split_ids', 'split_digest'))
    _keys(final, ('input_id', 'fingerprints', 'count', 'source', 'declaration'))
    _keys(selection, ('max_parents', 'parent_ids'))
    _source(dev['source'])
    _source(final['source'])
    _settings(selection['max_parents'], record['threads'])
    _require(isinstance(dev['rows'], list) and 16 <= len(dev['rows']) <= 50000, 'Invalid development row count.')
    for row in dev['rows']:
        _keys(row, ('review_id', 'label', 'fingerprint'))
        _require(_hash(row['review_id']) and _hash(row['fingerprint']) and type(row['label']) is int and
                 row['label'] in (0, 1), 'Invalid development identity, fingerprint or label.')
    ids = [row['review_id'] for row in dev['rows']]
    hashes = [row['fingerprint'] for row in dev['rows']]
    _require(ids == sorted(set(ids)) and len(set(hashes)) == len(hashes), 'Development identities/fingerprints must be unique and ordered.')
    _require(all(sum(row['label'] == label for row in dev['rows']) >= 8 for label in (0, 1)),
             'Development input requires at least 8 reviews per class.')
    _require(dev['input_id'] == _digest(ids), 'Development input identity disagrees with its rows.')
    _keys(dev['split_ids'], PARTITIONS)
    flattened = []
    for name in PARTITIONS:
        part = dev['split_ids'][name]
        _require(isinstance(part, list) and bool(part) and all(_hash(rid) for rid in part), 'Invalid ordered partition membership.')
        flattened.extend(part)
    _require(len(flattened) == len(ids) and set(flattened) == set(ids) and len(set(flattened)) == len(flattened),
             'Frozen partitions must be disjoint and exhaust the exact development input.')
    remainder = (2 * len(ids) + 4) // 5  # ceil(40%); matches the recorded 60/20/20 policy.
    sizes = (len(ids) - remainder, remainder // 2, (remainder + 1) // 2)
    _require(tuple(len(dev['split_ids'][name]) for name in PARTITIONS) == sizes,
             'Frozen partition sizes disagree with the 60/20/20 policy.')
    by_id = {row['review_id']: row['label'] for row in dev['rows']}
    _require(all({by_id[rid] for rid in dev['split_ids'][name]} == {0, 1} for name in PARTITIONS),
             'Each frozen partition needs both sentiment classes.')
    _require(dev['split_digest'] == _digest(dev['split_ids']), 'Frozen partition digest does not match.')
    _require(_hash(final['input_id']) and type(final['count']) is int and 16 <= final['count'] <= 50000 and
             final['declaration'] == DECLARATION, 'Invalid reserved final cohort or missing declaration.')
    fingerprints = final['fingerprints']
    _require(isinstance(fingerprints, list) and len(fingerprints) == final['count'] and
             all(_hash(v) for v in fingerprints) and fingerprints == sorted(set(fingerprints)), 'Invalid final exclusion fingerprints.')
    _require(not set(hashes) & set(fingerprints), 'Development/final text overlap detected in protocol.')
    _require(selection['parent_ids'] == _select(dev['split_ids'], dev['rows'], selection['max_parents']),
             'Parent selection differs from the frozen training-only policy.')
    # JSON distinguishes booleans from integers, unlike Python dict equality.
    _require(_digest(record['policy']) == _digest(POLICY), 'Unsupported generation, review or resource policy.')
    provenance = record['provenance']
    _keys(provenance, ('python', 'git_commit', 'dependencies', 'source_hashes'))
    _require(isinstance(provenance['python'], str) and bool(provenance['python']) and provenance['git_commit'] is None,
             'Invalid runtime provenance.')
    _keys(provenance['dependencies'], ('pandas', 'scikit-learn', 'torch', 'transformers', 'tokenizers'))
    _require(all(isinstance(v, str) and bool(v) for v in provenance['dependencies'].values()), 'Invalid dependency provenance.')
    _keys(provenance['source_hashes'], ('core.py', 'study.py'))
    _require(all(_hash(v) for v in provenance['source_hashes'].values()), 'Invalid source provenance.')


def attach_development(record, frame):
    """Reconstruct recorded order by exact ID, never call the splitter again."""
    validate_protocol(record)
    _require(isinstance(frame, pd.DataFrame), 'Supply the exact original development reviews as a pandas DataFrame.')
    clean = validate_reviews(frame)
    _require(exact_dataset_id(clean) == record['development']['input_id'],
             'This study needs its exact original development CSV; text or labels differ.')
    lookup = {review_id(row.text, row.label): (row.text, int(row.label)) for row in clean.itertuples()}
    actual = {rid: (label, text_fingerprint(text)) for rid, (text, label) in lookup.items()}
    expected = {row['review_id']: (row['label'], row['fingerprint']) for row in record['development']['rows']}
    _require(actual == expected, 'Exact development fingerprints or labels disagree with the protocol.')
    return Splits(*(pd.DataFrame([lookup[rid] for rid in record['development']['split_ids'][name]],
                                 columns=['text', 'label']) for name in PARTITIONS))


def load_study_csv(source):
    """Bound raw bytes before pandas, including CLI inputs; never truncate a CSV."""
    if isinstance(source, (str, os.PathLike)):
        fd = os.open(source, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(fd, 'rb') as file:
            info = os.fstat(file.fileno())
            _require(stat.S_ISREG(info.st_mode) and info.st_size <= MAX_CSV_BYTES,
                     'Study CSV must be a regular file of at most 10 MiB.')
            content = file.read(MAX_CSV_BYTES + 1)
    else:
        _require(hasattr(source, 'read'), 'Supply a CSV path or readable CSV stream.')
        content = source.read(MAX_CSV_BYTES + 1)
        if isinstance(content, str):
            content = content.encode('utf-8')
    _require(isinstance(content, bytes) and len(content) <= MAX_CSV_BYTES,
             'Study CSV exceeds 10 MiB or is not a readable byte/text stream.')
    # Reuse public CSV/label/row validation only after enforcing the byte bound.
    return load_csv(io.BytesIO(content))


def freeze_protocol(development_csv, final_csv, *, source, final_source,
                    attest_final_reserved, max_parents=50, threads=2, root=None):
    """Freeze two explicit CSVs; validation failure creates no artifact directory."""
    _settings(max_parents, threads)
    _source(source)
    _source(final_source)
    _require(attest_final_reserved is True, 'Explicit final-cohort reservation declaration is required.')
    record = build_protocol(load_study_csv(development_csv), load_study_csv(final_csv),
                            source=source, final_source=final_source,
                            attest_final_reserved=attest_final_reserved,
                            max_parents=max_parents, threads=threads)
    save_protocol(record, root)
    return record


def save_protocol(record, root=None):
    """Publish once without replacement. Preserve pending evidence on any failure.

    POSIX descriptor-relative writes pin the immediate root/directory. Hard-link
    publication is atomic and refuses an existing final name, unlike replace().
    Source directories are trusted local configuration, not an arbitrary upload.
    """
    validate_protocol(record)
    content = json.dumps(record, sort_keys=True, ensure_ascii=False, separators=(',', ':'),
                         allow_nan=False).encode('utf-8')
    _require(len(content) <= MAX_PROTOCOL_BYTES, 'Protocol exceeds 16 MiB. Use smaller study inputs.')
    root = Path(root) if root is not None else STUDIES_ROOT
    _require(not root.is_symlink(), 'Study root must not be a symlink.')
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_DIRECTORY
    root_fd = os.open(root, flags)
    try:
        os.mkdir(record['protocol_id'], mode=0o700, dir_fd=root_fd)
        os.fsync(root_fd)
        protocol_fd = os.open(record['protocol_id'], flags, dir_fd=root_fd)
        try:
            fd = os.open('protocol.pending', os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                         0o600, dir_fd=protocol_fd)
            with os.fdopen(fd, 'wb') as file:
                file.write(content)
                file.flush()
                os.fsync(file.fileno())
            os.link('protocol.pending', 'protocol.json', src_dir_fd=protocol_fd,
                    dst_dir_fd=protocol_fd, follow_symlinks=False)
            os.fsync(protocol_fd)
        finally:
            os.close(protocol_fd)
    finally:
        os.close(root_fd)
    return record['protocol_id']


def _unique_keys(pairs):
    result = {}
    for key, value in pairs:
        _require(key not in result, 'Duplicate JSON keys are not supported in a study protocol.')
        result[key] = value
    return result


def _reject_constant(_value):
    raise ValueError('Protocol contains a non-finite JSON number.')


def read_protocol(root, protocol_id):
    """Read a bounded published regular file; never follow an artifact symlink."""
    _protocol_id(protocol_id)
    flags = os.O_RDONLY | os.O_NOFOLLOW
    try:
        root_fd = os.open(root, flags | os.O_DIRECTORY)
        try:
            directory_fd = os.open(protocol_id, flags | os.O_DIRECTORY, dir_fd=root_fd)
            try:
                fd = os.open('protocol.json', flags | os.O_NONBLOCK, dir_fd=directory_fd)
                with os.fdopen(fd, 'rb') as file:
                    info = os.fstat(file.fileno())
                    _require(stat.S_ISREG(info.st_mode) and info.st_size <= MAX_PROTOCOL_BYTES,
                             'Study protocol must be a regular file of at most 16 MiB.')
                    content = file.read(MAX_PROTOCOL_BYTES + 1)
                    _require(len(content) <= MAX_PROTOCOL_BYTES, 'Study protocol exceeds 16 MiB.')
            finally:
                os.close(directory_fd)
        finally:
            os.close(root_fd)
    except FileNotFoundError:
        raise
    except OSError as exc:
        raise ValueError('Cannot safely read study protocol. Check permissions and avoid symlinks; preserve existing files.') from exc
    try:
        record = json.loads(content, object_pairs_hook=_unique_keys, parse_constant=_reject_constant)
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ValueError('Study protocol JSON is unreadable. Preserve it and use a compatible valid protocol.') from exc
    validate_protocol(record)
    _require(record['protocol_id'] == protocol_id, 'Protocol identity disagrees with its directory.')
    return record
