"""Local data-bearing pilot evidence and append-only human review.

This module never loads a model, generates text or trains. The supervisor
owns worker execution; it must reap its child before attesting terminal status.
Hashes detect inconsistencies, not an owner rewriting all local evidence.
"""
from contextlib import contextmanager
from datetime import datetime, timezone
from dataclasses import dataclass
import fcntl
import hashlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import platform
import re
import stat
import uuid

import pandas as pd

from signallab import study
from signallab.core import Splits, review_id

SYNTHETIC_ROOT = Path(__file__).resolve().parents[1] / 'artifacts/synthetic'
MAX_BYTES = 16 * 1024 * 1024
MAX_REVISIONS = 200
ASSET_NAMES = {'config.json', 'generation_config.json', 'tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json', 'model.safetensors'}
REQUIRED_ASSETS = {'config.json', 'tokenizer.json', 'tokenizer_config.json', 'model.safetensors'}
ERROR_CODES = {'worker_error', 'worker_exit', 'invalid_output', 'interrupted'}
TERMINAL_STATES = {'complete', 'failed', 'interrupted', 'stopped_budget'}
TERMINATION_CODES = {
    'complete': {'completed'},
    'failed': {'worker_error', 'worker_exit', 'invalid_output', 'model_load_error', 'rss_unavailable'},
    'interrupted': {'operator_interrupt', 'recovered_interruption'},
    'stopped_budget': {'wall_limit', 'rss_limit'},
}


class ReviewConflict(ValueError):
    """Review request was not applied; reload or retry the exact committed request."""


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _hash(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


def _id(value):
    _require(isinstance(value, str) and re.fullmatch('[0-9a-f]{32}', value) is not None, 'Use a full valid pilot or decision ID.')
    return value


def _keys(value, fields):
    _require(isinstance(value, dict) and set(value) == set(fields), 'Synthetic evidence fields do not match version 1.')


def _integer(value, low, high):
    return type(value) is int and low <= value <= high


def _tree(value, depth=0):
    _require(depth <= 12, 'Synthetic evidence nesting exceeds 12 levels.')
    if isinstance(value, dict):
        _require(len(value) <= 500 and all(isinstance(k, str) and len(k) <= 100 for k in value), 'Invalid synthetic object size/keys.')
        for child in value.values():
            _tree(child, depth + 1)
    elif isinstance(value, list):
        _require(len(value) <= 200, 'Synthetic evidence list is too long.')
        for child in value:
            _tree(child, depth + 1)
    elif isinstance(value, str):
        _require(len(value) <= 20000, 'Synthetic evidence string exceeds 20,000 characters.')
    else:
        _require(value is None or type(value) is bool or
                 (type(value) in (int, float) and abs(value) <= 1e10 and math.isfinite(value)),
                 'Synthetic evidence has a non-finite or unsupported value.')


def _document(kind, **fields):
    value = dict(schema_version=1, kind=kind, created_at=datetime.now(timezone.utc).isoformat(), **fields)
    value['content_digest'] = _digest(value)
    return value


def _envelope(value, kind, fields):
    _tree(value)
    _keys(value, {'schema_version', 'kind', 'created_at', 'content_digest', *fields})
    _require(type(value['schema_version']) is int and value['schema_version'] == 1 and value['kind'] == kind,
             'Unsupported synthetic evidence kind/version; preserve it and use a compatible reader.')
    try:
        _require(datetime.fromisoformat(value['created_at']).utcoffset() is not None, 'Timestamp requires a timezone.')
    except (ValueError, TypeError) as exc:
        raise ValueError('Invalid synthetic evidence timestamp.') from exc
    _require(_hash(value['content_digest']) and value['content_digest'] ==
             _digest({k: v for k, v in value.items() if k != 'content_digest'}), 'Synthetic evidence digest does not match.')


def _unique(pairs):
    result = {}
    for key, value in pairs:
        _require(key not in result, 'Duplicate JSON keys are unsupported.')
        result[key] = value
    return result


def _constant(_):
    raise ValueError('Non-finite JSON is unsupported.')


def _filename(name):
    _require(re.fullmatch(r'[A-Za-z0-9_.-]{1,120}', name) is not None and name not in ('.', '..'), 'Invalid evidence filename.')


def _read(fd, name, *, optional=False, limit=MAX_BYTES):
    _filename(name)
    try:
        file_fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=fd)
    except FileNotFoundError:
        if optional:
            return None
        raise
    except OSError as exc:
        raise ValueError('Cannot safely read synthetic evidence; avoid symlinks and check permissions.') from exc
    with os.fdopen(file_fd, 'rb') as file:
        info = os.fstat(file.fileno())
        _require(stat.S_ISREG(info.st_mode) and info.st_size <= limit, 'Synthetic evidence file type/size exceeds its bound.')
        data = file.read(limit + 1)
        _require(len(data) <= limit, 'Synthetic evidence exceeds its byte bound.')
    try:
        value = json.loads(data, object_pairs_hook=_unique, parse_constant=_constant)
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ValueError('Synthetic evidence JSON is unreadable. Preserve the damaged file.') from exc
    _tree(value)
    return value


def _publish(fd, name, value, *, limit=MAX_BYTES):
    _filename(name)
    _tree(value)
    data = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode()
    _require(len(data) <= limit, 'Synthetic evidence exceeds its byte bound.')
    pending = f'.{name}.{uuid.uuid4().hex}.pending'
    file_fd = os.open(pending, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=fd)
    with os.fdopen(file_fd, 'wb') as file:
        file.write(data)
        file.flush()
        os.fsync(file.fileno())
    os.link(pending, name, src_dir_fd=fd, dst_dir_fd=fd, follow_symlinks=False)
    os.fsync(fd)


@contextmanager
def _directory(root, pilot_id):
    _id(pilot_id)
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    try:
        root_fd = os.open(root, flags)
        try:
            fd = os.open(pilot_id, flags, dir_fd=root_fd)
        finally:
            os.close(root_fd)
    except FileNotFoundError:
        raise
    except OSError as exc:
        raise ValueError('Cannot safely open the pilot directory; check permissions and avoid symlinks.') from exc
    try:
        yield fd
    finally:
        os.close(fd)


@contextmanager
def _lock(fd, name, *, shared=False, conflict=ValueError):
    try:
        lock_fd = os.open(name, os.O_RDWR | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=fd)
    except OSError as exc:
        raise ValueError('Pilot lock is missing or unsafe. Preserve evidence and check permissions/symlinks.') from exc
    try:
        _require(stat.S_ISREG(os.fstat(lock_fd).st_mode), 'Evidence lock must be a regular file.')
        try:
            fcntl.flock(lock_fd, (fcntl.LOCK_SH if shared else fcntl.LOCK_EX) | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise conflict('Evidence is busy in another process. No change saved; refresh and retry.') from exc
        # Closing the final descriptor releases flock. An explicit LOCK_UN here
        # would also unlock a descriptor inherited by a still-running worker.
        yield lock_fd
    finally:
        os.close(lock_fd)


def _context(fd, pilot_id, studies_root):
    pilot = _read(fd, 'pilot.json')
    fields = {'pilot_id', 'protocol_id', 'protocol_digest', 'parent_ids', 'model_assets', 'generator', 'threads', 'provenance'}
    if isinstance(pilot, dict) and 'execution' in pilot:
        fields.add('execution')
    _envelope(pilot, 'synthetic-pilot', fields)
    if 'execution' in pilot:
        _execution(pilot['execution'])
    _require(pilot['pilot_id'] == pilot_id, 'Pilot identity disagrees with its directory.')
    _id(pilot['protocol_id'])
    protocol = study.read_protocol(studies_root, pilot['protocol_id'])
    _require(pilot['protocol_digest'] == protocol['content_digest'] and
             pilot['parent_ids'] == protocol['selection']['parent_ids'], 'Pilot does not match its frozen protocol/selection.')
    _require(_digest(pilot['generator']) == _digest(protocol['policy']['generator']) and
             type(pilot['threads']) is int and pilot['threads'] == protocol['threads'], 'Pilot generator settings differ from its protocol.')
    _assets(pilot['model_assets'])
    provenance = pilot['provenance']
    _keys(provenance, {'python', 'dependencies', 'source_hashes'})
    _require(isinstance(provenance['python'], str) and bool(provenance['python']), 'Invalid pilot Python provenance.')
    _keys(provenance['dependencies'], {'pandas', 'torch', 'transformers', 'tokenizers'})
    _require(all(isinstance(v, str) and v for v in provenance['dependencies'].values()), 'Invalid pilot dependency provenance.')
    _keys(provenance['source_hashes'], {'core.py', 'study.py', 'synthetic.py'})
    _require(all(_hash(v) for v in provenance['source_hashes'].values()), 'Invalid pilot source provenance.')
    return pilot, protocol


def _assets(assets):
    _require(isinstance(assets, dict) and REQUIRED_ASSETS <= set(assets) <= ASSET_NAMES and
             all(_hash(v) for v in assets.values()), 'Record checksums for the required allowlisted model assets.')


def _execution(value):
    _keys(value, {'preparation_id', 'prepared_digest', 'environment'})
    _id(value['preparation_id'])
    _require(_hash(value['prepared_digest']), 'Invalid prepared-snapshot digest.')
    env = value['environment']
    _keys(env, {'python', 'dependencies', 'source_hashes'})
    _require(isinstance(env['python'], str) and bool(env['python']), 'Invalid execution Python identity.')
    _keys(env['dependencies'], {'torch', 'transformers', 'tokenizers', 'huggingface-hub'})
    _require(all(isinstance(v, str) and bool(v) for v in env['dependencies'].values()), 'Invalid execution dependencies.')
    _keys(env['source_hashes'], {'synthetic_cli.py', 'synthetic_worker.py'})
    _require(all(_hash(v) for v in env['source_hashes'].values()), 'Invalid execution source identities.')


def create_pilot(protocol_id, frame, model_assets, *, root=None, studies_root=study.STUDIES_ROOT, execution=None):
    """Allocate evidence only. Asset hashes are declarations, not model readiness."""
    protocol = study.read_protocol(studies_root, protocol_id)
    study.attach_development(protocol, frame)
    _assets(model_assets)
    if execution is not None:
        _execution(execution)
    pilot_id = uuid.uuid4().hex
    provenance = {'python': platform.python_version(),
                  'dependencies': {n: version(n) for n in ('pandas', 'torch', 'transformers', 'tokenizers')},
                  'source_hashes': {n: hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest()
                                    for n in ('core.py', 'study.py', 'synthetic.py')}}
    pilot = _document('synthetic-pilot', pilot_id=pilot_id, protocol_id=protocol_id,
                      protocol_digest=protocol['content_digest'], parent_ids=list(protocol['selection']['parent_ids']),
                      model_assets=dict(model_assets), generator=protocol['policy']['generator'],
                      threads=protocol['threads'], provenance=provenance,
                      **({'execution': json.loads(json.dumps(execution))} if execution is not None else {}))
    root = Path(root) if root is not None else SYNTHETIC_ROOT
    _require(not root.is_symlink(), 'Synthetic root must not be a symlink.')
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    root_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.mkdir(pilot_id, mode=0o700, dir_fd=root_fd)
        os.fsync(root_fd)
    finally:
        os.close(root_fd)
    with _directory(root, pilot_id) as fd:
        for name in ('generation.lock', 'review.lock'):
            lock_fd = os.open(name, os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=fd)
            os.fsync(lock_fd)
            os.close(lock_fd)
        _publish(fd, 'pilot.json', pilot)
    return pilot_id


def _exclusion_index(protocol):
    rows = {r['review_id']: r for r in protocol['development']['rows']}
    return rows, {r['fingerprint'] for r in rows.values()}, set(protocol['final_cohort']['fingerprints'])


def _exclusions(text, eos, parent, rows, originals, finals, owners):
    fingerprint = study.text_fingerprint(text)
    reasons = []
    if not text.strip():
        reasons.append('blank')
    if fingerprint == rows[parent]['fingerprint']:
        reasons.append('unchanged_parent')
    if fingerprint in originals:
        reasons.append('original_collision')
    if fingerprint in finals:
        reasons.append('final_collision')
    if not eos:
        reasons.append('incomplete_output')
    if fingerprint in owners:
        reasons.append('candidate_collision')
    # Ownership is automatic, independent of later human decisions.
    if not reasons:
        owners.add(fingerprint)
    return reasons


def _resource_status(resources, protocol):
    fields = {'elapsed_seconds', 'peak_rss_bytes', 'samples', 'method'}
    if isinstance(resources, dict) and ('observed_peak_rss_bytes' in resources or 'observed_samples' in resources):
        fields |= {'observed_peak_rss_bytes', 'observed_samples'}
    _keys(resources, fields)
    if 'observed_samples' in resources:
        _require(resources['method'] == 'unavailable', 'Retained partial observations require unknown RSS, not a second qualifying measurement.')
        observed, count = resources['observed_peak_rss_bytes'], resources['observed_samples']
        _require(_integer(count, 0, 10000000) and ((count == 0 and observed is None) or
                 (count > 0 and _integer(observed, 1, int(1e10)))), 'Invalid retained RSS observations.')
    elapsed, peak, samples = (resources[k] for k in ('elapsed_seconds', 'peak_rss_bytes', 'samples'))
    _require(type(elapsed) in (int, float) and 0 <= elapsed <= 1e7 and math.isfinite(elapsed) and
             _integer(samples, 0, 10000000), 'Invalid elapsed time or resource sample count.')
    if resources['method'] == 'unavailable':
        _require(peak is None and samples == 0, 'Unavailable RSS must be recorded as unknown, without invented samples.')
        return 'fail' if elapsed >= protocol['policy']['resources']['wall_seconds'] else 'unknown'
    _require(resources['method'] == 'macos-ps-rss-kib-v1' and _integer(peak, 1, int(1e10)) and samples > 0,
             'Invalid RSS method or samples. Record unavailable measurement as unknown.')
    limits = protocol['policy']['resources']
    return 'fail' if elapsed >= limits['wall_seconds'] or peak >= limits['rss_bytes'] else 'pass'


def _termination_code(state, resources, protocol, *, code=None, recovery_only=False):
    status = _resource_status(resources, protocol)
    if state == 'complete':
        _require(status == 'pass', 'Complete generation requires observed resources within the guard.')
        default = 'completed'
    elif state == 'interrupted':
        default = 'recovered_interruption' if recovery_only else 'operator_interrupt'
    elif state == 'stopped_budget':
        _require(status == 'fail', 'Budget stop requires recorded threshold evidence.')
        default = 'wall_limit' if resources['elapsed_seconds'] >= protocol['policy']['resources']['wall_seconds'] else 'rss_limit'
    else:
        default = 'rss_unavailable' if status == 'unknown' else 'worker_error'
    code = default if code is None else code
    _require(isinstance(code, str) and code in TERMINATION_CODES[state], 'Use a supported sanitized terminal error code for this state.')
    if code == 'wall_limit':
        _require(resources['elapsed_seconds'] >= protocol['policy']['resources']['wall_seconds'], 'Wall-limit code lacks threshold evidence.')
    if code == 'rss_limit':
        _require(resources['peak_rss_bytes'] is not None and resources['peak_rss_bytes'] >= protocol['policy']['resources']['rss_bytes'],
                 'RSS-limit code lacks threshold evidence.')
    return code


def _attempts(fd, pilot, protocol, terminal):
    attempts, candidates, references, owners = [], {}, {}, set()
    rows, originals, finals = _exclusion_index(protocol)
    unfinished = False
    for index, parent in enumerate(pilot['parent_ids']):
        names = (f'attempt-{index:02d}.started.json', f'attempt-{index:02d}.result.json')
        intent, result = (_read(fd, name, optional=True) for name in names)
        if intent is not None:
            _envelope(intent, 'synthetic-attempt-start', {'pilot_id', 'parent_id', 'index', 'input_tokens'})
            _require(intent['pilot_id'] == pilot['pilot_id'] and intent['parent_id'] == parent and
                     type(intent['index']) is int and intent['index'] == index and
                     _integer(intent['input_tokens'], 1, 128), 'Attempt intent violates frozen order or token bounds.')
            references[names[0]] = intent['content_digest']
        state = 'started' if intent else 'unattempted'
        if result is not None:
            state = result.get('state') if isinstance(result, dict) else None
            common = {'pilot_id', 'parent_id', 'index', 'state', 'intent_digest'}
            extra = {'input_tokens'} if state == 'skipped_length' else {'error_code'} if state == 'failed' else {'text', 'text_digest', 'candidate_id', 'fingerprint', 'output_tokens', 'eos', 'exclusions'}
            _require(state in ('generated', 'skipped_length', 'failed'), 'Invalid attempt result state.')
            _envelope(result, 'synthetic-attempt-result', common | extra)
            _require(result['pilot_id'] == pilot['pilot_id'] and result['parent_id'] == parent and
                     type(result['index']) is int and result['index'] == index, 'Attempt result identity/order mismatch.')
            if state == 'skipped_length':
                _require(intent is None and result['intent_digest'] is None and
                         _integer(result['input_tokens'], 129, 100000), 'Only intact overlength inputs may be skipped.')
            else:
                _require(intent is not None and result['intent_digest'] == intent['content_digest'], 'Result lacks its durable started intent.')
                if state == 'failed':
                    _require(isinstance(result['error_code'], str) and result['error_code'] in ERROR_CODES, 'Unsupported sanitized worker error code.')
                else:
                    text = result['text']
                    _require(isinstance(text, str) and len(text) <= 20000 and
                             _integer(result['output_tokens'], 0, 128) and type(result['eos']) is bool and
                             (result['output_tokens'] > 0 or not result['eos']), 'Invalid generated text/token/EOS contract.')
                    text_digest = hashlib.sha256(text.encode('utf-8')).hexdigest()
                    candidate_id = _digest([pilot['pilot_id'], parent, text_digest])
                    _require(result['text_digest'] == text_digest and result['candidate_id'] == candidate_id and
                             result['fingerprint'] == study.text_fingerprint(text), 'Candidate identity disagrees with its literal text.')
                    exclusions = _exclusions(text, result['eos'], parent, rows, originals, finals, owners)
                    _require(result['exclusions'] == exclusions, 'Candidate automatic exclusions disagree with frozen evidence.')
                    candidates[candidate_id] = dict(result, label=rows[parent]['label'])
            references[names[1]] = result['content_digest']
        _require(not unfinished or (intent is None and result is None), 'Attempt records violate frozen sequential order.')
        if result is None:
            unfinished = True
        if terminal is not None and state == 'started':
            state = 'interrupted'
        attempts.append({'parent_id': parent, 'state': state, 'intent': intent, 'result': result})
    return attempts, candidates, references


def _load(fd, pilot, protocol):
    started = _read(fd, 'generation.started.json', optional=True)
    if started is not None:
        _envelope(started, 'synthetic-generation-start', {'pilot_id', 'pilot_digest'})
        _require(started['pilot_id'] == pilot['pilot_id'] and started['pilot_digest'] == pilot['content_digest'],
                 'Generation owner marker disagrees with its pilot.')
    terminal = _read(fd, 'terminal.json', optional=True)
    attempts, candidates, references = _attempts(fd, pilot, protocol, terminal)
    _require(started is not None or not references, 'Attempt records have no durable generation owner marker.')
    if terminal is not None:
        _envelope(terminal, 'synthetic-pilot-terminal', {'pilot_id', 'pilot_digest', 'protocol_digest', 'state',
                  'generation_digest', 'record_digests', 'counts', 'resources', 'resource_status', 'termination_code', 'worker_reaped'})
        _require(started is not None and terminal['generation_digest'] == started['content_digest'], 'Terminal generation binding is missing or inconsistent.')
        _require(terminal['pilot_id'] == pilot['pilot_id'] and terminal['pilot_digest'] == pilot['content_digest'] and
                 terminal['protocol_digest'] == protocol['content_digest'], 'Terminal manifest binding disagrees with pilot/protocol.')
        _require(isinstance(terminal['state'], str) and terminal['state'] in TERMINAL_STATES and terminal['worker_reaped'] is True,
                 'Terminal state requires a reaped worker attestation.')
        _require(terminal['record_digests'] == references, 'Terminal manifest does not bind the exact attempt records.')
        _require(_digest(terminal['counts']) == _digest(_counts(attempts, candidates)), 'Terminal counts disagree with attempt evidence.')
        _require(terminal['resource_status'] == _resource_status(terminal['resources'], protocol), 'Terminal resource status is inconsistent.')
        _termination_code(terminal['state'], terminal['resources'], protocol, code=terminal['termination_code'])
        if terminal['state'] == 'complete':
            _require(all(a['state'] in ('generated', 'skipped_length') for a in attempts), 'Complete pilot has unfinished or failed attempts.')
        if terminal['state'] == 'stopped_budget':
            _require(terminal['resource_status'] == 'fail', 'Budget stop requires recorded resource threshold evidence.')
    return {'pilot': pilot, 'protocol': protocol, 'attempts': attempts, 'candidates': candidates,
            'terminal': terminal, 'generation_state': terminal['state'] if terminal else 'incomplete'}


def _counts(attempts, candidates):
    result = {state: sum(a['state'] == state for a in attempts) for state in
              ('generated', 'skipped_length', 'failed', 'interrupted', 'unattempted', 'started')}
    result.update(selected=len(attempts), auto_ineligible=sum(bool(c['exclusions']) for c in candidates.values()))
    return result


class PilotWriter:
    """Only the generation lock owner publishes attempts. Not a model worker.

    T3 may pass lock_fd to its child solely to keep the lifetime lock held until
    that child exits. The supervisor still owns all durable publication.
    """
    def __init__(self, fd, lock_fd, pilot, protocol, recovery_only):
        self.fd, self.lock_fd, self.pilot, self.protocol = fd, lock_fd, pilot, protocol
        self.active = True
        self.recovery_only = recovery_only

    def _current(self, parent=None):
        _require(self.active, 'Generation writer context is closed.')
        result = _load(self.fd, self.pilot, self.protocol)
        _require(result['terminal'] is None, 'Pilot is already terminal; preserve it and create a new pilot for new work.')
        if parent is not None:
            _require(not self.recovery_only, 'Abandoned pilots cannot resume. Finalize interrupted evidence without new model calls.')
            _require(not any(a['state'] == 'failed' for a in result['attempts']), 'A failed attempt stops this pilot; publish failed terminal evidence.')
            remaining = [a for a in result['attempts'] if a['result'] is None]
            _require(bool(remaining) and remaining[0]['parent_id'] == parent, 'Use the next frozen training parent; no retries or reselection.')
        return result

    def start(self, parent, input_tokens):
        current = self._current(parent)
        attempt = next(a for a in current['attempts'] if a['parent_id'] == parent)
        _require(attempt['intent'] is None and _integer(input_tokens, 1, 128), 'Attempt already started or exceeds the intact input token bound.')
        index = self.pilot['parent_ids'].index(parent)
        _publish(self.fd, f'attempt-{index:02d}.started.json', _document('synthetic-attempt-start',
                 pilot_id=self.pilot['pilot_id'], parent_id=parent, index=index, input_tokens=input_tokens))

    def _result(self, parent, state, **fields):
        current = self._current(parent)
        index = self.pilot['parent_ids'].index(parent)
        intent = current['attempts'][index]['intent']
        value = _document('synthetic-attempt-result', pilot_id=self.pilot['pilot_id'], parent_id=parent,
                          index=index, state=state, intent_digest=intent['content_digest'] if intent else None, **fields)
        _publish(self.fd, f'attempt-{index:02d}.result.json', value)

    def skip(self, parent, input_tokens):
        current = self._current(parent)
        index = self.pilot['parent_ids'].index(parent)
        _require(current['attempts'][index]['intent'] is None and _integer(input_tokens, 129, 100000), 'Only intact overlength input may skip without a model call.')
        self._result(parent, 'skipped_length', input_tokens=input_tokens)

    def generated(self, parent, text, output_tokens, eos):
        current = self._current(parent)
        index = self.pilot['parent_ids'].index(parent)
        _require(current['attempts'][index]['intent'] is not None, 'Persist a started intent before receiving generated output.')
        _require(isinstance(text, str) and len(text) <= 20000 and _integer(output_tokens, 0, 128) and
                 type(eos) is bool and (output_tokens > 0 or not eos), 'Invalid generated text/token/EOS contract.')
        owners = {c['fingerprint'] for c in current['candidates'].values() if not c['exclusions']}
        rows, originals, finals = _exclusion_index(self.protocol)
        text_digest = hashlib.sha256(text.encode('utf-8')).hexdigest()
        self._result(parent, 'generated', text=text, text_digest=text_digest,
                     candidate_id=_digest([self.pilot['pilot_id'], parent, text_digest]),
                     fingerprint=study.text_fingerprint(text), output_tokens=output_tokens, eos=eos,
                     exclusions=_exclusions(text, eos, parent, rows, originals, finals, owners))

    def failed(self, parent, error_code):
        current = self._current(parent)
        index = self.pilot['parent_ids'].index(parent)
        _require(current['attempts'][index]['intent'] is not None and isinstance(error_code, str) and
                 error_code in ERROR_CODES, 'A failed result needs an intent and a supported sanitized error code.')
        self._result(parent, 'failed', error_code=error_code)

    def finish(self, state, resources, *, worker_reaped, error_code=None):
        current = self._current()
        _require(isinstance(state, str) and state in TERMINAL_STATES and worker_reaped is True,
                 'Finish requires a supported terminal state after worker exit/reaping.')
        _require(not self.recovery_only or state == 'interrupted', 'Abandoned pilots can only finalize interrupted evidence.')
        status = _resource_status(resources, self.protocol)
        code = _termination_code(state, resources, self.protocol, code=error_code, recovery_only=self.recovery_only)
        if state == 'complete':
            _require(all(a['state'] in ('generated', 'skipped_length') for a in current['attempts']), 'Cannot complete a pilot with failed or unfinished attempts.')
        if state == 'stopped_budget':
            _require(status == 'fail', 'Budget stop requires observed threshold evidence.')
        attempts, candidates, refs = _attempts(self.fd, self.pilot, self.protocol, True)
        value = _document('synthetic-pilot-terminal', pilot_id=self.pilot['pilot_id'], pilot_digest=self.pilot['content_digest'],
                          protocol_digest=self.protocol['content_digest'],
                          generation_digest=_read(self.fd, 'generation.started.json')['content_digest'], state=state, record_digests=refs,
                          counts=_counts(attempts, candidates), resources=dict(resources), resource_status=status,
                          termination_code=code, worker_reaped=True)
        _publish(self.fd, 'terminal.json', value)


@contextmanager
def generation(root, pilot_id, *, studies_root=study.STUDIES_ROOT):
    with _directory(root, pilot_id) as fd, _lock(fd, 'generation.lock') as lock_fd:
        pilot, protocol = _context(fd, pilot_id, studies_root)
        _load(fd, pilot, protocol)
        recovery_only = _exists(fd, 'generation.started.json')
        if not recovery_only:
            _publish(fd, 'generation.started.json', _document('synthetic-generation-start',
                     pilot_id=pilot_id, pilot_digest=pilot['content_digest']))
        writer = PilotWriter(fd, lock_fd, pilot, protocol, recovery_only)
        try:
            yield writer
        finally:
            writer.active = False


def read_pilot(root, pilot_id, *, studies_root=study.STUDIES_ROOT):
    """Read coherent idle/terminal evidence; active attempts are not a snapshot.

    While generation owns its lock, expose only immutable selection/provenance
    and the running status. Counters/candidates become available after release
    or terminal publication, rather than validating racing multi-file writes.
    """
    with _directory(root, pilot_id) as fd:
        pilot, protocol = _context(fd, pilot_id, studies_root)
        try:
            with _lock(fd, 'generation.lock', shared=True, conflict=ReviewConflict):
                return _load(fd, pilot, protocol)
        except ReviewConflict:
            if _exists(fd, 'terminal.json'):
                return _load(fd, pilot, protocol)  # No publication follows an immutable terminal.
            return {'pilot': pilot, 'protocol': protocol, 'attempts': None, 'candidates': {},
                    'terminal': None, 'generation_state': 'running'}


def _payload(pilot, candidate, *, verdict, reason, rubric_ack, expected_revision):
    _require(isinstance(verdict, str) and verdict in ('accept', 'reject', 'ambiguous'), 'Choose accept, reject or ambiguous.')
    _require(isinstance(reason, str) and bool(reason.strip()) and len(reason) <= 2000,
             'A review reason of 1–2,000 characters is required.')
    _require(rubric_ack is True, 'Explicitly acknowledge the meaning/polarity/facts/English/rewording rubric.')
    _require(_integer(expected_revision, 0, MAX_REVISIONS), 'Use the integer review revision displayed for this pilot.')
    return {'pilot_id': pilot['pilot_id'], 'candidate_id': candidate['candidate_id'],
            'parent_id': candidate['parent_id'], 'candidate_digest': candidate['content_digest'],
            'verdict': verdict, 'reason': reason, 'rubric_ack': True, 'expected_revision': expected_revision}


def _exists(fd, name):
    try:
        os.stat(name, dir_fd=fd, follow_symlinks=False)
        return True
    except FileNotFoundError:
        return False


def _reviews(fd, current):
    terminal = current['terminal']
    _require(terminal is not None, 'Human reviews require a terminal pilot. Refresh after generation or interruption recovery.')
    pilot = current['pilot']
    metadata, gap = [], False
    for revision in range(1, MAX_REVISIONS + 2):
        name = f'review-{revision:04d}'
        meta = _read(fd, name + '.meta.json', optional=True, limit=4096)
        if meta is None:
            _require(not _exists(fd, name + '.json'), 'Review publication is incomplete. Preserve evidence; no review or handoff may continue until explicitly recovered.')
            gap = True
            continue
        _require(not gap and revision <= MAX_REVISIONS, 'Review metadata has a gap or exceeds the 200-revision bound.')
        try:
            info = os.stat(name + '.json', dir_fd=fd, follow_symlinks=False)
        except FileNotFoundError as exc:
            raise ValueError('A committed historical review snapshot is missing; preserve the remaining evidence.') from exc
        _require(stat.S_ISREG(info.st_mode) and info.st_size <= MAX_BYTES,
                 'Historical review snapshot has an unsupported file type/size.')
        _envelope(meta, 'synthetic-review-commit', {'pilot_id', 'terminal_digest', 'revision', 'previous_meta_digest',
                                                   'snapshot_digest', 'decision_id', 'payload_digest'})
        _id(meta['decision_id'])
        _require(meta['pilot_id'] == pilot['pilot_id'] and meta['terminal_digest'] == terminal['content_digest'] and
                 type(meta['revision']) is int and meta['revision'] == revision and _hash(meta['snapshot_digest']) and
                 _hash(meta['payload_digest']) and meta['previous_meta_digest'] == (metadata[-1]['content_digest'] if metadata else None),
                 'Review metadata chain does not bind this pilot/terminal/revision.')
        metadata.append(meta)
    if not metadata:
        return {'revision': 0, 'content_digest': None, 'decisions': {}, 'requests': {}}
    latest = _read(fd, f'review-{len(metadata):04d}.json')
    _envelope(latest, 'synthetic-review-snapshot', {'pilot_id', 'terminal_digest', 'revision', 'previous_digest', 'decisions', 'requests'})
    _require(latest['pilot_id'] == pilot['pilot_id'] and latest['terminal_digest'] == terminal['content_digest'] and
             type(latest['revision']) is int and latest['revision'] == len(metadata) and
             latest['previous_digest'] == (metadata[-2]['snapshot_digest'] if len(metadata) > 1 else None) and
             latest['content_digest'] == metadata[-1]['snapshot_digest'], 'Latest review snapshot disagrees with its commit chain.')
    _require(isinstance(latest['requests'], dict) and len(latest['requests']) == len(metadata), 'Review request history is incomplete.')
    decisions = {}
    seen = set()
    for meta in metadata:
        request_id = meta['decision_id']
        _require(request_id not in seen and request_id in latest['requests'], 'Review decision ID was reused or omitted.')
        seen.add(request_id)
        request = latest['requests'][request_id]
        _keys(request, {'payload', 'payload_digest', 'revision'})
        payload = request['payload']
        _keys(payload, {'pilot_id', 'candidate_id', 'parent_id', 'candidate_digest', 'verdict', 'reason', 'rubric_ack', 'expected_revision'})
        _require(_hash(payload['candidate_id']) and payload['candidate_id'] in current['candidates'], 'Review refers to a missing candidate.')
        candidate = current['candidates'][payload['candidate_id']]
        expected = _payload(pilot, candidate, **{k: payload[k] for k in ('verdict', 'reason', 'rubric_ack', 'expected_revision')})
        _require(_digest(expected) == _digest(payload) and payload['expected_revision'] == meta['revision'] - 1 and
                 type(request['revision']) is int and request['revision'] == meta['revision'] and
                 request['payload_digest'] == _digest(payload) == meta['payload_digest'], 'Review request payload/identity/revision is inconsistent.')
        decisions[payload['candidate_id']] = {k: payload[k] for k in ('parent_id', 'candidate_digest', 'verdict', 'reason', 'rubric_ack')}
        decisions[payload['candidate_id']]['revision'] = request['revision']
    _require(_digest(decisions) == _digest(latest['decisions']), 'Review decisions do not match the committed request history.')
    return latest


def read_reviews(root, pilot_id, *, studies_root=study.STUDIES_ROOT):
    """Read bounded commit metadata and only the latest full review snapshot."""
    with _directory(root, pilot_id) as fd, _lock(fd, 'review.lock', shared=True, conflict=ReviewConflict):
        pilot, protocol = _context(fd, pilot_id, studies_root)
        _require(_exists(fd, 'terminal.json'), 'Human reviews require a terminal pilot; no racing generation evidence is read.')
        return _reviews(fd, _load(fd, pilot, protocol))


def read_review_revision(root, pilot_id, revision, *, studies_root=study.STUDIES_ROOT):
    """Verify one historical snapshot against the current committed audit chain."""
    _require(_integer(revision, 1, MAX_REVISIONS), 'Choose a committed integer review revision 1–200.')
    with _directory(root, pilot_id) as fd, _lock(fd, 'review.lock', shared=True, conflict=ReviewConflict):
        pilot, protocol = _context(fd, pilot_id, studies_root)
        _require(_exists(fd, 'terminal.json'), 'Human review history requires a terminal pilot.')
        current = _load(fd, pilot, protocol)
        latest = _reviews(fd, current)
        return _review_snapshot(fd, current, latest, revision)


def _review_snapshot(fd, current, latest, revision):
    _require(revision <= latest['revision'], 'That review revision has not been committed.')
    snapshot = _read(fd, f'review-{revision:04d}.json')
    meta = _read(fd, f'review-{revision:04d}.meta.json', limit=4096)
    previous_meta = _read(fd, f'review-{revision-1:04d}.meta.json', limit=4096) if revision > 1 else None
    _envelope(snapshot, 'synthetic-review-snapshot', {'pilot_id', 'terminal_digest', 'revision', 'previous_digest', 'decisions', 'requests'})
    expected_requests = {key: value for key, value in latest['requests'].items() if value['revision'] <= revision}
    decisions = {}
    for request in sorted(expected_requests.values(), key=lambda r: r['revision']):
        payload = request['payload']
        decisions[payload['candidate_id']] = {k: payload[k] for k in ('parent_id', 'candidate_digest', 'verdict', 'reason', 'rubric_ack')}
        decisions[payload['candidate_id']]['revision'] = request['revision']
    _require(snapshot['pilot_id'] == current['pilot']['pilot_id'] and snapshot['terminal_digest'] == current['terminal']['content_digest'] and
             type(snapshot['revision']) is int and snapshot['revision'] == revision and
             snapshot['content_digest'] == meta['snapshot_digest'] and
             snapshot['previous_digest'] == (previous_meta['snapshot_digest'] if previous_meta else None) and
             _digest(snapshot['requests']) == _digest(expected_requests) and
             _digest(snapshot['decisions']) == _digest(decisions), 'Historical review snapshot disagrees with the committed audit chain.')
    return snapshot


def save_review(root, pilot_id, frame, candidate_id, *, verdict, reason, rubric_ack,
                expected_revision, decision_id, studies_root=study.STUDIES_ROOT):
    """Commit a complete reviewed snapshot atomically under revision CAS.

    Identical request retries return their original committed revision, even
    after newer reviews. Changed payload under the same ID is always a conflict.
    Candidate automatic exclusions remain immutable and cannot be overridden.
    """
    _id(decision_id)
    _require(_hash(candidate_id), 'Use a valid generated candidate ID.')
    with _directory(root, pilot_id) as fd, _lock(fd, 'review.lock', conflict=ReviewConflict):
        pilot, protocol = _context(fd, pilot_id, studies_root)
        study.attach_development(protocol, frame)
        _require(_exists(fd, 'terminal.json'), 'Human reviews require a terminal pilot; no decision saved.')
        current = _load(fd, pilot, protocol)
        _require(candidate_id in current['candidates'], 'Candidate not found in this pilot.')
        payload = _payload(pilot, current['candidates'][candidate_id], verdict=verdict, reason=reason,
                           rubric_ack=rubric_ack, expected_revision=expected_revision)
        previous = _reviews(fd, current)
        digest = _digest(payload)
        if decision_id in previous['requests']:
            committed = previous['requests'][decision_id]
            if committed['payload_digest'] != digest:
                raise ReviewConflict('Decision ID already belongs to a different payload. No change saved; reload the current review.')
            os.fsync(fd)  # A lost response may have followed a directory-sync failure.
            return committed['revision']
        if expected_revision != previous['revision']:
            raise ReviewConflict(f"Stale review: expected revision {expected_revision}, current {previous['revision']}. No change saved; reload before resubmitting.")
        _require(previous['revision'] < MAX_REVISIONS, 'This pilot reached its 200-review revision limit. Preserve history; no pruning or automatic overwrite.')
        revision = previous['revision'] + 1
        requests = dict(previous['requests'])
        requests[decision_id] = {'payload': payload, 'payload_digest': digest, 'revision': revision}
        decisions = dict(previous['decisions'])
        decisions[candidate_id] = {k: payload[k] for k in ('parent_id', 'candidate_digest', 'verdict', 'reason', 'rubric_ack')}
        decisions[candidate_id]['revision'] = revision
        value = _document('synthetic-review-snapshot', pilot_id=pilot_id,
                          terminal_digest=current['terminal']['content_digest'], revision=revision,
                          previous_digest=previous['content_digest'], decisions=decisions, requests=requests)
        previous_meta = _read(fd, f'review-{revision-1:04d}.meta.json') if revision > 1 else None
        meta = _document('synthetic-review-commit', pilot_id=pilot_id,
                         terminal_digest=current['terminal']['content_digest'], revision=revision,
                         previous_meta_digest=previous_meta['content_digest'] if previous_meta else None,
                         snapshot_digest=value['content_digest'], decision_id=decision_id, payload_digest=digest)
        _publish(fd, f'review-{revision:04d}.json', value)
        _publish(fd, f'review-{revision:04d}.meta.json', meta, limit=4096)
        return revision


def discover_local(studies_root, root):
    """Bounded metadata-only listing; invalid entries remain visible, never followed.

    Scan at most 50 immediate entries per root. Re-read the selected record at
    use time; this inventory is neither a coherent snapshot nor newest-first.
    """
    def scan(folder, reader, identity):
        entries, truncated = [], False
        try:
            fd = os.open(folder, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        except FileNotFoundError:
            return entries, truncated
        try:
            with os.scandir(fd) as items:
                for index, item in enumerate(items):
                    if index == 50:
                        truncated = True
                        break
                    entry = {identity: item.name, 'status': 'Unreadable', 'detail': 'Preserve files; use a compatible reader and check permissions.'}
                    try:
                        _id(item.name)
                        _require(item.is_dir(follow_symlinks=False), 'Not a regular evidence directory.')
                        value = reader(item.name)
                        entry.update(value)
                    except (OSError, ValueError, TypeError, KeyError):
                        pass
                    entries.append(entry)
        finally:
            os.close(fd)
        return entries, truncated

    def protocol(pid):
        record = study.read_protocol(studies_root, pid)
        return {'status': 'Ready', 'detail': '', 'created_at': record['created_at']}

    def pilot(pid):
        value = read_pilot(root, pid, studies_root=studies_root)
        return {'status': value['generation_state'], 'detail': '', 'protocol_id': value['pilot']['protocol_id']}

    protocols, limited_protocols = scan(studies_root, protocol, 'protocol_id')
    pilots, limited_pilots = scan(root, pilot, 'pilot_id')
    return protocols, pilots, limited_protocols or limited_pilots


# Seals bind evidence; they do not certify reviewer accuracy or classifier gains.
MAX_SEALS = 200
MAX_PILOT_ENTRIES = 2048


def _accepted(current, reviews):
    return [candidate for cid, candidate in current['candidates'].items()
            if reviews['decisions'].get(cid, {}).get('verdict') == 'accept' and not candidate['exclusions']]


def _qualification(current, reviews):
    accepted = _accepted(current, reviews) if reviews is not None else []
    counts = None
    if reviews is not None:
        decisions = reviews['decisions']
        counts = dict(revision=reviews['revision'], reviewed=len(decisions),
                      pending=len(current['candidates'])-len(decisions),
                      **{v: sum(d['verdict'] == v for d in decisions.values()) for v in ('accept', 'reject', 'ambiguous')},
                      accepted_eligible=len(accepted), negative_eligible=sum(c['label'] == 0 for c in accepted),
                      positive_eligible=sum(c['label'] == 1 for c in accepted))
    policy = current['protocol']['policy']['feasibility']
    terminal = current['terminal']
    gates = {
        'generation_complete': terminal is not None and terminal['state'] == 'complete',
        'resources_pass': terminal is not None and terminal['resource_status'] == 'pass',
        'execution_provenance': 'execution' in current['pilot'],
        'all_generated_reviewed': reviews is not None and len(reviews['decisions']) == len(current['candidates']),
        'accepted_min': len(accepted) >= policy['accepted_min'],
        'accepted_per_class_min': all(sum(c['label'] == label for c in accepted) >= policy['accepted_per_class_min'] for label in (0, 1)),
    }
    return {'qualified': all(gates.values()), 'unmet_gates': [key for key, passed in gates.items() if not passed], 'reviews': counts}


def _seal_binding(current, reviews):
    qualification = _qualification(current, reviews)
    _require(qualification['qualified'], 'Pilot does not qualify for a reviewed-set seal: ' + ', '.join(qualification['unmet_gates']))
    pilot, protocol = current['pilot'], current['protocol']
    accepted = [{key: c[key] for key in ('candidate_id', 'parent_id', 'content_digest', 'label')} for c in _accepted(current, reviews)]
    # Keep a named candidate_digest at the handoff boundary.
    for row in accepted:
        row['candidate_digest'] = row.pop('content_digest')
    return dict(pilot_id=pilot['pilot_id'], protocol_id=protocol['protocol_id'],
                protocol_digest=protocol['content_digest'], pilot_digest=pilot['content_digest'],
                terminal_digest=current['terminal']['content_digest'], review_revision=reviews['revision'],
                review_digest=reviews['content_digest'], accepted=accepted,
                counts={key: qualification['reviews'][key] for key in
                        ('reviewed', 'accepted_eligible', 'negative_eligible', 'positive_eligible')})


def _historical_decisions(latest, revision):
    decisions = {}
    for request in sorted(latest['requests'].values(), key=lambda r: r['revision']):
        if request['revision'] > revision:
            break
        payload = request['payload']
        decisions[payload['candidate_id']] = {k: payload[k] for k in ('parent_id', 'candidate_digest', 'verdict', 'reason', 'rubric_ack')}
        decisions[payload['candidate_id']]['revision'] = request['revision']
    return decisions


def _sealed_records(fd, current, latest):
    """Validate bounded seals against committed metadata, not all raw snapshots."""
    records = []
    with os.scandir(fd) as entries:
        for index, entry in enumerate(entries):
            _require(index < MAX_PILOT_ENTRIES, 'Pilot directory exceeds its 2,048-entry safety bound; preserve evidence.')
            if not entry.name.startswith('seal-') or not entry.name.endswith('.json'):
                continue
            _require(re.fullmatch(r'seal-[0-9a-f]{32}\.json', entry.name) is not None, 'Unsupported seal filename; preserve evidence.')
            _require(len(records) < MAX_SEALS, 'Pilot exceeds its 200-seal allowance; preserve evidence.')
            record = _read(fd, entry.name)
            fields = {'seal_id', 'pilot_id', 'protocol_id', 'protocol_digest', 'pilot_digest', 'terminal_digest',
                      'review_revision', 'review_digest', 'accepted', 'counts'}
            _envelope(record, 'synthetic-reviewed-seal', fields)
            _id(record['seal_id'])
            _require(entry.name == f"seal-{record['seal_id']}.json", 'Seal identity disagrees with its filename.')
            revision = record['review_revision']
            _require(_integer(revision, 1, latest['revision']), 'Seal review revision is not committed.')
            meta = _read(fd, f'review-{revision:04d}.meta.json', limit=4096)
            snapshot = latest if revision == latest['revision'] else {
                'revision': revision, 'content_digest': meta['snapshot_digest'],
                'decisions': _historical_decisions(latest, revision)}
            expected = _seal_binding(current, snapshot)
            _require(_digest({k: record[k] for k in expected}) == _digest(expected), 'Seal disagrees with its exact qualified evidence.')
            records.append(record)
    return sorted(records, key=lambda r: (r['created_at'], r['seal_id']))


def _terminal_context(fd, pilot_id, studies_root):
    pilot, protocol = _context(fd, pilot_id, studies_root)
    _require(_exists(fd, 'terminal.json'), 'Sealing and handoff require a terminal pilot; refresh after generation.')
    current = _load(fd, pilot, protocol)
    return current, _reviews(fd, current)


def seal_pilot(root, pilot_id, frame, *, expected_revision, seal_id=None, studies_root=study.STUDIES_ROOT):
    """Publish one immutable qualified seal under the same CAS lock as reviews.

    An unchanged, current request may retry the same seal ID after a lost fsync
    acknowledgement. Review changes require a new identity; no old seal is edited.
    """
    seal_id = uuid.uuid4().hex if seal_id is None else _id(seal_id)
    _require(_integer(expected_revision, 1, MAX_REVISIONS), 'Use a committed integer review revision 1–200 for sealing.')
    with _directory(root, pilot_id) as fd, _lock(fd, 'review.lock', conflict=ReviewConflict):
        current, reviews = _terminal_context(fd, pilot_id, studies_root)
        study.attach_development(current['protocol'], frame)
        records = _sealed_records(fd, current, reviews)
        if expected_revision != reviews['revision']:
            raise ReviewConflict('Review changed before sealing. Refresh and inspect the current revision; no seal saved.')
        binding = _seal_binding(current, reviews)
        existing = next((r for r in records if r['seal_id'] == seal_id), None)
        if existing is not None:
            if _digest({k: existing[k] for k in binding}) != _digest(binding):
                raise ReviewConflict('Seal identity already binds different evidence. Use a new seal identity after review changes.')
            os.fsync(fd)
            return existing
        _require(len(records) < MAX_SEALS, 'This pilot reached its 200-seal allowance. Preserve history; no overwrite or pruning.')
        record = _document('synthetic-reviewed-seal', seal_id=seal_id, **binding)
        _publish(fd, f'seal-{seal_id}.json', record)
        return record


def read_seal(root, pilot_id, seal_id, *, studies_root=study.STUDIES_ROOT):
    """Read historical evidence explicitly; historical does not mean trainable."""
    _id(seal_id)
    with _directory(root, pilot_id) as fd, _lock(fd, 'review.lock', shared=True, conflict=ReviewConflict):
        current, reviews = _terminal_context(fd, pilot_id, studies_root)
        record = next((r for r in _sealed_records(fd, current, reviews) if r['seal_id'] == seal_id), None)
        _require(record is not None, 'Seal not found in this pilot.')
        _review_snapshot(fd, current, reviews, record['review_revision'])
        return {**record, 'current': record['review_revision'] == reviews['revision'] and record['review_digest'] == reviews['content_digest']}


def read_handoff(root, pilot_id, *, studies_root=study.STUDIES_ROOT):
    """Atomic text-free qualification/currency summary, without model assets.

    Read generation separately before taking the review lock. Active generation
    exposes no racing attempt counts; terminal evidence is immutable thereafter.
    """
    current = read_pilot(root, pilot_id, studies_root=studies_root)
    reviews, records = None, []
    if current['terminal'] is not None:
        with _directory(root, pilot_id) as fd, _lock(fd, 'review.lock', shared=True, conflict=ReviewConflict):
            current, reviews = _terminal_context(fd, pilot_id, studies_root)
            records = _sealed_records(fd, current, reviews)
    terminal = current['terminal']
    return dict(schema_version=1, pilot_id=pilot_id, protocol_id=current['pilot']['protocol_id'],
                generation_state=current['generation_state'], counts=terminal['counts'] if terminal else None,
                resource_status=terminal['resource_status'] if terminal else 'unknown',
                termination_code=terminal['termination_code'] if terminal else None,
                resources=terminal['resources'] if terminal else None,
                review_digest=reviews['content_digest'] if reviews else None,
                **_qualification(current, reviews),
                seals=[dict(seal_id=r['seal_id'], created_at=r['created_at'], review_revision=r['review_revision'],
                            current=r['review_revision'] == reviews['revision'] and r['review_digest'] == reviews['content_digest']) for r in records],
                no_training_has_run=True)


@dataclass(frozen=True)
class TrainingBundle:
    """Independent fixed partitions and one augmented train frame, never re-split."""
    splits: Splits
    training: pd.DataFrame
    lineage: pd.DataFrame
    protocol_id: str
    pilot_id: str
    seal_id: str
    review_revision: int
    seal_digest: str


def build_training_bundle(root, pilot_id, seal_id, original_frame, *, studies_root=study.STUDIES_ROOT):
    """Re-read authoritative current evidence under lock; supplied objects cannot bypass currency."""
    _id(seal_id)
    with _directory(root, pilot_id) as fd, _lock(fd, 'review.lock', shared=True, conflict=ReviewConflict):
        current, reviews = _terminal_context(fd, pilot_id, studies_root)
        splits = study.attach_development(current['protocol'], original_frame)
        record = next((r for r in _sealed_records(fd, current, reviews) if r['seal_id'] == seal_id), None)
        _require(record is not None, 'Seal not found in this pilot.')
        _require(record['review_revision'] == reviews['revision'] and record['review_digest'] == reviews['content_digest'],
                 'This seal is historical; current handoff requires a new seal of the current review revision.')
        _seal_binding(current, reviews)  # Recheck gates and automatic findings, never trust a cached summary.
        accepted = _accepted(current, reviews)
        training = pd.concat([splits.train.copy(deep=True), pd.DataFrame(
            [{'text': c['text'], 'label': c['label']} for c in accepted])], ignore_index=True)
        lineage = [{'training_index': i, 'source_kind': 'original', 'parent_id': review_id(r.text, r.label), 'candidate_id': None}
                   for i, r in enumerate(splits.train.itertuples())]
        lineage += [{'training_index': len(splits.train)+i, 'source_kind': 'synthetic', 'parent_id': c['parent_id'], 'candidate_id': c['candidate_id']}
                    for i, c in enumerate(accepted)]
        return TrainingBundle(splits=splits, training=training, lineage=pd.DataFrame(lineage),
                              protocol_id=current['protocol']['protocol_id'], pilot_id=pilot_id, seal_id=seal_id,
                              review_revision=reviews['revision'], seal_digest=record['content_digest'])
