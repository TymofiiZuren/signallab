"""Native review workspace. Models run only through the supervised CLI."""
from pathlib import Path
import uuid
import shlex

import pandas as pd
import streamlit as st

from signallab import study, synthetic
from signallab.core import review_id
from signallab.demo import demo_reviews
from signallab.synthetic import discover_local

ROOT = Path(__file__).resolve().parents[1]
VERDICTS = {'Accept': 'accept', 'Reject': 'reject', 'Ambiguous': 'ambiguous'}
FINDINGS = {
    'blank': 'Blank output', 'unchanged_parent': 'Unchanged original',
    'original_collision': 'Matches a development review', 'final_collision': 'Matches reserved final text',
    'candidate_collision': 'Duplicate of an earlier candidate', 'incomplete_output': 'Incomplete output (no end token)',
}


def _clear_editor():
    st.session_state.pop('synthetic_editor', None)
    st.session_state.pop('synthetic_notice', None)
    st.session_state.pop('synthetic_conflict_draft', None)


def _command_prefix(studies_root=None, root=None):
    command = 'uv run --locked python -m signallab.synthetic_cli'
    if studies_root is not None and Path(studies_root) != study.STUDIES_ROOT:
        command += ' --studies-root ' + shlex.quote(str(studies_root))
    if root is not None and Path(root) != synthetic.SYNTHETIC_ROOT:
        command += ' --pilots-root ' + shlex.quote(str(root))
    return command


def _commands(protocol_id=None, pilot_id=None, *, studies_root=None, root=None):
    command = _command_prefix(studies_root, root)
    st.write('Generation runs in your terminal. Freeze separate development and reserved final CSVs, then explicitly prepare the pinned model before generating locally.')
    st.code(command + ' --help', language='bash')
    if protocol_id:
        st.code(f'{command} prepare --protocol {protocol_id}\n'
                f'{command} generate --protocol {protocol_id} --csv development.csv', language='bash')
    else:
        st.code(command + ' freeze --help', language='bash')
    if pilot_id:
        st.code(f'{command} inspect --pilot {pilot_id}', language='bash')
    st.caption('See docs/synthetic-review.md for the ordered recipe. Preparation downloads public model assets; review does not download or run a model.')


def _attachment(protocol):
    st.header('Attach the original development reviews')
    st.write('Attach the exact development CSV to reveal training pairs and make decisions. Do not attach the reserved final CSV. Processing stays local; generated text and your review reasons persist in local pilot artifacts. Exclusion does not redact copied text.')
    choices = ['Not attached', 'Upload CSV', 'Authored demo']
    if (ROOT / 'data/reviews.csv').is_file():
        choices.append('Local review dataset')
    source = st.selectbox('Development source', choices, key='synthetic_attachment_source')
    if source == 'Not attached':
        _clear_editor()
        st.info('Unattached — summary is available; attach exact development reviews to inspect pairs.')
        return None, source
    try:
        with st.spinner('Checking exact original reviews…'):
            if source == 'Upload CSV':
                uploaded = st.file_uploader('Development CSV', type='csv', key='synthetic_upload', help='UTF-8 text,label CSV; maximum 10 MiB.')
                if uploaded is None:
                    _clear_editor()
                    st.info('Unattached — choose the original development CSV.')
                    return None, source
                frame = study.load_study_csv(uploaded)
            elif source == 'Authored demo':
                frame = demo_reviews()
            else:
                frame = study.load_study_csv(ROOT / 'data/reviews.csv')
            splits = study.attach_development(protocol, frame)
    except (OSError, ValueError):
        _clear_editor()
        st.error('Mismatch or unreadable source — no pair or decision form is available. Attach the exact original development CSV, within 10 MiB. Numeric evidence remains available.')
        return None, source
    st.success('Exact development source confirmed. Only frozen training parents are shown.')
    if source == 'Authored demo':
        st.warning('Authored demo reviews illustrate the workflow; they are not real customer evidence.')
    return (frame, splits.train), source


def _summary(current, reviews, handoff):
    terminal = current['terminal']
    st.header('Pilot status')
    st.write(f"Generation: {current['generation_state']} · selected parents: {len(current['pilot']['parent_ids'])}.")
    if terminal is None:
        st.write('Review completion: unavailable until generation reaches a terminal state.')
        st.write('Feasibility: not qualified. Seal currency: none; terminal generation and complete reviews are required.')
        st.info('Running pilots require an explicit refresh. Incomplete pilots require CLI recovery after the worker exits; generation is never resumed from this page.')
        return
    counts = terminal['counts']
    decisions = reviews['decisions']
    candidates = current['candidates']
    accepted = [c for cid, c in candidates.items() if decisions.get(cid, {}).get('verdict') == 'accept' and not c['exclusions']]
    st.write(f"Review completion: {len(decisions)} of {len(candidates)} generated candidates reviewed; {len(candidates)-len(decisions)} pending.")
    if handoff is None:
        st.write(f"Resource evidence: {terminal['resource_status']}. Feasibility and seal currency: unavailable until a coherent refresh.")
    else:
        feasibility = 'gates met for this reviewed pilot' if handoff['qualified'] else 'not qualified'
        currency = 'current' if any(s['current'] for s in handoff['seals']) else 'historical' if handoff['seals'] else 'none'
        st.write(f"Resource evidence: {terminal['resource_status']}. Feasibility: {feasibility}. Seal currency: {currency}.")
    rows = {'Generated': counts['generated'], 'Skipped overlength': counts['skipped_length'],
            'Failed': counts['failed'], 'Interrupted': counts['interrupted'], 'Unattempted': counts['unattempted'],
            'Started': counts['started'], 'Automatically excluded': counts['auto_ineligible'],
            **{v.title(): sum(d['verdict'] == v for d in decisions.values()) for v in VERDICTS.values()},
            'Accepted, automatically eligible': len(accepted),
            'Eligible negative': sum(c['label'] == 0 for c in accepted), 'Eligible positive': sum(c['label'] == 1 for c in accepted)}
    with st.expander('Counts and recorded resources'):
        st.dataframe(pd.DataFrame([{'Measure': k, 'Count': v} for k, v in rows.items()]), hide_index=True, width='stretch')
        st.json(terminal['resources'])
    st.caption(f"Review revision {reviews['revision']} · {synthetic.MAX_REVISIONS-reviews['revision']} revisions remaining. Ambiguous counts as reviewed and is excluded from training. Human acceptance cannot override an automatic exclusion.")


def _review(current, reviews, attachment, source, root, studies_root, *, review_busy=False, current_seal=False):
    frame, train = attachment
    candidates, decisions = current['candidates'], reviews['decisions']
    st.header('Review candidate')
    notice = st.session_state.get('synthetic_notice')
    st.write('Review every generated output, including automatic exclusions. Changing source, pilot, candidate or leaving this page discards unsaved form edits. Save explicitly before navigating.')
    if st.session_state.pop('synthetic_follow_pending', False):
        st.session_state.synthetic_filter = 'Pending'
    if st.session_state.pop('synthetic_reload_candidate', False):
        st.session_state.synthetic_filter = 'All'
    filter_name = st.selectbox('Review filter', ['Pending', 'All', 'Accepted', 'Rejected', 'Ambiguous', 'Auto-ineligible'], key='synthetic_filter')
    verdict_filter = {'Accepted': 'accept', 'Rejected': 'reject', 'Ambiguous': 'ambiguous'}
    choices = [cid for cid, c in candidates.items() if filter_name == 'All'
               or (filter_name == 'Pending' and cid not in decisions)
               or (filter_name == 'Auto-ineligible' and c['exclusions'])
               or decisions.get(cid, {}).get('verdict') == verdict_filter.get(filter_name, 'not-a-verdict')]
    editor = st.session_state.get('synthetic_editor')
    # Keep a conflicted candidate reachable even if another reviewer removed it
    # from Pending. An external save must never silently move a local draft.
    base_context = (str(root), str(studies_root), current['pilot']['pilot_id'], current['terminal']['content_digest'], source)
    if editor and editor['context'][:-1] == base_context and editor.get('filter') == filter_name and editor['context'][-1] not in choices:
        choices.insert(0, editor['context'][-1])
    if not choices:
        if notice and st.session_state.get('synthetic_next') and notice['pilot_id'] == current['pilot']['pilot_id']:
            st.success(f"Decision saved in revision {notice['revision']}. No pending candidates remain in this filter.")
        st.session_state.pop('synthetic_next', None)
        _clear_editor()
        st.info('No candidates in this filter. Choose All to revisit reviewed decisions.')
        return
    next_id = st.session_state.pop('synthetic_next', None)
    if next_id in choices:
        st.session_state.synthetic_candidate = next_id
    if st.session_state.get('synthetic_candidate') not in choices:
        st.session_state.synthetic_candidate = choices[0]
    cid = st.selectbox('Candidate', choices, key='synthetic_candidate',
                       format_func=lambda v: f"Parent {current['pilot']['parent_ids'].index(candidates[v]['parent_id'])+1} · {v[:8]}")
    context = (*base_context, cid)
    if not editor or editor['context'] != context:
        if editor:
            st.session_state.pop('synthetic_notice', None)
            notice = None
        editor = {'context': context, 'revision': reviews['revision'], 'decision_id': uuid.uuid4().hex, 'nonce': uuid.uuid4().hex, 'filter': filter_name}
        st.session_state.synthetic_editor = editor
    if notice and notice['pilot_id'] == current['pilot']['pilot_id']:
        st.success(f"Decision saved in revision {notice['revision']}. Next pending candidate selected when available.")
    retained = st.session_state.get('synthetic_conflict_draft')
    if retained and retained['context'] != context:
        st.session_state.pop('synthetic_conflict_draft', None)
    candidate = candidates[cid]
    parents = {review_id(r.text, r.label): r for r in train.itertuples()}
    parent = parents[candidate['parent_id']]
    left, right = st.columns(2)
    with left:
        st.subheader('Original training review')
        st.write(f"Recorded sentiment: {'Positive' if parent.label else 'Negative'} (read only)")
        st.text(parent.text, width='stretch')
    with right:
        st.subheader('Generated paraphrase')
        st.text(candidate['text'], width='stretch')
        st.caption(f"Output tokens: {candidate['output_tokens']} · End token: {'yes' if candidate['eos'] else 'no'}")
    old = decisions.get(cid)
    st.write(f"Human decision: {old['verdict'].title() if old else 'Pending'}.")
    findings = candidate['exclusions']
    st.write('Training eligibility: Excluded — ' + '; '.join(FINDINGS[f] for f in findings) if findings else
             'Training eligibility: automatic checks pass; requires human acceptance and a qualified seal before training.')
    if old:
        with st.expander('Current saved decision'):
            st.text(old['reason'], width='stretch')
            st.caption(f"Saved in revision {old['revision']}")
    draft = st.session_state.get('synthetic_conflict_draft')
    if draft and draft['context'] == context and not editor.get('conflict'):
        st.write('Earlier unsaved proposal — copy manually if still appropriate')
        st.text(draft['reason'], width='stretch')
    if editor.get('conflict'):
        st.error('Review changed or is busy. Your proposed decision is unsaved; reload the current decision before submitting again.')
        st.write('Unsaved proposal — retained only for this candidate')
        st.text(editor['draft']['reason'], width='stretch')
        st.caption(f"Proposed verdict: {editor['draft']['verdict']}; rubric acknowledged: {editor['draft']['rubric_ack']}")
    if editor.get('conflict') or editor.get('uncertain'):
        if st.button('Reload current decision'):
            if editor.get('draft'):
                st.session_state.synthetic_conflict_draft = {'context': context, **editor['draft']}
            st.session_state.pop('synthetic_editor', None)
            st.session_state.synthetic_reload_candidate = True
            st.rerun()
    if reviews['revision'] >= synthetic.MAX_REVISIONS:
        st.warning('The 200-revision allowance is exhausted. Preserve history; decisions cannot be overwritten or pruned.')
        return
    if current_seal:
        st.warning('Saving a new review revision makes the current seal historical. Training handoff will require a new qualified seal; history is preserved.')
    prefix = editor['nonce']
    with st.form(f'synthetic_decision_{prefix}'):
        verdict = st.radio('Human decision', list(VERDICTS), index=list(VERDICTS.values()).index(old['verdict']) if old else None, key=f'verdict_{prefix}')
        reason = st.text_area('Reason for this decision', value=old['reason'] if old else '', max_chars=2000, key=f'reason_{prefix}')
        ack = st.checkbox('I checked meaning, sentiment and negation, facts and numbers, English, and meaningful rewording.', key=f'ack_{prefix}')
        st.caption('Accept only if every check passes. Reject clear violations or instruction drift; choose Ambiguous when preservation is uncertain. No training has run.')
        submit = st.form_submit_button('Save decision and next pending', type='primary', disabled=bool(editor.get('conflict')) or review_busy)
    if submit:
        if review_busy:
            st.error('Review history is busy. Your form is retained; refresh before saving again. No decision saved.')
            return
        if verdict is None or not reason.strip() or not ack:
            st.error('Choose a verdict, give a reason, and acknowledge all rubric checks. No decision saved.')
            return
        payload = {'verdict': VERDICTS[verdict], 'reason': reason, 'rubric_ack': ack}
        if editor.get('uncertain') and payload != editor['draft']:
            st.error('The previous save could not be confirmed. Retry the unchanged proposal or reload current decision before editing it.')
            return
        editor['draft'] = payload
        try:
            with st.spinner('Saving decision…'):
                revision = synthetic.save_review(root, current['pilot']['pilot_id'], frame, cid, **payload,
                                                 expected_revision=editor['revision'], decision_id=editor['decision_id'], studies_root=studies_root)
        except synthetic.ReviewConflict:
            editor['conflict'] = True
            st.rerun()
        except (OSError, ValueError):
            editor['uncertain'] = True
            st.error('Save could not be confirmed. Preserve evidence; retry this unchanged decision or reload current decision. No success is assumed.')
            return
        st.session_state.pop('synthetic_conflict_draft', None)
        st.session_state.synthetic_notice = {'pilot_id': current['pilot']['pilot_id'], 'revision': revision}
        st.session_state.pop('synthetic_editor', None)
        remaining = [v for v in candidates if v != cid and v not in decisions]
        st.session_state.synthetic_follow_pending = bool(remaining)
        st.session_state.synthetic_next = remaining[0] if remaining else cid
        st.rerun()


def render_synthetic_lab(*, studies_root=None, root=None):
    studies_root = Path(studies_root) if studies_root is not None else study.STUDIES_ROOT
    root = Path(root) if root is not None else synthetic.SYNTHETIC_ROOT
    with st.container(key='page_intro'):
        st.title('Synthetic Review Lab')
        st.write('Review training-only paraphrases, inspect automatic exclusions, and record your reasoning.')
    st.write('No training has run in this workspace. A completed pilot does not demonstrate better sentiment predictions.')
    identity = (str(studies_root), str(root))
    if st.session_state.get('synthetic_roots') != identity:
        st.session_state.synthetic_roots = identity
        st.session_state.pop('synthetic_inventory', None)
        _clear_editor()
    if st.button('Refresh local studies'):
        try:
            with st.spinner('Reading local studies…'):
                st.session_state.synthetic_inventory = discover_local(studies_root, root)
        except (OSError, ValueError):
            st.session_state.pop('synthetic_inventory', None)
            _clear_editor()
            st.error('Cannot read local evidence roots. Check permissions and preserve existing files, then refresh.')
    inventory = st.session_state.get('synthetic_inventory')
    if inventory is None:
        st.info('Refresh to browse frozen protocols and local generation pilots. Source attachment is separate from the sidebar dataset.')
        _clear_editor()
        _commands(studies_root=studies_root, root=root)
        return
    protocols, pilots, truncated = inventory
    if truncated:
        st.warning('Reached the 50-entry scan limit per root. This bounded listing is not complete or globally newest; use the CLI to inspect a known pilot.')
    invalid = [dict(Kind='Protocol', Identity=p['protocol_id'], Status=p['status']) for p in protocols if p['status'] != 'Ready']
    invalid += [dict(Kind='Pilot', Identity=p['pilot_id'], Status=p['status']) for p in pilots if p['status'] == 'Unreadable']
    if invalid:
        with st.expander('Unreadable or incompatible local evidence'):
            st.dataframe(pd.DataFrame(invalid), hide_index=True, width='stretch')
            st.write('Preserve these files. Check permissions or use a compatible reader; incomplete publication is never repaired automatically.')
    ready = [p['protocol_id'] for p in protocols if p['status'] == 'Ready']
    if not ready:
        st.info('No readable frozen protocols found. Freeze exact development and reserved final inputs first.')
        _clear_editor()
        _commands(studies_root=studies_root, root=root)
        return
    pid = st.selectbox('Frozen protocol', ready, key='synthetic_protocol')
    try:
        protocol = study.read_protocol(studies_root, pid)
        st.write(protocol['policy']['question'])
        available = [p['pilot_id'] for p in pilots if p.get('protocol_id') == pid and p['status'] != 'Unreadable']
        if not available:
            _clear_editor()
            st.info('No readable pilot for this protocol. Prepare the pinned model and generate once through the CLI, then refresh.')
            _commands(pid, studies_root=studies_root, root=root)
            return
        pilot_id = st.selectbox('Generation pilot', available, key='synthetic_pilot',
                                format_func=lambda v: f"{v[:8]} · {next(p['status'] for p in pilots if p['pilot_id'] == v)}")
        current = synthetic.read_pilot(root, pilot_id, studies_root=studies_root)
        if current['pilot']['protocol_id'] != pid or current['protocol']['content_digest'] != protocol['content_digest']:
            raise ValueError('Selected protocol and pilot disagree.')
        cache_identity = (str(root), str(studies_root), pilot_id,
                          current['terminal']['content_digest'] if current['terminal'] else None)
        review_busy = False
        try:
            reviews = synthetic.read_reviews(root, pilot_id, studies_root=studies_root) if current['terminal'] else None
        except synthetic.ReviewConflict:
            cached = st.session_state.get('synthetic_review_cache')
            if not cached or cached['identity'] != cache_identity:
                st.warning('Review history is busy in another process. Refresh after it finishes; no decision can be saved.')
                return
            reviews = cached['reviews']
            review_busy = True
            st.warning(f"Review history is busy. Showing last verified revision {reviews['revision']} read only; your form is retained. Refresh before saving.")
        else:
            st.session_state.synthetic_review_cache = {'identity': cache_identity, 'reviews': reviews}
    except (OSError, ValueError, TypeError, KeyError):
        _clear_editor()
        st.error('Selected evidence changed or cannot be read. No pair or save is available. Preserve files, check compatibility and permissions, then refresh.')
        return
    st.caption(f"Protocol {pid[:8]} · pilot {pilot_id[:8]} · seed {protocol['seed']}")
    handoff = None
    if not review_busy:
        try:
            handoff = synthetic.read_handoff(root, pilot_id, studies_root=studies_root)
            if reviews is not None and (handoff['reviews']['revision'] != reviews['revision'] or handoff['review_digest'] != reviews['content_digest']):
                handoff = None
                st.warning('Review changed during refresh. Refresh again before sealing; no mixed snapshot can authorize a seal.')
        except synthetic.ReviewConflict:
            st.warning('Evidence is busy. Sealing is disabled; refresh after the other process finishes.')
        except (OSError, ValueError):
            _clear_editor()
            st.error('Seal evidence cannot be validated. Preserve files and check compatibility before continuing.')
            return
    _summary(current, reviews, handoff)
    with st.expander('Protocol, model and evidence provenance'):
        st.json({'protocol': protocol, 'pilot': current['pilot'], 'terminal': current['terminal']})
    if current['terminal']:
        attachment, source = _attachment(protocol)
        if attachment is not None:
            _review(current, reviews, attachment, source, root, studies_root, review_busy=review_busy,
                    current_seal=handoff is not None and any(s['current'] for s in handoff['seals']))
        _seal_panel(current, handoff, attachment, source, root, studies_root)
    with st.expander('Next steps and CLI inspection'):
        _commands(pid, pilot_id, studies_root=studies_root, root=root)
        st.write('The validated bundle is a read-only training handoff. Controlled multi-condition experiments follow later. No training has run here, and classifier improvement is not established. Low yield is a valid result; do not replenish selected parents to reach a threshold.')


def _seal_panel(current, handoff, attachment, source, root, studies_root):
    st.header('Seal the reviewed accepted set')
    st.write('A seal pins one qualified review revision and its eligible descendants. It does not train a classifier or prove that augmentation improves predictions.')
    if handoff is None:
        st.info('Refresh coherent evidence before sealing. Saving is unavailable while history is busy or changed.')
        return
    policy = current['protocol']['policy']['feasibility']
    labels = {'generation_complete': 'Generation must complete.', 'resources_pass': 'Observed resources must pass the guard.',
              'execution_provenance': 'The pilot must record the supervised preparation/generation provenance.',
              'all_generated_reviewed': 'Review every generated candidate, including automatic exclusions.',
              'accepted_min': f"Need at least {policy['accepted_min']} human-accepted, automatically eligible candidates.",
              'accepted_per_class_min': f"Need at least {policy['accepted_per_class_min']} eligible candidates in each sentiment class."}
    if handoff['unmet_gates']:
        st.write('Unmet qualification gates:')
        for gate in handoff['unmet_gates']:
            st.write('• ' + labels[gate])
    if handoff['seals']:
        st.dataframe(pd.DataFrame([{'Seal': s['seal_id'][:8], 'Review revision': s['review_revision'],
                                   'Currency': 'Current' if s['current'] else 'Historical'} for s in handoff['seals']]),
                     hide_index=True, width='stretch')
        with st.expander('Sealed handoff identities and inspection'):
            st.json(handoff['seals'])
            active = next((s for s in handoff['seals'] if s['current']), None)
            if active:
                st.code(_command_prefix(studies_root, root) + f" inspect-bundle --pilot {current['pilot']['pilot_id']} --seal {active['seal_id']} --csv '<original-development.csv>'", language='bash')
            st.write('No training has run. Historical seals remain readable; the training builder rejects them after review edits.')
    context = (str(root), str(studies_root), current['pilot']['pilot_id'], current['terminal']['content_digest'], source)
    if attachment is None:
        st.session_state.pop('synthetic_seal_request', None)
        st.session_state.pop('synthetic_seal_notice', None)
        st.info('Attach the exact development source above before sealing. Summary and historical identities remain readable.')
        return
    revision = handoff['reviews']['revision']
    request = st.session_state.get('synthetic_seal_request')
    if not request or request['context'] != context:
        request = {'context': context, 'revision': revision, 'seal_id': uuid.uuid4().hex, 'uncertain': False}
        st.session_state.synthetic_seal_request = request
        st.session_state.pop('synthetic_seal_notice', None)
    changed = request['revision'] != revision
    if changed:
        st.warning('Review changed since the seal controls were shown. Inspect the current evidence, then reload seal context before sealing.')
        if st.button('Reload seal context'):
            st.session_state.pop('synthetic_seal_request', None)
            st.session_state.pop('synthetic_seal_notice', None)
            st.rerun()
    notice = st.session_state.get('synthetic_seal_notice')
    if notice and notice['context'] == context and notice['revision'] == revision:
        st.success(f"Seal saved: {notice['seal_id'][:8]} · review revision {revision}. No training has run.")
    active = any(s['current'] for s in handoff['seals'])
    if active and not request['uncertain']:
        st.info('A current seal already exists for this review revision. Inspect its validated handoff above.')
    disabled = not handoff['qualified'] or changed or (active and not request['uncertain'])
    if st.button('Seal accepted set', disabled=disabled, type='primary'):
        try:
            record = synthetic.seal_pilot(root, current['pilot']['pilot_id'], attachment[0],
                       expected_revision=request['revision'], seal_id=request['seal_id'], studies_root=studies_root)
        except synthetic.ReviewConflict:
            st.error('Review changed or is busy. No seal success is assumed; refresh and reload seal context before retrying.')
            return
        except (OSError, ValueError):
            request['uncertain'] = True
            st.error('Seal publication could not be confirmed. Preserve evidence; retry the unchanged seal request. No training has run.')
            return
        request['uncertain'] = False
        st.session_state.synthetic_seal_notice = {'context': context, 'revision': revision, 'seal_id': record['seal_id']}
        st.rerun()
