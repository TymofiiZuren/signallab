"""Native Case Study workspace; numeric evidence never requires a CSV or model."""
import json
from pathlib import Path

import pandas as pd
import streamlit as st

from signallab import runs
from signallab.case_study import QUESTION, LIMITATIONS, build_case_study, case_study_examples, case_study_markdown
from signallab.core import load_csv
from signallab.demo import demo_reviews

ROOT = Path(__file__).resolve().parents[1]


def render_case_study(refresh, run_label, open_analysis):
    with st.container(key='page_intro'):
        st.title('Case study')
        st.write('Read what changed, inspect mistakes, and download the evidence summary.')
    st.write(QUESTION)
    if st.button('Refresh saved runs', key='case_refresh'):
        refresh()
    entries = st.session_state.get('saved_run_entries')
    if entries is None:
        st.info('Refresh to find saved comparisons. Create one in Experiments with Run baseline comparison, then Save experiment.')
        return
    if st.session_state.get('saved_runs_truncated'):
        st.warning('Refresh reached the 50-entry limit. This is a bounded listing, not a complete or globally newest history.')
    ready = [entry for entry in entries if entry['status'] == 'Ready']
    if not ready:
        st.info('No completed baseline comparisons available. In Experiments, run a baseline comparison, choose Save experiment, then refresh here.')
        if entries:
            st.dataframe(pd.DataFrame([{'Run': e['run_id'][:8], 'Status': e['status'], 'Details': e['detail']} for e in entries]), hide_index=True, width='stretch')
        return
    choices = [e['run_id'] for e in ready]
    if st.session_state.get('case_study_run') not in choices:
        st.session_state.pop('case_study_run', None)
    run_id = st.selectbox('Saved experiment', choices, key='case_study_run',
                          format_func=lambda value: run_label(next(e for e in ready if e['run_id'] == value)))
    partition = st.selectbox('Evaluation partition', ['validation', 'test'], key='case_study_partition')
    try:
        record = runs.read_record(runs.RUNS_ROOT, run_id)
        study = build_case_study(record, partition)
    except (OSError, ValueError, TypeError, KeyError):
        st.session_state.pop('case_study_context', None)
        st.session_state.pop('case_attach', None)
        st.error('This experiment changed or cannot be read. Refresh the list; preserve the files and save a new comparison. See README: Save and reopen an experiment.')
        return
    identity = (study.run_id, study.content_id, study.partition)
    if st.session_state.get('case_study_context') != identity:
        st.session_state.case_study_context = identity
        st.session_state.case_attach = False
    st.caption(f'Run {study.run_id[:8]} · {study.partition} · content {study.content_id[:8]} · {study.support} evaluated reviews')
    for warning in study.warnings:
        st.warning(warning)
    st.header('Evidence and method')
    st.write(f'Recorded source: {study.source}. Seed {study.seed}. Completed prefix ratio: {study.ratio!r} synthetic examples per real training example.')
    st.write('TF-IDF word uni/bigrams and balanced logistic regression. Both conditions use the same held-out reviews; the prefix is added only to sampled training reviews.')
    with st.expander('Recorded provenance'):
        st.json({'run_id': study.run_id, 'content_digest': study.content_id, 'exact_input_id': study.input_id,
                 'normalised_dataset_id': study.dataset_id, 'split_digest': study.split_digest,
                 'saved_at': study.created_at, 'partition': study.partition,
                 **json.loads(study.recorded_environment)})
    st.header('What changed?')
    st.dataframe(pd.DataFrame([{'Condition': c.name, 'Training reviews': c.training_rows,
                               'Macro F1': f'{c.macro_f1:.4f}', 'Accuracy': f'{c.accuracy:.2%}'} for c in study.conditions]), hide_index=True, width='stretch')
    f1 = 0.0 if round(study.f1_delta, 4) == 0 else study.f1_delta
    pp = 0.0 if round(study.accuracy_delta_pp, 2) == 0 else study.accuracy_delta_pp
    st.write(f'Macro F1 change: {f1:+.4f}. Accuracy change: {pp:+.2f} percentage points. Direction: augmented minus real-only.')
    st.write(study.interpretation)
    st.header('Which mistakes changed?')
    st.dataframe(pd.DataFrame(study.paired_counts, columns=['Paired outcome', 'Reviews']), hide_index=True, width='stretch')
    st.write('Corrected means the augmented model fixed a real-only error; regressed means it introduced one. Counts describe the same evaluated reviews.')
    for category, count in study.paired_counts:
        if not count:
            st.write(f'{category}: no reviews in this partition.')
    with st.expander('Attach original reviews to inspect examples'):
        st.write('Optional. Exact original reviews are required. Examples stay on screen and are excluded from Markdown.')
        sources = ['Authored demo', 'Upload CSV']
        if (ROOT / 'data/reviews.csv').exists():
            sources.append('Local review dataset')
        source = st.selectbox('Original review source', sources, key='case_review_source')
        attached = st.toggle('Attach original reviews', key='case_attach')
        if attached:
            try:
                if source == 'Upload CSV':
                    uploaded = st.file_uploader('Original reviews CSV', type='csv', key=f'case_upload_{study.content_id}_{partition}')
                    if uploaded is None:
                        st.info('Upload the exact original CSV for examples. Numeric results and Markdown remain available.')
                        frame = None
                    elif uploaded.size > 10 * 1024 * 1024:
                        raise ValueError('CSV exceeds 10 MB. Use the exact original supported CSV.')
                    else:
                        frame = load_csv(uploaded)
                elif source == 'Local review dataset':
                    frame = load_csv(ROOT / 'data/reviews.csv')
                else:
                    frame = demo_reviews()
                if frame is not None:
                    with st.spinner('Checking the original dataset…'):
                        examples = case_study_examples(record, frame, partition)
                    st.write('Illustrative mistakes: at most two per corrected, regressed and both-wrong category, sorted by exact review ID. This is not a representative sample.')
                    if not examples:
                        st.info('No mistakes to preview in this partition.')
                    for example in examples:
                        st.write(f'{example.category} · Actual: {example.actual} · Real-only: {example.real_prediction} · Augmented: {example.augmented_prediction}')
                        st.caption(f'Exact review ID: {example.review_id}')
                        st.text(example.text)
            except (OSError, ValueError, TypeError, KeyError) as exc:
                st.error(f'Cannot attach examples: {exc} Numeric evidence is still available. Select the exact original dataset. See README: Save and reopen an experiment.')
    st.button('Inspect this partition in Analysis', on_click=open_analysis, args=(run_id, partition))
    st.caption('Analysis may need you to select or upload the exact original reviews again. This handoff retains the run and partition.')
    st.header('What this does not establish')
    for limitation in LIMITATIONS:
        st.write(limitation)
    st.header('Download the summary')
    st.write('Includes numeric evidence and recorded provenance. Review text is excluded. The download matches the run, content identity and partition shown above.')
    try:
        data = case_study_markdown(study)
    except ValueError as exc:
        st.error(str(exc))
        return
    st.download_button('Download Markdown', data, f'signallab-case-study-{run_id}-{partition}.md',
                       'text/markdown', key=f'case_download_{study.content_id}_{partition}', type='primary', on_click='ignore')
