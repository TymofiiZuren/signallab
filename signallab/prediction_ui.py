"""Saved transformer controls; importing this module never loads model weights."""
import json
import math
from pathlib import Path
import re

import pandas as pd
import streamlit as st

from signallab.core import LABELS

TRANSFORMER_ROOT = Path(__file__).resolve().parents[1] / 'artifacts/transformer'
VARIANTS = {'real': 'Real-only transformer', 'augmented': 'Prefix-augmented transformer'}


def clear_prediction():
    st.session_state.pop('prediction', None)


def reconcile_prediction(identity):
    if st.session_state.get('prediction_identity') != identity:
        clear_prediction()
        st.session_state.prediction_identity = identity


def saved_model_context(slug):
    from signallab.inference import artifact_identity

    if slug not in VARIANTS:
        raise ValueError('Choose a supported saved model.')
    if TRANSFORMER_ROOT.is_symlink():
        raise ValueError('Saved transformer root must be a regular local directory.')
    report_path = TRANSFORMER_ROOT / 'report.json'
    if report_path.is_symlink() or report_path.stat().st_size > 16 * 1024 * 1024:
        raise ValueError('Completed transformer report is unsafe or too large. Use a supported local training output.')
    report = json.loads(report_path.read_text())
    if not isinstance(report, dict) or type(report.get('schema_version')) is not int or report['schema_version'] != 1:
        raise ValueError('Unsupported transformer report. Use the output of signallab.train.')
    if report.get('training', {}).get('max_length') != 128:
        raise ValueError('Saved model tokenisation policy is unsupported. Retrain with the current command.')
    dataset = report.get('dataset_id')
    checkpoint = report.get('checkpoint')
    ratio = report.get('augmentation', {}).get('ratio')
    seed = report.get('seed')
    if not isinstance(dataset, str) or re.fullmatch('[0-9a-f]{64}', dataset) is None or not isinstance(checkpoint, str) or not 1 <= len(checkpoint) <= 256 or type(ratio) not in (int, float) or not math.isfinite(ratio) or not 0 <= ratio <= 1 or type(seed) is not int:
        raise ValueError('Saved transformer provenance is incomplete. Check the completed training report.')
    return {'identity': artifact_identity(TRANSFORMER_ROOT / slug), 'dataset_id': dataset,
            'checkpoint': checkpoint, 'ratio': ratio, 'seed': seed}


def render_saved_prediction():
    from signallab.inference import predict_saved

    slug = st.selectbox('Saved model', list(VARIANTS), format_func=VARIANTS.get, key='saved_prediction_model', on_change=clear_prediction)
    try:
        context = saved_model_context(slug)
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        clear_prediction()
        st.error('This saved transformer is unavailable or incompatible. Check artifacts/transformer/report.json and the selected model files, or choose the other model. Existing files are preserved.')
        st.code('uv run python -m signallab.train --csv data/reviews.csv --output artifacts/new-transformer-run', language='bash')
        st.caption('New output directories are preserved but are not automatically discovered by this prediction form.')
        return
    identity = ('saved', slug, context['identity'], context['dataset_id'], context['checkpoint'], context['ratio'], context['seed'])
    reconcile_prediction(identity)
    st.caption(f"Saved model · {VARIANTS[slug]} · training dataset {context['dataset_id'][:8]} · seed {context['seed']} · experiment augmentation ratio {context['ratio']:.0%}")
    st.caption(f"Base checkpoint: {context['checkpoint']}. Legacy report: model revision and exact original input provenance were not recorded.")
    st.info('Prediction uses the saved model and its training dataset, independently of the dataset selected in the sidebar. All inference runs locally on CPU.')
    with st.form('saved_prediction_form'):
        text = st.text_area('Review', placeholder='The sound is excellent, but the battery barely lasts an hour.', max_chars=20000)
        submitted = st.form_submit_button('Analyse review', type='primary')
    if submitted:
        clear_prediction()
        try:
            if not text.strip():
                raise ValueError('Enter a review before analysing it.')
            with st.spinner('Loading the local classifier and analysing your review…'):
                result = predict_saved(TRANSFORMER_ROOT / slug, context['identity'], text)
            st.session_state.prediction = {**result, 'model': VARIANTS[slug]}
        except (OSError, ValueError, RuntimeError, KeyError, TypeError) as exc:
            st.error(f'Prediction failed: {exc}. Check the selected local model, or try another completed model. See README: Predict with a saved transformer. Existing files are preserved.')
    if 'prediction' in st.session_state:
        result = st.session_state.prediction
        st.success(f"Predicted sentiment: {LABELS[result['label']]}")
        st.caption(f"Last submitted review · {result['model']}")
        st.dataframe(pd.DataFrame({'Sentiment': list(LABELS.values()), 'Model probability': result['scores']}), hide_index=True, width='stretch')
        if result['truncated']:
            st.warning(f"This review exceeded {result['token_limit']} tokens. The model used its first {result['token_limit']} tokens, including special tokens, matching training/evaluation preprocessing.")
        st.caption('Probabilities are uncalibrated. This binary classifier must choose Positive or Negative, even for neutral, mixed or sarcastic reviews.')
