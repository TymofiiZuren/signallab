"""Review-level evidence shared by baseline and transformer evaluations.

Exact IDs deliberately differ from the normalised dataset deduplication hash.
Reports contain IDs and labels, never review text.
"""
import pandas as pd

from signallab.core import LABELS, evidence_report, review_id


def analysis_rows(report, frame):
    evidence = report.get('evidence')
    if report.get('evidence_version') != 1 or not isinstance(evidence, list):
        raise ValueError('This report lacks exact review provenance. Run a new experiment for detailed analysis.')
    lookup = {review_id(row.text, row.label): row for row in frame.itertuples()}
    ids = [item.get('review_id') for item in evidence]
    if len(set(ids)) != len(ids) or set(ids) != set(lookup):
        raise ValueError('Evaluated reviews do not match this dataset partition. Select the original dataset or rerun.')
    records = []
    for item in evidence:
        row = lookup[item['review_id']]
        prediction = item.get('prediction')
        if item.get('label') != row.label or prediction not in LABELS:
            raise ValueError('Evaluation labels are invalid or do not match the reviews.')
        words = len(row.text.split())
        records.append(dict(review_id=item['review_id'], text=row.text, label=int(row.label),
                            prediction=prediction, Actual=LABELS[row.label], Predicted=LABELS[prediction],
                            correct=row.label == prediction, Words=words,
                            Length='Short (≤20 words)' if words <= 20 else 'Medium (21–80 words)' if words <= 80 else 'Long (>80 words)'))
    return pd.DataFrame(records)


def slice_metrics(rows, dimension):
    return pd.DataFrame([
        {'Slice': name, 'Reviews': len(group), 'Errors': int((~group.correct).sum()),
         'Accuracy': float(group.correct.mean())}
        for name, group in rows.groupby(dimension, sort=True)
    ])
