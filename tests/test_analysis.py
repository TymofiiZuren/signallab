import unittest

import pandas as pd

from signallab.analysis import evidence_report, analysis_rows, slice_metrics


class AnalysisTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame({'text': ['bad item', 'really good product', 'fine'], 'label': [0, 1, 1]})

    def test_alignment_uses_exact_ids_not_positions(self):
        report = evidence_report(self.frame, [0, 0, 1])
        rows = analysis_rows(report, self.frame.iloc[::-1])
        self.assertEqual(rows.prediction.tolist(), [0, 0, 1])
        self.assertEqual(rows.correct.tolist(), [True, False, True])
        changed = self.frame.copy()
        changed.loc[0, 'text'] = 'BAD item'
        with self.assertRaisesRegex(ValueError, 'match'):
            analysis_rows(report, changed)

    def test_legacy_and_invalid_predictions_rejected(self):
        with self.assertRaisesRegex(ValueError, 'provenance'):
            analysis_rows({'predictions': [0, 0, 1]}, self.frame)
        with self.assertRaises(ValueError):
            evidence_report(self.frame, [0])
        with self.assertRaises(ValueError):
            evidence_report(self.frame, [0, 2, 1])

    def test_slices_include_support_and_errors(self):
        rows = analysis_rows(evidence_report(self.frame, [0, 0, 1]), self.frame)
        slices = slice_metrics(rows, 'Actual')
        positive = slices[slices.Slice == 'Positive'].iloc[0]
        self.assertEqual(positive.Reviews, 2)
        self.assertEqual(positive.Errors, 1)
        self.assertEqual(positive.Accuracy, .5)
