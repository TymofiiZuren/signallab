"""Evidence, privacy and interpretation contracts, using internally valid runs."""
import copy
import unittest
from unittest.mock import patch

from signallab.core import run_comparison, prediction_report
from signallab.demo import demo_reviews
from signallab.runs import build_record, validate_record
from signallab.case_study import build_case_study, case_study_markdown, case_study_examples


class CaseStudyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frame = demo_reviews()
        report, _, splits = run_comparison(cls.frame)
        report['source'] = 'Authored demo'
        cls.original = build_record(report, cls.frame, splits)

    def setUp(self):
        self.record = copy.deepcopy(self.original)

    def predictions(self, first, second):
        for run, values in zip(self.record['runs'], (first, second)):
            evaluation = run['validation']
            for row, prediction in zip(evaluation['evidence'], values):
                row['prediction'] = prediction
            evaluation.update(prediction_report([e['label'] for e in evaluation['evidence']], values))
        validate_record(self.record)

    def test_signed_changes_and_no_change_are_descriptive(self):
        labels = [e['label'] for e in self.record['runs'][0]['validation']['evidence']]
        wrong = [1-x for x in labels]
        for first, second, expected, word in ((wrong, labels, 1, 'higher'), (labels, wrong, -1, 'lower'), (labels, labels, 0, 'No measured change')):
            with self.subTest(expected=expected):
                self.predictions(first, second)
                study = build_case_study(self.record)
                self.assertEqual(study.f1_delta, expected)
                self.assertEqual(study.accuracy_delta_pp, expected*100)
                self.assertIn(word, study.interpretation)
                self.assertEqual(sum(dict(study.paired_counts).values()), study.support)

    def test_all_paired_categories_and_bounded_deterministic_examples(self):
        labels = [e['label'] for e in self.record['runs'][0]['validation']['evidence']]
        first = [y if i%4 in (1,3) else 1-y for i,y in enumerate(labels)]
        second = [y if i%4 in (0,3) else 1-y for i,y in enumerate(labels)]
        self.predictions(first, second)
        study = build_case_study(self.record)
        self.assertEqual(dict(study.paired_counts), {'Corrected':3,'Regressed':3,'Both wrong':3,'Both correct':3})
        examples = case_study_examples(self.record, self.frame)
        self.assertEqual(len(examples), 6)
        self.assertEqual(examples, case_study_examples(self.record, self.frame.sample(frac=1, random_state=3)))
        for category in ('Corrected','Regressed','Both wrong'):
            ids = [e.review_id for e in examples if e.category == category]
            self.assertEqual(ids, sorted(ids))
        self.assertTrue(all(e.text for e in examples))

    def test_exact_attachment_blocks_case_change_and_no_errors_needs_no_examples(self):
        changed = self.frame.copy()
        changed.loc[0,'text'] = changed.loc[0,'text'].upper()
        with self.assertRaises(ValueError):
            case_study_examples(self.record, changed)
        labels = [e['label'] for e in self.record['runs'][0]['validation']['evidence']]
        self.predictions(labels, labels)
        self.assertEqual(case_study_examples(self.record, self.frame), ())

    def test_text_free_export_keeps_identity_metrics_limits_and_environment(self):
        study = build_case_study(self.record)
        # Text-bearing examples have a separate contract and never enter the exporter.
        case_study_examples(self.record, self.frame)
        data = case_study_markdown(study)
        text = data.decode('utf-8')
        for value in (study.run_id, study.content_id, study.input_id, study.split_digest, 'validation', 'Authored demo', 'Review text is excluded', 'uncalibrated', 'single seed'):
            self.assertIn(value, text)
        for review in self.frame.text:
            self.assertNotIn(review, text)
        self.assertIn('scikit-learn',text)
        self.assertNotIn('python -m signallab.train',text)
        self.assertNotIn('/Users/',text)
        self.assertEqual(data, case_study_markdown(build_case_study(self.record)))

    def test_source_warnings_test_partition_and_invalid_records(self):
        self.record['manifest'].pop('data_source')
        study = build_case_study(self.record, 'test')
        self.assertTrue(any('Source not recorded' in s for s in study.warnings))
        self.assertTrue(any('Repeated inspection' in s for s in study.warnings))
        self.assertIn('test',case_study_markdown(study).decode())
        for partition in ('train', '', None):
            with self.assertRaises(ValueError):
                build_case_study(self.record, partition)
        self.record['schema_version'] = 1
        with self.assertRaises(ValueError):
            build_case_study(self.record)

    def test_same_id_content_replacement_changes_story_identity(self):
        first = build_case_study(self.record)
        self.record['manifest']['python'] = 'recorded-other-version'
        second = build_case_study(self.record)
        self.assertEqual(first.run_id, second.run_id)
        self.assertNotEqual(first.content_id, second.content_id)
        self.assertNotEqual(case_study_markdown(first),case_study_markdown(second))

    def test_export_bounds_and_metadata_fences(self):
        self.record['manifest']['dependencies']['numpy'] = '```\n![untrusted](https://example.invalid)\n```'
        text = case_study_markdown(build_case_study(self.record)).decode()
        self.assertIn('````json',text)
        self.assertIn('\\n',text)
        self.assertNotIn('\n![untrusted]',text)
        with patch('signallab.case_study.MAX_MARKDOWN_BYTES',10):
            with self.assertRaisesRegex(ValueError, '1 MiB'):
                case_study_markdown(build_case_study(self.record))

    def test_subprecision_delta_is_not_reported_as_exact_tie(self):
        from signallab.case_study import describe_change
        self.assertIn('precision',describe_change(.0000001))
        self.assertIn('precision',describe_change(-.0000001))

    def test_export_retains_exact_augmentation_setting(self):
        report, _, splits = run_comparison(self.frame, ratio=.375)
        record = build_record(report, self.frame, splits)
        text = case_study_markdown(build_case_study(record)).decode()
        self.assertIn('"ratio": 0.375', text)
        self.assertIn('ratio=0.375', text)
