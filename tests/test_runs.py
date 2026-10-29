"""Persistence contracts: exact identity, truthful metrics and local file boundaries."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from signallab.core import run_comparison, make_splits
from signallab.demo import demo_reviews
from signallab.runs import build_record, validate_record, save_record, read_record, discover_runs, attach_reviews, MAX_REPORT_BYTES

ROOT = Path(__file__).resolve().parents[1]


class SavedRunTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frame = demo_reviews()
        cls.report, _, cls.splits = run_comparison(cls.frame)

    def setUp(self):
        self.record = build_record(self.report, self.frame, self.splits)
        self.temp = tempfile.TemporaryDirectory(dir=ROOT / '.scratch')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / 'runs'

    def test_source_provenance_is_explicit_and_older_snapshots_stay_readable(self):
        report = copy.deepcopy(self.report)
        report["source"] = "Authored demo"
        record = build_record(report, self.frame, self.splits)
        self.assertEqual(record["manifest"]["data_source"], "Authored demo")
        record["manifest"].pop("data_source")
        validate_record(record)
        self.assertEqual(record["manifest"].get("data_source", "Source not recorded"), "Source not recorded")

    def test_save_rejects_report_relabelled_with_other_training_data(self):
        changed = self.frame.copy()
        original = self.splits.train.iloc[0].text
        changed.loc[changed.text == original, "text"] = original.upper()
        with self.assertRaises(ValueError):
            build_record(self.report, changed, make_splits(changed))

    def test_save_uses_training_provenance_not_current_environment(self):
        with patch("signallab.core.version", return_value="999.0"), patch("pathlib.Path.read_bytes", return_value=b"changed source"):
            later = build_record(self.report, self.frame, self.splits)
        self.assertEqual(later["manifest"]["dependencies"], self.record["manifest"]["dependencies"])
        self.assertEqual(later["manifest"]["source_hashes"], self.record["manifest"]["source_hashes"])

    def test_round_trip_preserves_two_immutable_text_free_records(self):
        first = save_record(self.record, self.root)
        second_record = build_record(self.report, self.frame, self.splits)
        second = save_record(second_record, self.root)
        self.assertNotEqual(first, second)
        self.assertEqual(read_record(self.root, first), self.record)
        with self.assertRaises(FileExistsError):
            save_record(self.record, self.root)
        entries, truncated = discover_runs(self.root)
        self.assertFalse(truncated)
        self.assertEqual({r['run_id'] for r in entries}, {first, second})
        serialized = (self.root / first / 'report.json').read_text()
        self.assertNotIn(self.frame.iloc[0].text, serialized)
        self.assertNotIn('"text"', serialized)
        self.assertTrue(self.record['manifest']['source_hashes'])

    def test_exact_input_and_membership_before_analysis(self):
        attached = attach_reviews(self.record, self.frame.sample(frac=1, random_state=1))
        self.assertEqual(attached.validation.text.tolist(), self.splits.validation.text.tolist())
        changed = self.frame.copy()
        changed.loc[0, 'text'] = changed.loc[0, 'text'].upper()
        with self.assertRaisesRegex(ValueError, 'exact'):
            attach_reviews(self.record, changed)
        broken = copy.deepcopy(self.record)
        broken['manifest']['split_ids']['validation'].reverse()
        with self.assertRaises(ValueError):
            validate_record(broken)

    def test_metrics_and_labels_are_validated_not_trusted(self):
        for mutate in (
            lambda r: r['runs'][0]['validation'].update(accuracy=.123),
            lambda r: r['runs'][0]['validation']['evidence'][0].update(prediction=2),
            lambda r: r['runs'][0]['validation']['evidence'][0].update(label=True),
            lambda r: r.update(schema_version=999),
            lambda r: r['augmentation'].update(ratio=float('nan')),
            lambda r: r.update(text='must never be persisted'),
        ):
            record = copy.deepcopy(self.record)
            mutate(record)
            with self.assertRaises(ValueError):
                validate_record(record)

    def test_discovery_is_explicit_bounded_and_handles_bad_entries(self):
        self.assertEqual(discover_runs(self.root), ([], False))
        run_id = save_record(self.record, self.root)
        (self.root / 'partial').mkdir()
        (self.root / 'broken').mkdir()
        (self.root / 'broken/report.json').write_text('{')
        (self.root / 'legacy').mkdir()
        (self.root / 'legacy/report.json').write_text(json.dumps(self.report))
        entries, _ = discover_runs(self.root)
        statuses = {entry['run_id']: entry['status'] for entry in entries}
        self.assertEqual(statuses[run_id], 'Ready')
        self.assertEqual(statuses['partial'], 'Incomplete')
        self.assertEqual(statuses['broken'], 'Invalid')
        self.assertEqual(statuses['legacy'], 'Legacy')
        entries, limited = discover_runs(self.root, limit=2)
        self.assertTrue(limited)
        self.assertEqual(len(entries), 2)

    def test_no_symlink_or_traversal_and_bounded_file_reads(self):
        run_id = save_record(self.record, self.root)
        with self.assertRaises(ValueError):
            read_record(self.root, '../outside')
        outside = Path(self.temp.name) / 'outside'
        outside.mkdir()
        (outside / 'report.json').write_text(json.dumps(self.record))
        (self.root / 'link').symlink_to(outside, target_is_directory=True)
        with self.assertRaises(ValueError):
            read_record(self.root, 'link')
        original = self.root / run_id / 'report.json'
        original.write_bytes(b' ' * (MAX_REPORT_BYTES + 1))
        with self.assertRaisesRegex(ValueError, 'size'):
            read_record(self.root, run_id)

    def test_malformed_json_future_versions_and_root_errors(self):
        self.root.mkdir()
        fixtures = {
            "duplicate": '{"schema_version":2,"schema_version":1}',
            "nonfinite": '{"schema_version":NaN}',
            "future": '{"schema_version":99}',
            "null": 'null',
        }
        for name, content in fixtures.items():
            folder = self.root / name
            folder.mkdir()
            (folder / "report.json").write_text(content)
        entries, _ = discover_runs(self.root)
        states = {entry["run_id"]: entry["status"] for entry in entries}
        self.assertEqual(states, {"duplicate": "Invalid", "nonfinite": "Invalid", "future": "Incompatible", "null": "Invalid"})
        root_file = Path(self.temp.name) / "file-root"
        root_file.write_text("not a directory")
        with self.assertRaises(OSError):
            discover_runs(root_file)
        with patch("signallab.runs.os.scandir", side_effect=PermissionError("denied")):
            with self.assertRaises(PermissionError):
                discover_runs(self.root)

    def test_symlinked_report_and_root_are_not_followed(self):
        first = save_record(self.record, self.root)
        linked_id = "a" * 32
        linked = self.root / linked_id
        linked.mkdir()
        (linked / "report.json").symlink_to(self.root / first / "report.json")
        with self.assertRaises(ValueError):
            read_record(self.root, linked_id)
        root_link = Path(self.temp.name) / "root-link"
        root_link.symlink_to(self.root, target_is_directory=True)
        with self.assertRaises(ValueError):
            discover_runs(root_link)
        with self.assertRaises(ValueError):
            read_record(root_link, first)

    def test_failed_publish_never_looks_completed(self):
        with patch('signallab.runs.os.replace', side_effect=OSError('simulated disk failure')):
            with self.assertRaises(OSError):
                save_record(self.record, self.root)
        self.assertFalse((self.root / self.record['run_id'] / 'report.json').exists())
        entries, _ = discover_runs(self.root)
        self.assertEqual(entries[0]['status'], 'Incomplete')
