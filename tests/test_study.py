"""Frozen protocol contracts; no network, model assets or held-out text output."""
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import subprocess
import sys
import unittest
from unittest.mock import patch

import pandas as pd

from signallab.core import canonical_review_key, exact_dataset_id, make_splits, review_id, validate_reviews
from signallab import study

ROOT = Path(__file__).resolve().parents[1]


def reviews(prefix='development', per_class=50):
    return pd.DataFrame([
        {'text': f'{prefix} {"excellent" if label else "broken"} item {i}', 'label': label}
        for label in (0, 1) for i in range(per_class)
    ])


class StudyProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dev = reviews()
        cls.final = reviews('reserved', 10)
        cls.scratch = Path(tempfile.mkdtemp(prefix='study-contract-', dir=ROOT / '.scratch'))

    def build(self, **kwargs):
        return study.build_protocol(self.dev, self.final, source='Authored development fixture',
                                    final_source='Separate authored reserved fixture',
                                    attest_final_reserved=True, **kwargs)

    def test_final_overlap_is_rejected_even_with_changed_label_or_prefix(self):
        for text in (self.dev.iloc[0].text, ' In my experience, In my experience, ' + self.dev.iloc[0].text.upper() + '  '):
            final = self.final.copy()
            final.loc[0, ['text', 'label']] = [text, 1]
            with self.assertRaisesRegex(ValueError, 'overlap'):
                study.build_protocol(self.dev, final, source='dev', final_source='final', attest_final_reserved=True)

    def test_canonical_key_keeps_legacy_dedup_and_exact_identity(self):
        self.assertEqual(canonical_review_key(' In MY experience, in my experience, A  REVIEW\n'), 'a review')
        self.assertEqual(canonical_review_key('A  REVIEW'), 'a review')
        extra = self.dev.iloc[[0]].copy()
        extra['text'] = 'In my experience, ' + extra.text
        clean = validate_reviews(pd.concat([self.dev, extra]))
        expected = validate_reviews(self.dev)
        # Legacy tie-break sorts literal text: uppercase I precedes lowercase d.
        expected.loc[0, 'text'] = extra.iloc[0].text
        pd.testing.assert_frame_equal(clean, expected)
        self.assertEqual(exact_dataset_id(clean), exact_dataset_id(expected))

    def test_frozen_membership_is_authoritative_and_attachment_returns_copies(self):
        record = self.build()
        expected = make_splits(self.dev)
        with patch('signallab.study.make_splits', side_effect=AssertionError('must not resplit')):
            actual = study.attach_development(record, self.dev.sample(frac=1, random_state=7))
        for name in ('train', 'validation', 'test'):
            pd.testing.assert_frame_equal(getattr(actual, name), getattr(expected, name))
        actual.train.loc[0, 'text'] = 'changed caller copy'
        self.assertNotEqual(study.attach_development(record, self.dev).train.iloc[0].text, 'changed caller copy')
        for field in ('text', 'label'):
            changed = self.dev.copy()
            changed.loc[0, field] = changed.loc[0, field].upper() if field == 'text' else 1
            with self.assertRaisesRegex(ValueError, 'exact'):
                study.attach_development(record, changed)

    def test_selection_is_balanced_training_only_and_fixed(self):
        record = self.build()
        again = study.build_protocol(self.dev.sample(frac=1, random_state=2), self.final,
                                     source='dev', final_source='final', attest_final_reserved=True)
        self.assertEqual(record['selection']['parent_ids'], again['selection']['parent_ids'])
        rows = {r['review_id']: r for r in record['development']['rows']}
        parents = record['selection']['parent_ids']
        self.assertEqual(len(parents), 50)
        self.assertTrue(set(parents) <= set(record['development']['split_ids']['train']))
        self.assertEqual([rows[x]['label'] for x in parents], [0, 1] * 25)
        for label in (0, 1):
            ids = [x for x in record['development']['split_ids']['train'] if rows[x]['label'] == label]
            expected = sorted(ids, key=lambda x: hashlib.sha256(f'selection-v1|42|{x}'.encode()).hexdigest())[:25]
            self.assertEqual([x for x in parents if rows[x]['label'] == label], expected)
        small = study.build_protocol(reviews(per_class=8), self.final, source='dev', final_source='final', attest_final_reserved=True)
        self.assertLess(len(small['selection']['parent_ids']), 30)
        self.assertEqual(small['policy']['feasibility']['accepted_min'], 30)

    def test_protocol_contains_no_original_or_final_text_or_final_labels(self):
        record = self.build()
        encoded = json.dumps(record)
        for text in list(self.dev.text) + list(self.final.text):
            self.assertNotIn(text, encoded)
        self.assertEqual(set(record['final_cohort']), {'input_id', 'fingerprints', 'count', 'source', 'declaration'})
        self.assertEqual(record['development']['input_id'], exact_dataset_id(validate_reviews(self.dev)))
        self.assertEqual(record['final_cohort']['input_id'], exact_dataset_id(validate_reviews(self.final)))
        self.assertEqual(record['policy']['generator']['revision'], '0fc9ddf78a1e988dac52e2dac162b0ede4fd74ab')
        self.assertEqual(record['policy']['resources']['rss_bytes'], 4 * 1024**3)
        self.assertEqual(record['provenance']['source_hashes'].keys(), {'core.py', 'study.py'})

    def test_declaration_sources_and_integer_settings_are_required(self):
        for overrides in ({'attest_final_reserved':False}, {'attest_final_reserved':1}, {'source':''},
                          {'final_source':'  '}, {'source':'x'*2001}, {'max_parents':True},
                          {'max_parents':0}, {'max_parents':51}, {'threads':0}, {'threads':9}, {'threads':1.5}):
            args = dict(source='dev', final_source='final', attest_final_reserved=True)
            args.update(overrides)
            with self.assertRaises(ValueError):
                study.build_protocol(self.dev, self.final, **args)
        self.assertEqual(len(self.build(max_parents=1, threads=1)['selection']['parent_ids']), 1)

    def test_round_trip_is_private_text_free_and_never_overwrites(self):
        record = self.build()
        root = self.scratch / record['protocol_id']
        self.assertEqual(study.save_protocol(record, root), record['protocol_id'])
        self.assertEqual(study.read_protocol(root, record['protocol_id']), record)
        saved = root / record['protocol_id'] / 'protocol.json'
        original = saved.read_bytes()
        self.assertEqual(saved.stat().st_mode & 0o777, 0o600)
        self.assertEqual(saved.parent.stat().st_mode & 0o777, 0o700)
        with self.assertRaises(FileExistsError):
            study.save_protocol(record, root)
        self.assertEqual(saved.read_bytes(), original)

    def test_bounded_csv_and_freeze_validate_before_creating_output(self):
        dev = io.BytesIO(self.dev.to_csv(index=False).encode())
        final = io.BytesIO(self.final.to_csv(index=False).encode())
        root = self.scratch / 'freeze-valid'
        record = study.freeze_protocol(dev, final, root=root, source='dev', final_source='final', attest_final_reserved=True)
        self.assertEqual(study.read_protocol(root, record['protocol_id']), record)
        for content in (b'', b'wrong,columns\n1,2', b'text,label\n\xff,0', b'"unterminated', b'x' * (study.MAX_CSV_BYTES + 1)):
            with self.subTest(size=len(content)):
                rejected = self.scratch / ('rejected-' + hashlib.sha256(content).hexdigest())
                with self.assertRaises(ValueError):
                    study.freeze_protocol(io.BytesIO(content), io.BytesIO(self.final.to_csv(index=False).encode()),
                                          root=rejected, source='dev', final_source='final', attest_final_reserved=True)
                self.assertFalse(rejected.exists())
        with self.assertRaises(FileNotFoundError):
            study.load_study_csv(self.scratch / 'absent.csv')

    def test_reader_rejects_corruption_types_digests_and_membership(self):
        record = self.build()
        mutations = [
            lambda r: r.update(schema_version=True),
            lambda r: r.update(schema_version=999),
            lambda r: r.update(created_at='yesterday'),
            lambda r: r.update(seed=True),
            lambda r: r.update(threads=True),
            lambda r: r['development']['split_ids']['test'].append(r['development']['split_ids']['train'][0]),
            lambda r: r['development']['rows'][0].update(label=True),
            lambda r: r['final_cohort'].update(count=True),
            lambda r: r['final_cohort']['fingerprints'].append(r['final_cohort']['fingerprints'][0]),
            lambda r: r['selection']['parent_ids'].reverse(),
            lambda r: r['policy']['generator'].update(do_sample=0),
            lambda r: r['policy']['resources'].update(wall_seconds=float('nan')),
            lambda r: r['provenance'].update(source_hashes={}),
            lambda r: r.update(extra='unsupported'),
        ]
        for mutate in mutations:
            broken = copy.deepcopy(record)
            mutate(broken)
            # Re-sign finite malformed contracts to test validation beyond the digest.
            try:
                broken['content_digest'] = study._digest({k:v for k,v in broken.items() if k != 'content_digest'})
            except ValueError:
                pass
            with self.assertRaises(ValueError):
                study.validate_protocol(broken)
        broken = copy.deepcopy(record)
        broken['threads'] = 1
        with self.assertRaisesRegex(ValueError, 'digest'):
            study.validate_protocol(broken)

    def test_safe_reader_rejects_symlinks_pending_and_bad_documents(self):
        record = self.build()
        root = self.scratch / ('safe-' + record['protocol_id'])
        study.save_protocol(record, root)
        linked = self.scratch / ('linked-' + record['protocol_id'])
        linked.symlink_to(root, target_is_directory=True)
        with self.assertRaises(ValueError):
            study.read_protocol(linked, record['protocol_id'])
        bad_id = 'b' * 32
        (root / bad_id).symlink_to(root / record['protocol_id'], target_is_directory=True)
        with self.assertRaises(ValueError):
            study.read_protocol(root, bad_id)
        for name in ('../escape', '', None, 'a'*33):
            with self.assertRaises(ValueError):
                study.read_protocol(root, name)
        for number, content in enumerate((b'{', b'{"a":1,"a":2}', b'{"a":NaN}', b'['*1500, b'\xff', b'x'*(study.MAX_PROTOCOL_BYTES+1))):
            identifier = f'{number:032x}'
            folder = root / identifier
            folder.mkdir()
            (folder / 'protocol.json').write_bytes(content)
            with self.assertRaises(ValueError):
                study.read_protocol(root, identifier)
        pending_id = 'c'*32
        (root / pending_id).mkdir()
        (root / pending_id / 'protocol.pending').write_text('{}')
        with self.assertRaises(FileNotFoundError):
            study.read_protocol(root, pending_id)

    def test_file_publication_failure_never_reports_success(self):
        record = self.build()
        root = self.scratch / ('failed-' + record['protocol_id'])
        with patch('signallab.study.os.link', side_effect=OSError('injected full disk')):
            with self.assertRaises(OSError):
                study.save_protocol(record, root)
        self.assertTrue((root / record['protocol_id'] / 'protocol.pending').exists())
        self.assertFalse((root / record['protocol_id'] / 'protocol.json').exists())

    def test_csv_row_and_text_limits_and_empty_classes(self):
        invalid = [reviews(per_class=7), pd.DataFrame({'text':['x'], 'label':['neutral']}),
                   pd.DataFrame({'text':[None], 'label':[0]}),
                   pd.DataFrame({'text':['x'*20001], 'label':[0]}),
                   pd.DataFrame({'text':['x']*50001, 'label':[0]*50001})]
        for frame in invalid:
            with self.assertRaises(ValueError):
                study.build_protocol(frame, self.final, source='dev', final_source='final', attest_final_reserved=True)

    def test_csv_path_stream_bounds_and_artifact_file_symlink(self):
        folder = self.scratch / 'csv-paths'
        folder.mkdir(exist_ok=True)
        csv = folder / 'original.csv'
        csv.write_text(self.dev.to_csv(index=False))
        pd.testing.assert_frame_equal(study.load_study_csv(csv), validate_reviews(self.dev))
        linked = folder / 'linked.csv'
        linked.symlink_to(csv)
        with self.assertRaises(OSError):
            study.load_study_csv(linked)
        with self.assertRaises(ValueError):
            study.load_study_csv(io.StringIO('é' * (study.MAX_CSV_BYTES // 2 + 1)))
        record = self.build()
        root = folder / 'studies'
        study.save_protocol(record, root)
        other_id = 'd' * 32
        (root / other_id).mkdir()
        (root / other_id / 'protocol.json').symlink_to(root / record['protocol_id'] / 'protocol.json')
        with self.assertRaises(ValueError):
            study.read_protocol(root, other_id)
        with self.assertRaises(ValueError):
            study.save_protocol(record, linked)

    def test_parallel_publish_has_one_winner_and_keeps_the_exact_record(self):
        record = self.build()
        root = self.scratch / ('parallel-' + record['protocol_id'])
        program = ('import json,sys; from signallab.study import save_protocol; '
                   'save_protocol(json.load(sys.stdin), sys.argv[1])')
        processes = [subprocess.Popen([sys.executable, '-c', program, str(root)],
                                      stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                      text=True, cwd=ROOT) for _ in range(2)]
        for process in processes:
            process.stdin.write(json.dumps(record))
            process.stdin.close()
        outcomes = []
        for process in processes:
            process.wait(timeout=20)
            outcomes.append((process.returncode, process.stderr.read()))
            process.stdout.close()
            process.stderr.close()
        self.assertEqual(sorted(code for code, _ in outcomes), [0, 1])
        self.assertIn('FileExistsError', next(error for code, error in outcomes if code))
        self.assertEqual(study.read_protocol(root, record['protocol_id']), record)

    def test_invalid_partition_shape_is_rejected_even_with_updated_digests(self):
        record = self.build()
        ids = record['development']['split_ids']
        ids['train'].append(ids['test'].pop())
        record['development']['split_digest'] = study._digest(ids)
        record['selection']['parent_ids'] = study._select(ids, record['development']['rows'], 50)
        record['content_digest'] = study._digest({k:v for k,v in record.items() if k != 'content_digest'})
        with self.assertRaisesRegex(ValueError, 'partition sizes'):
            study.validate_protocol(record)

    def test_missing_or_wrong_frame_inputs_are_actionable_validation_errors(self):
        record = self.build()
        for frame in (None, [], {'text': ['review'], 'label': [0]}):
            with self.assertRaisesRegex(ValueError, 'DataFrame'):
                study.build_protocol(frame, self.final, source='dev', final_source='final', attest_final_reserved=True)
            with self.assertRaisesRegex(ValueError, 'DataFrame'):
                study.build_protocol(self.dev, frame, source='dev', final_source='final', attest_final_reserved=True)
            with self.assertRaisesRegex(ValueError, 'DataFrame'):
                study.attach_development(record, frame)


if __name__ == '__main__':
    unittest.main()
