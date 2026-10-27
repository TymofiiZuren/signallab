"""Review workspace contracts; all generation is explicitly authored test evidence."""
from pathlib import Path
import tempfile
import unittest
import uuid
from unittest.mock import patch

from streamlit.testing.v1 import AppTest
from signallab import study, synthetic
from signallab.demo import demo_reviews

ROOT = Path(__file__).resolve().parents[1]
ASSETS = {n: 'a'*64 for n in ('config.json', 'tokenizer.json', 'tokenizer_config.json', 'model.safetensors')}
RESOURCES = dict(elapsed_seconds=1., peak_rss_bytes=1000, samples=4, method='macos-ps-rss-kib-v1')


class SyntheticUITests(unittest.TestCase):
    def setUp(self):
        self.folder = Path(tempfile.mkdtemp(prefix='synthetic-ui-', dir=ROOT / '.scratch'))
        self.studies, self.pilots = self.folder / 'studies', self.folder / 'pilots'
        self.frame = demo_reviews()
        self.protocol = study.build_protocol(self.frame, self.frame.assign(text='reserved fixture ' + self.frame.text),
                                            source='Authored demo', final_source='Authored reserved fixture',
                                            max_parents=4, attest_final_reserved=True)
        study.save_protocol(self.protocol, self.studies)
        self.pilot = synthetic.create_pilot(self.protocol['protocol_id'], self.frame, ASSETS,
                                            root=self.pilots, studies_root=self.studies)
        with synthetic.generation(self.pilots, self.pilot, studies_root=self.studies) as writer:
            for i, parent in enumerate(self.protocol['selection']['parent_ids']):
                writer.start(parent, 20)
                writer.generated(parent, '<b>literal candidate</b>' if i == 0 else f'Authored candidate {i}', 12, True)
            writer.finish('complete', RESOURCES, worker_reaped=True)

    def app(self):
        script = ('import streamlit as st\nfrom signallab.synthetic_ui import render_synthetic_lab\n'
                  f'render_synthetic_lab(studies_root={str(self.studies)!r}, root={str(self.pilots)!r})')
        app = AppTest.from_string(script, default_timeout=30).run()
        self.button(app, 'Refresh local studies').click().run()
        self.assertFalse(app.exception)
        return app

    @staticmethod
    def button(app, label):
        return next(b for b in app.button if b.label == label)

    @staticmethod
    def select(app, label):
        return next(s for s in app.selectbox if s.label == label)

    def attach(self, app):
        self.select(app, 'Development source').set_value('Authored demo').run()
        self.assertFalse(app.exception)

    def submit(self, app):
        next(r for r in app.radio if r.label == 'Human decision').set_value('Accept')
        app.text_area[0].set_value('Meaning and sentiment checked.')
        app.checkbox[0].check()
        self.button(app, 'Save decision and next pending').click().run()

    def test_discovery_is_bounded_and_preserves_invalid_entries(self):
        from signallab.synthetic_ui import discover_local
        (self.pilots / ('f'*32)).mkdir()
        protocols, pilots, truncated = discover_local(self.studies, self.pilots)
        self.assertFalse(truncated)
        self.assertEqual(len(protocols), 1)
        self.assertEqual({p['status'] for p in pilots}, {'complete', 'Unreadable'})
        for i in range(51):
            (self.studies / f'{i:032x}').mkdir(exist_ok=True)
        protocols, _, truncated = discover_local(self.studies, self.pilots)
        self.assertEqual(len(protocols), 50)
        self.assertTrue(truncated)

    def test_attachment_required_literal_pair_and_save_advances(self):
        app = self.app()
        self.assertFalse(app.text_area)
        self.assertTrue(any('Unattached' in e.value for e in app.info))
        self.attach(app)
        self.assertEqual(app.radio[0].value, None)
        self.assertIn('<b>literal candidate</b>', [e.value for e in app.text])
        before = self.select(app, 'Candidate').value
        self.submit(app)
        self.assertFalse(app.exception)
        self.assertNotEqual(self.select(app, 'Candidate').value, before)
        self.assertTrue(any('revision 1' in e.value for e in app.success))
        saved = synthetic.read_reviews(self.pilots, self.pilot, studies_root=self.studies)
        self.assertEqual(saved['revision'], 1)
        self.assertEqual(saved['decisions'][before]['verdict'], 'accept')
        self.assertIsNone(app.radio[0].value)

    def test_no_default_verdict_and_invalid_submit_does_not_advance(self):
        app = self.app()
        self.attach(app)
        before = self.select(app, 'Candidate').value
        self.button(app, 'Save decision and next pending').click().run()
        self.assertTrue(app.error)
        self.assertEqual(before, self.select(app, 'Candidate').value)
        self.assertEqual(synthetic.read_reviews(self.pilots, self.pilot, studies_root=self.studies)['revision'], 0)

    def test_mismatch_removes_pair_and_form_but_keeps_summary(self):
        app = self.app()
        self.attach(app)
        changed = self.frame.copy()
        changed.loc[0, 'text'] += ' changed'
        with patch('signallab.synthetic_ui.demo_reviews', return_value=changed):
            app.run()
        self.assertFalse(app.exception)
        self.assertFalse(app.text_area)
        self.assertFalse(app.text)
        self.assertTrue(any('Mismatch' in e.value for e in app.error))
        self.assertTrue(any('Review completion' in e.value for e in app.markdown))

    def test_conflict_retains_draft_and_requires_explicit_reload(self):
        app = self.app()
        self.attach(app)
        candidate = self.select(app, 'Candidate').value
        synthetic.save_review(self.pilots, self.pilot, self.frame, candidate, verdict='reject', reason='Another reviewer decision.',
                              rubric_ack=True, expected_revision=0, decision_id=uuid.uuid4().hex, studies_root=self.studies)
        self.submit(app)
        self.assertFalse(app.exception)
        self.assertTrue(any('changed' in e.value.lower() for e in app.error))
        self.assertIn('Meaning and sentiment checked.', [e.value for e in app.text])
        self.assertEqual(synthetic.read_reviews(self.pilots, self.pilot, studies_root=self.studies)['revision'], 1)
        self.button(app, 'Reload current decision').click().run()
        self.assertEqual(app.radio[0].value, 'Reject')
        self.assertEqual(app.text_area[0].value, 'Another reviewer decision.')

    def test_corrupt_selected_pilot_removes_stale_form(self):
        app = self.app()
        self.attach(app)
        (self.pilots / self.pilot / 'terminal.json').write_text('{}')
        app.run()
        self.assertFalse(app.exception)
        self.assertFalse(app.text_area)
        self.assertFalse(app.text)
        self.assertTrue(app.error)

    def test_route_is_before_shared_upload_gate(self):
        app = AppTest.from_file(str(ROOT / 'app.py'), default_timeout=30).run()
        app.sidebar.selectbox[0].set_value('Upload CSV').run()
        app.sidebar.radio[0].set_value('Synthetic Review Lab').run()
        self.assertFalse(app.exception)
        self.assertIn('Synthetic Review Lab', [e.value for e in app.title])
        self.assertNotIn('Bring your reviews', [e.value for e in app.title])

    def test_acceptance_does_not_override_automatic_exclusion(self):
        second = synthetic.create_pilot(self.protocol['protocol_id'], self.frame, ASSETS, root=self.pilots, studies_root=self.studies)
        with synthetic.generation(self.pilots, second, studies_root=self.studies) as writer:
            for parent in self.protocol['selection']['parent_ids']:
                writer.start(parent, 20)
                writer.generated(parent, 'Same authored duplicate.', 12, True)
            writer.finish('complete', RESOURCES, worker_reaped=True)
        app = self.app()
        self.select(app, 'Generation pilot').set_value(second).run()
        self.attach(app)
        self.select(app, 'Review filter').set_value('Auto-ineligible').run()
        candidate = self.select(app, 'Candidate').value
        self.submit(app)
        saved = synthetic.read_reviews(self.pilots, second, studies_root=self.studies)
        self.assertEqual(saved['decisions'][candidate]['verdict'], 'accept')
        self.assertTrue(synthetic.read_pilot(self.pilots, second, studies_root=self.studies)['candidates'][candidate]['exclusions'])
        self.assertEqual(self.select(app, 'Review filter').value, 'Pending')
        self.select(app, 'Review filter').set_value('All').run()
        self.select(app, 'Candidate').set_value(candidate).run()
        self.assertTrue(any('Training eligibility: Excluded' in e.value for e in app.markdown))

    def test_lost_response_retry_keeps_identity_and_does_not_double_commit(self):
        app = self.app()
        self.attach(app)
        real_save = synthetic.save_review
        def committed_then_failed(*args, **kwargs):
            real_save(*args, **kwargs)
            raise OSError('simulated lost acknowledgement')
        with patch('signallab.synthetic.save_review', side_effect=committed_then_failed):
            self.submit(app)
        self.assertFalse(app.exception)
        self.assertTrue(app.error)
        self.assertFalse(any('Decision saved' in e.value for e in app.success))
        self.button(app, 'Save decision and next pending').click().run()
        self.assertFalse(app.exception)
        self.assertTrue(any('revision 1' in e.value for e in app.success))
        self.assertEqual(synthetic.read_reviews(self.pilots, self.pilot, studies_root=self.studies)['revision'], 1)

    def test_running_pilot_has_no_racing_counters_or_review_form(self):
        running = synthetic.create_pilot(self.protocol['protocol_id'], self.frame, ASSETS, root=self.pilots, studies_root=self.studies)
        with synthetic.generation(self.pilots, running, studies_root=self.studies):
            app = self.app()
            self.select(app, 'Generation pilot').set_value(running).run()
            self.assertFalse(app.exception)
            self.assertFalse(app.text_area)
            self.assertFalse(app.file_uploader)
            self.assertTrue(any('Generation: running' in e.value for e in app.markdown))
            self.assertFalse(any('0 of 0' in e.value for e in app.markdown))

    def test_source_and_page_navigation_reset_editor_and_attachment(self):
        with patch('signallab.study.STUDIES_ROOT', self.studies), patch('signallab.synthetic.SYNTHETIC_ROOT', self.pilots):
            app = AppTest.from_file(str(ROOT / 'app.py'), default_timeout=30).run()
            app.sidebar.radio[0].set_value('Synthetic Review Lab').run()
            self.button(app, 'Refresh local studies').click().run()
            self.attach(app)
            self.button(app, 'Save decision and next pending').click().run()
            nonce = app.session_state['synthetic_editor']['nonce']
            app.sidebar.radio[0].set_value('Explore').run()
            app.sidebar.radio[0].set_value('Synthetic Review Lab').run()
            self.assertFalse(app.exception)
            self.assertFalse(app.text_area)
            self.attach(app)
            self.assertNotEqual(nonce, app.session_state['synthetic_editor']['nonce'])
            self.select(app, 'Development source').set_value('Not attached').run()
            self.assertFalse(app.text)
            self.assertNotIn('synthetic_editor', app.session_state)

    def test_last_pending_save_reports_success_without_another_form(self):
        current = synthetic.read_pilot(self.pilots, self.pilot, studies_root=self.studies)
        for revision, cid in enumerate(list(current['candidates'])[:-1]):
            synthetic.save_review(self.pilots, self.pilot, self.frame, cid, verdict='ambiguous', reason='Fixture uncertainty.',
                                  rubric_ack=True, expected_revision=revision, decision_id=uuid.uuid4().hex, studies_root=self.studies)
        app = self.app()
        self.attach(app)
        self.submit(app)
        self.assertFalse(app.exception)
        self.assertTrue(any('revision 4' in e.value for e in app.success))
        self.assertFalse(app.text_area)
        self.assertTrue(any('4 of 4' in e.value for e in app.markdown))

    def test_discovery_does_not_follow_symlinks_or_hide_missing_roots(self):
        from signallab.synthetic_ui import discover_local
        empty = self.folder / 'absent'
        self.assertEqual(discover_local(empty, empty), ([], [], False))
        link_id = 'e'*32
        (self.pilots / link_id).symlink_to(self.pilots / self.pilot, target_is_directory=True)
        _, entries, _ = discover_local(self.studies, self.pilots)
        self.assertEqual(next(e for e in entries if e['pilot_id'] == link_id)['status'], 'Unreadable')
        root_link = self.folder / 'root_link'
        root_link.symlink_to(self.pilots, target_is_directory=True)
        with self.assertRaises(OSError):
            discover_local(self.studies, root_link)

    def test_busy_review_read_preserves_pair_and_unsaved_form_without_writing(self):
        app = self.app()
        self.attach(app)
        next(r for r in app.radio if r.label == 'Human decision').set_value('Reject')
        app.text_area[0].set_value('Draft must survive a temporary review lock.')
        app.checkbox[0].check()
        with patch('signallab.synthetic.read_reviews', side_effect=synthetic.ReviewConflict('busy')):
            self.button(app, 'Save decision and next pending').click().run()
        self.assertFalse(app.exception)
        self.assertEqual(app.text_area[0].value, 'Draft must survive a temporary review lock.')
        self.assertTrue(self.button(app, 'Save decision and next pending').disabled)
        self.assertEqual(synthetic.read_reviews(self.pilots, self.pilot, studies_root=self.studies)['revision'], 0)
        self.button(app, 'Refresh local studies').click().run()
        self.assertEqual(app.text_area[0].value, 'Draft must survive a temporary review lock.')
        self.assertFalse(self.button(app, 'Save decision and next pending').disabled)

    def test_save_from_accepted_filter_switches_to_next_pending(self):
        app = self.app()
        self.attach(app)
        first = self.select(app, 'Candidate').value
        self.submit(app)
        next_pending = self.select(app, 'Candidate').value
        self.select(app, 'Review filter').set_value('Accepted').run()
        self.assertEqual(self.select(app, 'Candidate').value, first)
        self.assertFalse(any('Decision saved' in e.value for e in app.success))
        self.submit(app)
        self.assertFalse(app.exception)
        self.assertEqual(self.select(app, 'Review filter').value, 'Pending')
        self.assertEqual(self.select(app, 'Candidate').value, next_pending)
        self.assertTrue(any('revision 2' in e.value for e in app.success))
