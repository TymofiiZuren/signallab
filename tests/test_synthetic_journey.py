"""Main-app to CLI synthetic workflow, with authored evidence and no network/model preparation."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from streamlit.testing.v1 import AppTest
from signallab import study, synthetic, synthetic_cli as cli
import test_synthetic_handoff as handoff_fixtures

ROOT=Path(__file__).resolve().parents[1]


class SyntheticJourneyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Reuse the owning contract fixture builder, never simulate a real study.
        cls.scratch=Path(tempfile.mkdtemp(prefix='synthetic-journey-',dir=ROOT/'.scratch'))
        if not hasattr(handoff_fixtures.HandoffTests,'base'):
            handoff_fixtures.HandoffTests.setUpClass()
        cls.frame=handoff_fixtures.HandoffTests.frame

    def setUp(self):
        self.folder=Path(tempfile.mkdtemp(prefix='flow-',dir=self.scratch))
        self.studies,self.pilots=self.folder/'studies',self.folder/'pilots'
        self.patches=[patch('signallab.study.STUDIES_ROOT',self.studies),
                      patch('signallab.synthetic.SYNTHETIC_ROOT',self.pilots),
                      patch('signallab.synthetic_ui.demo_reviews',return_value=self.frame),
                      patch('signallab.synthetic_cli._download_assets',side_effect=AssertionError('Unexpected download')),
                      patch('signallab.synthetic_cli.generate_pilot',side_effect=AssertionError('Unexpected model execution'))]
        for item in self.patches: item.start(); self.addCleanup(item.stop)

    def fixture(self, *, pending=0):
        self.protocol,self.pilot=handoff_fixtures.HandoffTests.make_fixture(self.folder,pending=pending)

    @staticmethod
    def button(app,label): return next(b for b in app.button if b.label==label)

    @staticmethod
    def select(app,label): return next(s for s in app.selectbox if s.label==label)

    def app(self, *, attach=True):
        app=AppTest.from_file(str(ROOT/'app.py'),default_timeout=30).run()
        app.sidebar.selectbox[0].set_value('Upload CSV').run()
        app.sidebar.radio[0].set_value('Synthetic Review Lab').run()
        self.button(app,'Refresh local studies').click().run()
        self.assertFalse(app.exception)
        self.assertIn('Synthetic Review Lab',[e.value for e in app.title])
        if attach:
            self.select(app,'Development source').set_value('Authored demo').run()
        self.assertFalse(app.exception)
        return app

    def submit(self,app,verdict,reason):
        next(r for r in app.radio if r.label=='Human decision').set_value(verdict)
        next(t for t in app.text_area if t.label=='Reason for this decision').set_value(reason)
        next(c for c in app.checkbox if 'I checked meaning' in c.label).check()
        self.button(app,'Save decision and next pending').click().run()
        self.assertFalse(app.exception)

    def test_two_app_sessions_conflict_reload_seal_edit_and_cli_currency(self):
        self.fixture(pending=1)
        a,b=self.app(),self.app()
        self.assertEqual(self.select(a,'Candidate').value,self.select(b,'Candidate').value)
        self.submit(a,'Accept','Authored fixture: first session decision.')
        self.submit(b,'Reject','Authored fixture: stale second-session draft.')
        self.assertTrue(any('unsaved' in e.value.lower() for e in b.error))
        self.assertIn('Authored fixture: stale second-session draft.',[e.value for e in b.text])
        self.assertEqual(synthetic.read_reviews(self.pilots,self.pilot,studies_root=self.studies)['revision'],30)
        self.button(b,'Reload current decision').click().run()
        self.assertEqual(next(r for r in b.radio if r.label=='Human decision').value,'Accept')
        self.assertEqual(next(t for t in b.text_area if t.label=='Reason for this decision').value,'Authored fixture: first session decision.')
        self.button(a,'Reload seal context').click().run()
        self.button(a,'Seal accepted set').click().run()
        self.assertFalse(a.exception)
        first=next(s for s in synthetic.read_handoff(self.pilots,self.pilot,studies_root=self.studies)['seals'] if s['current'])
        original=(self.pilots/self.pilot/f"seal-{first['seal_id']}.json").read_bytes()
        self.submit(b,'Accept','Authored fixture: explicit second-session revision.')
        a.run()
        self.assertTrue(any('Seal currency: historical' in e.value for e in a.markdown))
        self.frame.to_csv(self.folder/'development.csv',index=False)
        roots=['--studies-root',str(self.studies),'--pilots-root',str(self.pilots)]
        with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(cli.main([*roots,'inspect-bundle','--pilot',self.pilot,'--seal',first['seal_id'],'--csv',str(self.folder/'development.csv'),'--json']),1)
        self.button(a,'Reload seal context').click().run()
        self.button(a,'Seal accepted set').click().run()
        self.assertFalse(a.exception)
        status=synthetic.read_handoff(self.pilots,self.pilot,studies_root=self.studies)
        self.assertEqual(sum(s['current'] for s in status['seals']),1)
        current=next(s for s in status['seals'] if s['current'])
        with contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(cli.main([*roots,'inspect-bundle','--pilot',self.pilot,'--seal',current['seal_id'],'--csv',str(self.folder/'development.csv'),'--json']),0)
        report=json.loads(output.getvalue())
        self.assertEqual(report['original_partitions'],{'train':72,'validation':24,'test':24})
        self.assertEqual(report['training_rows'],102)
        self.assertEqual(report['review_revision'],31)
        self.assertTrue(report['no_training_has_run'])
        self.assertNotIn('Authored',output.getvalue())
        self.assertEqual((self.pilots/self.pilot/f"seal-{first['seal_id']}.json").read_bytes(),original)

    def test_exact_attachment_loss_removes_mutation_but_preserves_numeric_evidence(self):
        self.fixture()
        app=self.app()
        self.button(app,'Seal accepted set').click().run()
        self.select(app,'Review filter').set_value('All').run()
        changed=self.frame.copy(); changed.loc[0,'text']+=' changed'
        with patch('signallab.synthetic_ui.demo_reviews',return_value=changed): app.run()
        self.assertFalse(app.exception)
        self.assertFalse(app.text_area)
        self.assertFalse(app.text)
        self.assertFalse(any(b.label=='Seal accepted set' and not b.disabled for b in app.button))
        self.assertTrue(any('Mismatch' in e.value for e in app.error))
        self.assertTrue(any('Seal currency: current' in e.value for e in app.markdown))
        app.sidebar.radio[0].set_value('Explore').run()
        app.sidebar.radio[0].set_value('Synthetic Review Lab').run()
        self.assertFalse(app.exception)
        self.assertFalse(app.text_area)
        self.assertEqual(self.select(app,'Development source').value,'Not attached')
        self.assertFalse(any('Seal saved' in e.value for e in app.success))

    def test_damaged_review_history_is_preserved_and_blocks_app_and_cli(self):
        self.fixture()
        app=self.app()
        self.button(app,'Seal accepted set').click().run()
        path=self.pilots/self.pilot/'review-0030.meta.json'
        damaged=b'{"schema_version":'
        path.write_bytes(damaged)
        app.run()
        self.assertFalse(app.exception)
        self.assertTrue(app.error)
        self.assertFalse(app.text_area)
        self.assertFalse(any('Seal saved' in e.value for e in app.success))
        with contextlib.redirect_stdout(io.StringIO()) as output,contextlib.redirect_stderr(io.StringIO()):
            code=cli.main(['--studies-root',str(self.studies),'--pilots-root',str(self.pilots),'inspect','--pilot',self.pilot,'--json'])
        self.assertEqual(code,1)
        self.assertEqual(output.getvalue(),'')
        self.assertEqual(path.read_bytes(),damaged)
        app.sidebar.radio[0].set_value('Explore').run()
        self.assertFalse(app.exception)
        self.assertIn('Bring your reviews',[e.value for e in app.title])
        app.sidebar.selectbox[0].set_value('Authored demo').run()
        self.assertFalse(app.exception)
        self.assertIn('Explore reviews',[e.value for e in app.title])

    def test_documented_consumer_recipe_executes_on_current_bundle_without_resplitting(self):
        import re
        self.fixture()
        seal=synthetic.seal_pilot(self.pilots,self.pilot,self.frame,expected_revision=30,studies_root=self.studies)
        csv=self.folder/'development.csv'; self.frame.to_csv(csv,index=False)
        doc=(ROOT/'docs/synthetic-experiment-walkthrough.md').read_text()
        snippet=re.search(r'```python\n(.*?)```',doc,re.S).group(1)
        replacements={'"artifacts/synthetic"':repr(str(self.pilots)), '"artifacts/studies"':repr(str(self.studies)),
                      '"PILOT_ID"':repr(self.pilot), '"SEAL_ID"':repr(seal['seal_id']), '"data/development.csv"':repr(str(csv))}
        for before,after in replacements.items(): snippet=snippet.replace(before,after)
        with patch('signallab.core.make_splits',side_effect=AssertionError('Documentation re-split the bundle')),contextlib.redirect_stdout(io.StringIO()) as output:
            exec(compile(snippet,'docs/synthetic-experiment-walkthrough.md','exec'),{})
        result=json.loads(output.getvalue())
        self.assertEqual(result['validation']['n'],24)
        self.assertEqual(result['training_rows'],102)
        self.assertEqual(result['seal_id'],seal['seal_id'])
        self.assertNotIn('Authored',output.getvalue())
        self.assertNotIn('test',result)
        self.assertIn('no augmentation benefit',result['interpretation'])
