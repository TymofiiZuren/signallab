"""Qualified handoff contracts. All text, asset hashes and resource values are authored fixtures."""
import contextlib
import copy
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
import uuid
from unittest.mock import patch

import pandas as pd
from pandas.testing import assert_frame_equal

from signallab import study, synthetic, synthetic_cli as cli
from signallab.core import review_id

ROOT = Path(__file__).resolve().parents[1]
ASSETS = {name: 'a'*64 for name in ('config.json', 'tokenizer.json', 'tokenizer_config.json', 'model.safetensors')}
RESOURCES = dict(elapsed_seconds=1., peak_rss_bytes=1000, samples=4, method='macos-ps-rss-kib-v1')
EXECUTION = dict(preparation_id='a'*32, prepared_digest='b'*64, environment=dict(
    python='authored-fixture', dependencies={n:'fixture' for n in ('torch','transformers','tokenizers','huggingface-hub')},
    source_hashes={n:'c'*64 for n in ('synthetic_cli.py','synthetic_worker.py')}))


class HandoffTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scratch = Path(tempfile.mkdtemp(prefix='handoff-contract-', dir=ROOT/'.scratch'))
        cls.frame = pd.DataFrame([{'text':f'Authored source class {label} item {i}: '+('disappointing quality.' if label==0 else 'excellent quality.'), 'label':label}
                                  for label in (0,1) for i in range(60)])
        cls.base = cls.scratch/'base'
        cls.protocol_id, cls.pilot_id = cls.make_fixture(cls.base)

    @classmethod
    def make_fixture(cls, folder, *, max_parents=30, execution=True, resources=None, accept=None, pending=0, state='complete', output_text=None):
        protocol = study.build_protocol(cls.frame, cls.frame.assign(text='Reserved final fixture '+cls.frame.text),
                                        source='Authored contract development', final_source='Authored reserved fixture',
                                        max_parents=max_parents, attest_final_reserved=True)
        study.save_protocol(protocol, folder/'studies')
        pilot = synthetic.create_pilot(protocol['protocol_id'], cls.frame, ASSETS, root=folder/'pilots',
                                      studies_root=folder/'studies', execution=EXECUTION if execution else None)
        with synthetic.generation(folder/'pilots', pilot, studies_root=folder/'studies') as writer:
            for i,parent in enumerate(protocol['selection']['parent_ids']):
                writer.start(parent, 20)
                writer.generated(parent, output_text(i,parent) if output_text else f'Authored candidate {i}: independent fixture output.', 12, True)
            writer.finish(state, resources or RESOURCES, worker_reaped=True)
        current = synthetic.read_pilot(folder/'pilots', pilot, studies_root=folder/'studies')
        for revision,(cid,candidate) in enumerate(list(current['candidates'].items())[:max_parents-pending]):
            verdict = 'accept' if accept is None or accept(revision,candidate) else 'reject'
            synthetic.save_review(folder/'pilots',pilot,cls.frame,cid,verdict=verdict,reason='Authored rubric decision, not real model evidence.',
                                  rubric_ack=True,expected_revision=revision,decision_id=uuid.uuid4().hex,studies_root=folder/'studies')
        return protocol['protocol_id'],pilot

    def setUp(self):
        self.folder = self.scratch/uuid.uuid4().hex
        shutil.copytree(self.base,self.folder)
        self.studies,self.root = self.folder/'studies', self.folder/'pilots'

    def seal(self, **kwargs):
        return synthetic.seal_pilot(self.root,self.pilot_id,self.frame,expected_revision=kwargs.pop('revision',30),
                                    studies_root=self.studies,**kwargs)

    def status(self):
        return synthetic.read_handoff(self.root,self.pilot_id,studies_root=self.studies)

    def bundle(self, seal_id, frame=None):
        return synthetic.build_training_bundle(self.root,self.pilot_id,seal_id,self.frame if frame is None else frame,studies_root=self.studies)

    def edit(self, *, verdict='accept', reason='Updated authored review.'):
        current=synthetic.read_pilot(self.root,self.pilot_id,studies_root=self.studies)
        cid=next(iter(current['candidates']))
        revision=synthetic.read_reviews(self.root,self.pilot_id,studies_root=self.studies)['revision']
        synthetic.save_review(self.root,self.pilot_id,self.frame,cid,verdict=verdict,reason=reason,rubric_ack=True,
                              expected_revision=revision,decision_id=uuid.uuid4().hex,studies_root=self.studies)

    def test_seal_and_bundle_preserve_exact_partitions_lineage_and_independent_copies(self):
        seal=self.seal()
        self.assertEqual(seal['review_revision'],30)
        self.assertEqual(seal['counts']['accepted_eligible'],30)
        self.assertTrue(self.status()['qualified'])
        with patch('signallab.core.make_splits',side_effect=AssertionError('resplit attempted')):
            bundle=self.bundle(seal['seal_id'])
        protocol=study.read_protocol(self.studies,self.protocol_id)
        original=study.attach_development(protocol,self.frame)
        for name in ('train','validation','test'):
            assert_frame_equal(getattr(bundle.splits,name),getattr(original,name))
        self.assertEqual(len(bundle.training),len(original.train)+30)
        self.assertEqual(len(bundle.lineage),len(bundle.training))
        descendants=bundle.lineage[bundle.lineage.source_kind=='synthetic']
        self.assertEqual(descendants.parent_id.tolist(),[c['parent_id'] for c in seal['accepted']])
        self.assertEqual(descendants.candidate_id.tolist(),[c['candidate_id'] for c in seal['accepted']])
        self.assertTrue(set(descendants.parent_id)<=set(protocol['development']['split_ids']['train']))
        self.assertEqual(set(descendants.parent_id)&set(protocol['development']['split_ids']['validation']),set())
        bundle.training.loc[0,'text']='changed returned frame'
        self.assertNotEqual(bundle.splits.train.iloc[0].text,'changed returned frame')
        assert_frame_equal(self.bundle(seal['seal_id']).splits.train,original.train)

    def test_twenty_nine_accepts_fails_and_no_seal_is_published(self):
        self.edit(verdict='reject')
        status=self.status()
        self.assertFalse(status['qualified'])
        self.assertEqual(status['reviews']['accepted_eligible'],29)
        self.assertIn('accepted_min',status['unmet_gates'])
        with self.assertRaisesRegex(ValueError,'qualif'):
            self.seal(revision=31)
        self.assertFalse(list((self.root/self.pilot_id).glob('seal-*.json')))

    def test_per_class_gate_is_independent_of_total_acceptance(self):
        folder=self.scratch/uuid.uuid4().hex
        self.protocol_id,self.pilot_id=self.make_fixture(folder,max_parents=50,
            accept=lambda i,c: (c['label']==0 and i//2<9) or (c['label']==1 and i//2<21))
        self.root,self.studies=folder/'pilots',folder/'studies'
        status=self.status()
        self.assertEqual(status['reviews']['accepted_eligible'],30)
        self.assertEqual(status['reviews']['negative_eligible'],9)
        self.assertIn('accepted_per_class_min',status['unmet_gates'])
        self.assertNotIn('accepted_min',status['unmet_gates'])

    def test_all_generated_must_be_reviewed_even_if_excluded_or_low_yield(self):
        folder=self.scratch/uuid.uuid4().hex
        self.protocol_id,self.pilot_id=self.make_fixture(folder,max_parents=32,pending=1,
            output_text=lambda i,p: ('Reserved final fixture '+self.frame.iloc[0].text) if i==31 else f'Pending-gate fixture {i}')
        self.root,self.studies=folder/'pilots',folder/'studies'
        self.assertEqual(self.status()['reviews']['accepted_eligible'],31)
        self.assertIn('all_generated_reviewed',self.status()['unmet_gates'])
        with self.assertRaises(ValueError): self.seal(revision=31)

    def test_resources_generation_and_execution_provenance_gate_qualification(self):
        for options,gate in [({'execution':False},'execution_provenance'),
                             ({'resources':dict(elapsed_seconds=1.,peak_rss_bytes=None,samples=0,method='unavailable'),'state':'failed'},'resources_pass'),
                             ({'state':'interrupted'},'generation_complete')]:
            with self.subTest(gate=gate):
                folder=self.scratch/uuid.uuid4().hex
                _,pid=self.make_fixture(folder,**options)
                status=synthetic.read_handoff(folder/'pilots',pid,studies_root=folder/'studies')
                self.assertIn(gate,status['unmet_gates'])
                with self.assertRaises(ValueError):
                    synthetic.seal_pilot(folder/'pilots',pid,self.frame,expected_revision=30,studies_root=folder/'studies')

    def test_review_edit_makes_old_seal_historical_and_consumer_rejects_it(self):
        first=self.seal()
        path=self.root/self.pilot_id/f"seal-{first['seal_id']}.json"
        before=path.read_bytes()
        self.edit()
        historical=synthetic.read_seal(self.root,self.pilot_id,first['seal_id'],studies_root=self.studies)
        self.assertFalse(historical['current'])
        self.assertFalse(self.status()['seals'][0]['current'])
        with self.assertRaisesRegex(ValueError,'historical|current|stale'): self.bundle(first['seal_id'])
        second=self.seal(revision=31)
        self.assertNotEqual(first['seal_id'],second['seal_id'])
        self.assertEqual(self.bundle(second['seal_id']).review_revision,31)
        self.assertEqual(before,path.read_bytes())

    def test_stale_revision_and_reused_seal_identity_cannot_bypass_currency(self):
        seal_id=uuid.uuid4().hex
        first=self.seal(seal_id=seal_id)
        self.assertEqual(first,self.seal(seal_id=seal_id))
        for invalid in (True,0,31):
            with self.assertRaises(ValueError): self.seal(revision=invalid)
        self.edit()
        with self.assertRaises(synthetic.ReviewConflict): self.seal(seal_id=seal_id,revision=31)
        with self.assertRaises(synthetic.ReviewConflict): self.seal(revision=30)

    def test_wrong_source_and_tampered_evidence_fail_closed(self):
        seal=self.seal()
        changed=self.frame.copy()
        changed.loc[0,'text']+=' changed'
        with self.assertRaises(ValueError): self.bundle(seal['seal_id'],changed)
        folder=self.root/self.pilot_id
        targets=[folder/'terminal.json',folder/'pilot.json',folder/'review-0030.json',
                 folder/f"seal-{seal['seal_id']}.json",self.studies/self.protocol_id/'protocol.json']
        targets.append(next(folder.glob('attempt-*.result.json')))
        for path in targets:
            with self.subTest(file=path.name):
                before=path.read_bytes()
                path.write_text('{}')
                with self.assertRaises(ValueError): self.bundle(seal['seal_id'])
                path.write_bytes(before)

    def test_seal_publication_failure_has_idempotent_retry_and_no_overwrite(self):
        seal_id=uuid.uuid4().hex
        with patch('signallab.synthetic.os.fsync',side_effect=[None,OSError('authored directory sync failure')]):
            with self.assertRaises(OSError): self.seal(seal_id=seal_id)
        path=self.root/self.pilot_id/f'seal-{seal_id}.json'
        before=path.read_bytes()
        self.assertEqual(self.seal(seal_id=seal_id)['seal_id'],seal_id)
        self.assertEqual(before,path.read_bytes())
        self.assertEqual(len(list((self.root/self.pilot_id).glob('seal-*.json'))),1)

    def test_bounded_seals_reject_unsafe_ids_symlinks_and_oversized_json(self):
        self.assertEqual(synthetic.MAX_SEALS,200)
        with patch('signallab.synthetic.MAX_SEALS',2):
            self.seal(); self.seal()
            with self.assertRaises(ValueError): self.seal()
        for value in ('../escape','x'*32,True):
            with self.assertRaises(ValueError): self.bundle(value)
        first=self.status()['seals'][0]['seal_id']
        path=self.root/self.pilot_id/('seal-'+uuid.uuid4().hex+'.json')
        path.symlink_to(self.root/self.pilot_id/f'seal-{first}.json')
        with self.assertRaises(ValueError): self.bundle(first)

    def test_actual_process_review_lock_excludes_seal_and_review_writers(self):
        self.frame.to_csv(self.folder/'development.csv',index=False)
        script='''import sys,pandas as pd
from signallab import synthetic
root,studies,pilot,csv=sys.argv[1:]
try: synthetic.seal_pilot(root,pilot,pd.read_csv(csv),expected_revision=30,studies_root=studies)
except synthetic.ReviewConflict: print('blocked'); sys.exit(0)
sys.exit(2)
'''
        with synthetic._directory(self.root,self.pilot_id) as fd, synthetic._lock(fd,'review.lock'):
            result=subprocess.run([sys.executable,'-c',script,str(self.root),str(self.studies),self.pilot_id,str(self.folder/'development.csv')],cwd=ROOT,capture_output=True,text=True,timeout=20)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertEqual(result.stdout.strip(),'blocked')
            self.assertFalse(list((self.root/self.pilot_id).glob('seal-*.json')))
        self.assertEqual(self.seal()['review_revision'],30)

    def test_cli_seal_and_inspect_bundle_are_text_free_and_do_not_load_models(self):
        csv=self.folder/'development.csv'; self.frame.to_csv(csv,index=False)
        roots=['--studies-root',str(self.studies),'--pilots-root',str(self.root)]
        seal_id=uuid.uuid4().hex
        out=io.StringIO()
        with patch('signallab.synthetic_cli._download_assets',side_effect=AssertionError('network called')),contextlib.redirect_stdout(out):
            code=cli.main([*roots,'seal','--pilot',self.pilot_id,'--csv',str(csv),'--expected-revision','30','--seal-id',seal_id,'--json'])
        self.assertEqual(code,0)
        self.assertEqual(json.loads(out.getvalue())['seal_id'],seal_id)
        out=io.StringIO()
        with contextlib.redirect_stdout(out):
            code=cli.main([*roots,'inspect-bundle','--pilot',self.pilot_id,'--seal',seal_id,'--csv',str(csv),'--json'])
        self.assertEqual(code,0)
        record=json.loads(out.getvalue())
        self.assertTrue(record['no_training_has_run'])
        self.assertEqual(record['accepted_synthetic'],30)
        self.assertEqual(record['training_rows'],record['original_partitions']['train']+30)
        self.assertNotIn('Authored',out.getvalue())
        self.edit()
        with contextlib.redirect_stderr(io.StringIO()),contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(cli.main([*roots,'inspect-bundle','--pilot',self.pilot_id,'--seal',seal_id,'--csv',str(csv),'--json']),1)

    def test_inspected_bundle_can_feed_existing_baseline_without_resplitting(self):
        from signallab.core import train_baseline, evaluate
        bundle=self.bundle(self.seal()['seal_id'])
        model=train_baseline(bundle.training)
        report=evaluate(model,bundle.splits.validation)
        self.assertEqual(report['n'],len(bundle.splits.validation))
        self.assertEqual(len(bundle.splits.test),24)

    def test_accepted_final_collision_never_enters_the_seal_or_training(self):
        folder=self.scratch/uuid.uuid4().hex
        final_text='Reserved final fixture '+self.frame.iloc[0].text
        self.protocol_id,self.pilot_id=self.make_fixture(folder,max_parents=32,
            output_text=lambda i,parent: final_text if i==0 else f'Unique authored candidate {i}')
        self.root,self.studies=folder/'pilots',folder/'studies'
        status=self.status()
        self.assertTrue(status['qualified'])
        self.assertEqual(status['reviews']['reviewed'],32)
        self.assertEqual(status['reviews']['accepted_eligible'],31)
        seal=self.seal(revision=32)
        bundle=self.bundle(seal['seal_id'])
        self.assertNotIn(study.text_fingerprint(final_text),set(bundle.training.text.map(study.text_fingerprint)))
        excluded=next(iter(synthetic.read_pilot(self.root,self.pilot_id,studies_root=self.studies)['candidates']))
        self.assertNotIn(excluded,bundle.lineage.candidate_id.tolist())

    def test_oversized_or_unsupported_seal_is_preserved_and_rejected(self):
        seal=self.seal()
        path=self.root/self.pilot_id/f"seal-{seal['seal_id']}.json"
        original=path.read_bytes()
        path.write_bytes(b' '*(synthetic.MAX_BYTES+1))
        with self.assertRaises(ValueError): self.bundle(seal['seal_id'])
        self.assertEqual(path.stat().st_size,synthetic.MAX_BYTES+1)
        record=json.loads(original)
        record['schema_version']=2
        record['content_digest']=synthetic._digest({k:v for k,v in record.items() if k!='content_digest'})
        path.write_text(json.dumps(record))
        with self.assertRaises(ValueError): self.bundle(seal['seal_id'])
        self.assertEqual(json.loads(path.read_text())['schema_version'],2)

    def test_ui_seals_exact_source_and_labels_historical_evidence_after_edit(self):
        from streamlit.testing.v1 import AppTest
        script=('from signallab.synthetic_ui import render_synthetic_lab\n'
                f'render_synthetic_lab(studies_root={str(self.studies)!r},root={str(self.root)!r})')
        with patch('signallab.synthetic_ui.demo_reviews',return_value=self.frame):
            app=AppTest.from_string(script,default_timeout=30).run()
            next(b for b in app.button if b.label=='Refresh local studies').click().run()
            self.assertFalse(app.exception)
            self.assertTrue(any('Feasibility: gates met' in e.value for e in app.markdown))
            next(s for s in app.selectbox if s.label=='Development source').set_value('Authored demo').run()
            next(b for b in app.button if b.label=='Seal accepted set').click().run()
            self.assertFalse(app.exception)
            self.assertTrue(any('Seal saved' in e.value for e in app.success))
            self.assertTrue(self.status()['seals'][0]['current'])
            next(s for s in app.selectbox if s.label=='Review filter').set_value('All').run()
            self.assertTrue(any('historical' in e.value for e in app.warning))
            self.edit()
            app.run()
            self.assertFalse(app.exception)
            self.assertTrue(any('Seal currency: historical' in e.value for e in app.markdown))
            self.assertFalse(self.status()['seals'][0]['current'])

    def test_ui_does_not_silently_seal_a_newer_unseen_review_revision(self):
        from streamlit.testing.v1 import AppTest
        script=('from signallab.synthetic_ui import render_synthetic_lab\n'
                f'render_synthetic_lab(studies_root={str(self.studies)!r},root={str(self.root)!r})')
        with patch('signallab.synthetic_ui.demo_reviews',return_value=self.frame):
            app=AppTest.from_string(script,default_timeout=30).run()
            next(b for b in app.button if b.label=='Refresh local studies').click().run()
            next(s for s in app.selectbox if s.label=='Development source').set_value('Authored demo').run()
            self.edit()
            next(b for b in app.button if b.label=='Seal accepted set').click().run()
            self.assertFalse(app.exception)
            self.assertFalse(self.status()['seals'])
            self.assertTrue(any('Review changed' in e.value for e in app.warning))
            next(b for b in app.button if b.label=='Reload seal context').click().run()
            next(b for b in app.button if b.label=='Seal accepted set').click().run()
            self.assertTrue(self.status()['seals'][0]['current'])

    def test_seal_publication_holds_review_lock_against_an_actual_reviewer_process(self):
        self.frame.to_csv(self.folder/'development.csv',index=False)
        real_publish=synthetic._publish
        script='''import sys,uuid,pandas as pd
from signallab import synthetic
root,studies,pilot,csv=sys.argv[1:]
cid=next(iter(synthetic.read_pilot(root,pilot,studies_root=studies)['candidates']))
try: synthetic.save_review(root,pilot,pd.read_csv(csv),cid,verdict='reject',reason='Racing fixture edit.',rubric_ack=True,expected_revision=30,decision_id=uuid.uuid4().hex,studies_root=studies)
except synthetic.ReviewConflict: print('blocked'); sys.exit(0)
sys.exit(2)
'''
        def publish(fd,name,record,**kwargs):
            real_publish(fd,name,record,**kwargs)
            if name.startswith('seal-'):
                result=subprocess.run([sys.executable,'-c',script,str(self.root),str(self.studies),self.pilot_id,str(self.folder/'development.csv')],cwd=ROOT,capture_output=True,text=True,timeout=20)
                self.assertEqual(result.returncode,0,result.stderr)
                self.assertEqual(result.stdout.strip(),'blocked')
        with patch('signallab.synthetic._publish',side_effect=publish): seal=self.seal()
        self.assertEqual(self.bundle(seal['seal_id']).review_revision,30)
        self.edit(verdict='reject')
        with self.assertRaises(ValueError): self.bundle(seal['seal_id'])

    def test_ui_uncertain_seal_retry_uses_same_identity_without_duplicate_publication(self):
        from streamlit.testing.v1 import AppTest
        script=('from signallab.synthetic_ui import render_synthetic_lab\n'
                f'render_synthetic_lab(studies_root={str(self.studies)!r},root={str(self.root)!r})')
        real_seal=synthetic.seal_pilot
        def committed_then_failed(*args,**kwargs):
            real_seal(*args,**kwargs)
            raise OSError('Authored lost seal acknowledgement')
        with patch('signallab.synthetic_ui.demo_reviews',return_value=self.frame):
            app=AppTest.from_string(script,default_timeout=30).run()
            next(b for b in app.button if b.label=='Refresh local studies').click().run()
            next(s for s in app.selectbox if s.label=='Development source').set_value('Authored demo').run()
            with patch('signallab.synthetic.seal_pilot',side_effect=committed_then_failed):
                next(b for b in app.button if b.label=='Seal accepted set').click().run()
            self.assertTrue(app.error)
            self.assertFalse(any('Seal saved' in e.value for e in app.success))
            app.run()
            next(b for b in app.button if b.label=='Seal accepted set').click().run()
            self.assertTrue(any('Seal saved' in e.value for e in app.success))
            self.assertEqual(len(self.status()['seals']),1)

    def test_all_rejected_is_an_inspectable_negative_result_not_an_operational_error(self):
        folder=self.scratch/uuid.uuid4().hex
        _,pid=self.make_fixture(folder,accept=lambda i,c:False)
        status=synthetic.read_handoff(folder/'pilots',pid,studies_root=folder/'studies')
        self.assertFalse(status['qualified'])
        self.assertEqual(status['generation_state'],'complete')
        self.assertEqual(status['reviews']['pending'],0)
        self.assertEqual(status['reviews']['reject'],30)
        with contextlib.redirect_stdout(io.StringIO()) as output:
            code=cli.main(['--studies-root',str(folder/'studies'),'--pilots-root',str(folder/'pilots'),'inspect','--pilot',pid,'--json'])
        self.assertEqual(code,0)
        self.assertEqual(json.loads(output.getvalue())['reviews']['accepted_eligible'],0)
