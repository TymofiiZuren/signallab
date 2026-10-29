"""Synthetic evidence contracts use authored outputs, never a downloaded model."""
import copy
import json
import os
from pathlib import Path
import tempfile
import subprocess
import sys
import unittest
from unittest.mock import patch
import uuid

from signallab import study, synthetic
from signallab.core import review_id
from signallab.demo import demo_reviews

ROOT = Path(__file__).resolve().parents[1]
ASSETS = {name: 'a'*64 for name in ('config.json', 'tokenizer.json', 'tokenizer_config.json', 'model.safetensors')}
RESOURCES = {'elapsed_seconds': 1.0, 'peak_rss_bytes': 1000, 'samples': 4, 'method': 'macos-ps-rss-kib-v1'}


class SyntheticEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scratch = Path(tempfile.mkdtemp(prefix='synthetic-contract-', dir=ROOT / '.scratch'))
        cls.frame = demo_reviews()
        cls.final = cls.frame.assign(text='reserved fixture ' + cls.frame.text)

    def setUp(self):
        self.folder = self.scratch / uuid.uuid4().hex
        self.studies = self.folder / 'studies'
        self.root = self.folder / 'synthetic'
        self.protocol = study.build_protocol(self.frame, self.final, source='authored development',
                                             final_source='authored final fixture', attest_final_reserved=True,
                                             max_parents=4)
        study.save_protocol(self.protocol, self.studies)
        self.pilot_id = synthetic.create_pilot(self.protocol['protocol_id'], self.frame, ASSETS,
                                               root=self.root, studies_root=self.studies)
        self.parents = self.protocol['selection']['parent_ids']

    def writer(self):
        return synthetic.generation(self.root, self.pilot_id, studies_root=self.studies)

    def read(self):
        return synthetic.read_pilot(self.root, self.pilot_id, studies_root=self.studies)

    def complete(self, texts=None):
        texts = texts or [f'Fresh paraphrase fixture {i}' for i in range(4)]
        with self.writer() as writer:
            for parent, text in zip(self.parents, texts):
                writer.start(parent, 30)
                writer.generated(parent, text, 12, True)
            writer.finish('complete', RESOURCES, worker_reaped=True)
        return self.read()

    def review(self, candidate, *, expected=0, decision_id=None, verdict='accept', reason='Preserves the authored meaning.'):
        return synthetic.save_review(self.root, self.pilot_id, self.frame, candidate,
                                     verdict=verdict, reason=reason, rubric_ack=True,
                                     expected_revision=expected, decision_id=decision_id or uuid.uuid4().hex,
                                     studies_root=self.studies)

    def test_attempt_order_intent_and_terminal_are_immutable(self):
        with self.writer() as writer:
            with self.assertRaises(ValueError):
                writer.generated(self.parents[0], 'no intent', 3, True)
            with self.assertRaises(ValueError):
                writer.start(self.parents[1], 30)
            writer.start(self.parents[0], 30)
            with self.assertRaises(ValueError):
                writer.start(self.parents[0], 30)
            writer.generated(self.parents[0], 'A new literal candidate.', 10, True)
            writer.skip(self.parents[1], 129)
            writer.start(self.parents[2], 128)
            writer.failed(self.parents[2], 'worker_error')
            with self.assertRaises(ValueError):
                writer.finish('complete', RESOURCES, worker_reaped=True)
            writer.finish('failed', RESOURCES, worker_reaped=True)
            with self.assertRaises(ValueError):
                writer.start(self.parents[3], 12)
        result = self.read()
        self.assertEqual(result['generation_state'], 'failed')
        self.assertEqual([a['state'] for a in result['attempts']], ['generated', 'skipped_length', 'failed', 'unattempted'])
        before = (self.root / self.pilot_id / 'terminal.json').read_bytes()
        with self.writer() as writer:
            with self.assertRaises(ValueError):
                writer.finish('failed', RESOURCES, worker_reaped=True)
        self.assertEqual((self.root / self.pilot_id / 'terminal.json').read_bytes(), before)

    def test_auto_exclusions_cover_all_originals_and_final_without_labels(self):
        lookup = {review_id(r.text,r.label): r.text for r in self.frame.itertuples()}
        result = self.complete(['   ', lookup[self.parents[1]].upper(), self.final.iloc[0].text, self.frame.iloc[0].text])
        reasons = [c['exclusions'] for c in result['candidates'].values()]
        self.assertIn('blank', reasons[0])
        self.assertIn('unchanged_parent', reasons[1])
        self.assertIn('original_collision', reasons[1])
        self.assertIn('final_collision', reasons[2])
        self.assertIn('original_collision', reasons[3])
        first = next(iter(result['candidates']))
        self.review(first)
        current = synthetic.read_reviews(self.root, self.pilot_id, studies_root=self.studies)
        self.assertEqual(current['decisions'][first]['verdict'], 'accept')
        self.assertTrue(self.read()['candidates'][first]['exclusions'])

    def test_duplicate_owner_is_frozen_before_human_decisions(self):
        result = self.complete(['Unique paraphrase.', ' UNIQUE paraphrase. ', 'Third output', 'Fourth output'])
        candidates = list(result['candidates'])
        self.assertEqual(result['candidates'][candidates[0]]['exclusions'], [])
        self.assertIn('candidate_collision', result['candidates'][candidates[1]]['exclusions'])
        self.review(candidates[0], verdict='reject', reason='Human detected meaning change.')
        self.assertIn('candidate_collision', self.read()['candidates'][candidates[1]]['exclusions'])

    def test_review_requires_terminal_exact_attachment_reason_and_ack(self):
        with self.assertRaises(ValueError):
            self.review('a'*64)
        result = self.complete()
        candidate = next(iter(result['candidates']))
        changed = self.frame.copy()
        changed.loc[0,'text'] += ' changed'
        with self.assertRaises(ValueError):
            synthetic.save_review(self.root,self.pilot_id,changed,candidate,verdict='accept',reason='Checked',
                                  rubric_ack=True,expected_revision=0,decision_id=uuid.uuid4().hex,studies_root=self.studies)
        for changes in ({'reason':''}, {'reason':'x'*2001}, {'rubric_ack':False}, {'rubric_ack':1},
                        {'verdict':'positive'}, {'expected_revision':True}, {'decision_id':'../bad'}):
            args=dict(verdict='accept',reason='Checked',rubric_ack=True,expected_revision=0,decision_id=uuid.uuid4().hex)
            args.update(changes)
            with self.assertRaises(ValueError):
                synthetic.save_review(self.root,self.pilot_id,self.frame,candidate,studies_root=self.studies,**args)
        self.assertEqual(synthetic.read_reviews(self.root,self.pilot_id,studies_root=self.studies)['revision'],0)

    def test_review_cas_idempotence_payload_binding_and_history(self):
        result = self.complete()
        candidate = next(iter(result['candidates']))
        decision = uuid.uuid4().hex
        self.assertEqual(self.review(candidate, decision_id=decision),1)
        first = (self.root/self.pilot_id/'review-0001.json').read_bytes()
        self.assertEqual(self.review(candidate, decision_id=decision),1)
        with self.assertRaises(synthetic.ReviewConflict):
            self.review(candidate, decision_id=decision, reason='Different request under same ID.')
        with self.assertRaises(synthetic.ReviewConflict):
            self.review(candidate, expected=0)
        self.assertEqual(self.review(candidate, expected=1, verdict='ambiguous',reason='Meaning uncertain.'),2)
        self.assertEqual((self.root/self.pilot_id/'review-0001.json').read_bytes(),first)
        current=synthetic.read_reviews(self.root,self.pilot_id,studies_root=self.studies)
        self.assertEqual(current['revision'],2)
        self.assertEqual(current['decisions'][candidate]['verdict'],'ambiguous')
        self.assertEqual(len(current['requests']),2)
        self.assertEqual(self.review(candidate, decision_id=decision),1)

    def test_interrupted_intent_and_unknown_resources_remain_inspectable(self):
        with self.writer() as writer:
            writer.start(self.parents[0],30)
        result=self.read()
        self.assertEqual(result['generation_state'],'incomplete')
        self.assertEqual(result['attempts'][0]['state'],'started')
        with self.writer() as writer:
            writer.finish('interrupted',dict(RESOURCES,peak_rss_bytes=None,samples=0,method='unavailable'),worker_reaped=True)
        result=self.read()
        self.assertEqual(result['attempts'][0]['state'],'interrupted')
        self.assertEqual(result['terminal']['resource_status'],'unknown')
        self.assertEqual(len(result['candidates']),0)

    def test_abandoned_generation_only_allows_interruption_finalization(self):
        with self.writer() as writer:
            writer.start(self.parents[0],30)
        with self.writer() as recovered:
            with self.assertRaises(ValueError):
                recovered.generated(self.parents[0],'late output',12,True)
            with self.assertRaises(ValueError):
                recovered.finish('failed',RESOURCES,worker_reaped=True)
            recovered.finish('interrupted',RESOURCES,worker_reaped=True)
        with self.assertRaises(ValueError):
            writer.start(self.parents[1],30)

    def test_failure_stops_remaining_parent_writes(self):
        with self.writer() as writer:
            writer.start(self.parents[0],30)
            writer.failed(self.parents[0],'worker_error')
            with self.assertRaises(ValueError):
                writer.start(self.parents[1],30)
            writer.finish('failed',RESOURCES,worker_reaped=True)

    def test_generation_lock_and_running_status(self):
        with self.writer():
            self.assertEqual(self.read()['generation_state'],'running')
            with self.assertRaises(ValueError):
                with self.writer():
                    pass
        self.assertEqual(self.read()['generation_state'],'incomplete')

    def test_two_processes_cannot_overwrite_the_same_review_revision(self):
        candidate=next(iter(self.complete()['candidates']))
        script='''import json,sys,pandas as pd
from signallab.synthetic import save_review,ReviewConflict
data=json.load(sys.stdin)
try:
 save_review(data['root'],data['pilot'],pd.DataFrame(data['frame']),data['candidate'],
  verdict='accept',reason='Concurrent authored review',rubric_ack=True,
  expected_revision=0,decision_id=data['decision'],studies_root=data['studies'])
except ReviewConflict:
 sys.exit(3)
'''
        children=[]
        for _ in range(2):
            child=subprocess.Popen([sys.executable,'-c',script],stdin=subprocess.PIPE,stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE,text=True,cwd=ROOT)
            child.stdin.write(json.dumps(dict(root=str(self.root),pilot=self.pilot_id,frame=self.frame.to_dict('records'),
                                             candidate=candidate,decision=uuid.uuid4().hex,studies=str(self.studies))))
            child.stdin.close()
            children.append(child)
        codes=[]
        for child in children:
            child.wait(timeout=20)
            codes.append(child.returncode)
            self.assertEqual(child.stderr.read(),'')
            child.stderr.close()
            child.stdout.close()
        self.assertEqual(sorted(codes),[0,3])
        self.assertEqual(synthetic.read_reviews(self.root,self.pilot_id,studies_root=self.studies)['revision'],1)

    def test_request_identity_binds_reason_whitespace_and_expected_revision(self):
        candidate=next(iter(self.complete()['candidates']))
        request=uuid.uuid4().hex
        self.review(candidate,decision_id=request,reason='Exact reason')
        for changes in ({'reason':'Exact reason '},{'expected':1}):
            args=dict(decision_id=request,reason='Exact reason')
            args.update(changes)
            with self.assertRaises(synthetic.ReviewConflict):
                self.review(candidate,**args)

    def test_limits_eos_and_literal_text_are_enforced_before_publication(self):
        with self.writer() as writer:
            for tokens in (True,0,129,1.5):
                with self.assertRaises(ValueError):
                    writer.start(self.parents[0],tokens)
            writer.start(self.parents[0],128)
            for text,tokens,eos in (('x'*20001,12,True),('x',129,True),('x',True,True),('x',0,True),('x',12,1)):
                with self.assertRaises(ValueError):
                    writer.generated(self.parents[0],text,tokens,eos)
            literal='<script>alert("fixture")</script> **literal**'
            writer.generated(self.parents[0],literal,128,False)
            for parent in self.parents[1:]:
                writer.skip(parent,129)
            writer.finish('complete',RESOURCES,worker_reaped=True)
        candidate=next(iter(self.read()['candidates'].values()))
        self.assertEqual(candidate['text'],literal)
        self.assertEqual(candidate['exclusions'],['incomplete_output'])

    def test_review_publication_failure_preserves_pending_and_old_evidence(self):
        candidate=next(iter(self.complete()['candidates']))
        with patch('signallab.synthetic.os.fsync',side_effect=OSError('injected disk error')):
            with self.assertRaises(OSError):
                self.review(candidate)
        self.assertEqual(synthetic.read_reviews(self.root,self.pilot_id,studies_root=self.studies)['revision'],0)
        self.assertEqual(self.review(candidate),1)
        first=(self.root/self.pilot_id/'review-0001.json').read_bytes()
        publish=synthetic._publish
        def fail_commit(fd,name,value,**kwargs):
            if name=='review-0002.meta.json':
                raise OSError('injected commit publication error')
            return publish(fd,name,value,**kwargs)
        with patch('signallab.synthetic._publish',side_effect=fail_commit):
            with self.assertRaises(OSError):
                self.review(candidate,expected=1,verdict='reject')
        self.assertEqual((self.root/self.pilot_id/'review-0001.json').read_bytes(),first)
        with self.assertRaisesRegex(ValueError,'incomplete'):
            synthetic.read_reviews(self.root,self.pilot_id,studies_root=self.studies)

    def test_resource_thresholds_unknowns_and_huge_numbers_fail_safely(self):
        with self.writer() as writer:
            for values in (dict(RESOURCES,elapsed_seconds=True),dict(RESOURCES,elapsed_seconds=10**1000),
                           dict(RESOURCES,peak_rss_bytes=None),dict(RESOURCES,samples=0)):
                with self.assertRaises(ValueError):
                    writer.finish('interrupted',values,worker_reaped=True)
            with self.assertRaises(ValueError):
                writer.finish('interrupted',RESOURCES,worker_reaped=False)
            writer.finish('stopped_budget',dict(RESOURCES,peak_rss_bytes=4*1024**3),worker_reaped=True)
        self.assertEqual(self.read()['terminal']['resource_status'],'fail')

    def test_running_reader_never_validates_a_racing_multi_file_attempt(self):
        with self.writer():
            with patch('signallab.synthetic._load',side_effect=AssertionError('running records are not a coherent snapshot')):
                result=self.read()
            self.assertEqual(result['generation_state'],'running')
            self.assertIsNone(result['attempts'])
            self.assertEqual(result['candidates'],{})

    def test_history_can_be_read_without_overwriting_prior_decisions(self):
        candidate=next(iter(self.complete()['candidates']))
        self.review(candidate)
        self.review(candidate,expected=1,verdict='reject',reason='Later review found factual drift.')
        old=synthetic.read_review_revision(self.root,self.pilot_id,1,studies_root=self.studies)
        self.assertEqual(old['decisions'][candidate]['verdict'],'accept')
        self.assertEqual(old['revision'],1)
        with self.assertRaises(ValueError):
            synthetic.read_review_revision(self.root,self.pilot_id,3,studies_root=self.studies)
        old_file=self.root/self.pilot_id/'review-0001.json'
        old_file.rename(old_file.with_name('preserved-old-snapshot.json'))
        with self.assertRaises(ValueError):
            synthetic.read_reviews(self.root,self.pilot_id,studies_root=self.studies)

    def test_retry_reestablishes_directory_durability_before_success(self):
        candidate=next(iter(self.complete()['candidates']))
        decision=uuid.uuid4().hex
        fsync=os.fsync
        calls=0
        def fail_last(fd):
            nonlocal calls
            calls+=1
            if calls==4:
                raise OSError('injected directory sync failure after commit link')
            fsync(fd)
        with patch('signallab.synthetic.os.fsync',side_effect=fail_last):
            with self.assertRaises(OSError):
                self.review(candidate,decision_id=decision)
        self.assertEqual(synthetic.read_reviews(self.root,self.pilot_id,studies_root=self.studies)['revision'],1)
        with patch('signallab.synthetic.os.fsync',side_effect=OSError('directory still unavailable')):
            with self.assertRaises(OSError):
                self.review(candidate,decision_id=decision)
        self.assertEqual(self.review(candidate,decision_id=decision),1)

    def test_safe_reader_rejects_symlinks_corruption_and_oversized_records(self):
        self.complete()
        linked=self.folder/'linked-root'
        linked.symlink_to(self.root,target_is_directory=True)
        with self.assertRaises(ValueError):
            synthetic.read_pilot(linked,self.pilot_id,studies_root=self.studies)
        for bad in ('../escape',None,'x'*32):
            with self.assertRaises(ValueError):
                synthetic.read_pilot(self.root,bad,studies_root=self.studies)
        terminal=self.root/self.pilot_id/'terminal.json'
        original=terminal.read_bytes()
        for content in (b'{',b'{"x":1,"x":2}',b'{"x":NaN}',b'['*1500,b'x'*(synthetic.MAX_BYTES+1),
                        ('{"huge":'+ '9'*1000 +'}').encode()):
            terminal.write_bytes(content)
            with self.assertRaises(ValueError):
                self.read()
        terminal.write_bytes(original)
        terminal.rename(terminal.with_name('terminal-preserved.json'))
        terminal.symlink_to(terminal.with_name('terminal-preserved.json'))
        with self.assertRaises(ValueError):
            self.read()

    def test_review_revision_bound_and_metadata_only_history_reads(self):
        candidate=next(iter(self.complete()['candidates']))
        # Smaller bound exercises the same publication/CAS boundary without 200 fsync cycles.
        self.assertEqual(synthetic.MAX_REVISIONS,self.protocol['policy']['review']['max_revisions'])
        with patch('signallab.synthetic.MAX_REVISIONS',2):
            self.review(candidate)
            self.review(candidate,expected=1,verdict='reject')
            with self.assertRaisesRegex(ValueError,'revision limit'):
                self.review(candidate,expected=2)
            read=synthetic._read
            def track(fd,name,**kwargs):
                if name=='review-0001.json':
                    raise AssertionError('ordinary latest read must not deserialize all older snapshots')
                return read(fd,name,**kwargs)
            with patch('signallab.synthetic._read',side_effect=track):
                latest=synthetic.read_reviews(self.root,self.pilot_id,studies_root=self.studies)
            self.assertEqual(latest['revision'],2)

    def test_terminal_reason_is_sanitized_and_cannot_claim_success_over_budget(self):
        with self.writer() as writer:
            for parent in self.parents:
                writer.skip(parent,129)
            with self.assertRaises(ValueError):
                writer.finish('complete',dict(RESOURCES,elapsed_seconds=1800),worker_reaped=True)
            with self.assertRaises(ValueError):
                writer.finish('failed',RESOURCES,worker_reaped=True,error_code='raw arbitrary traceback')
            writer.finish('stopped_budget',dict(RESOURCES,elapsed_seconds=1800),worker_reaped=True)
        self.assertEqual(self.read()['terminal']['termination_code'],'wall_limit')

    def test_model_identity_and_protocol_are_revalidated_before_review(self):
        for assets in ({},dict(ASSETS,other='a'*64),dict(ASSETS,**{'config.json':True})):
            with self.assertRaises(ValueError):
                synthetic.create_pilot(self.protocol['protocol_id'],self.frame,assets,root=self.root,studies_root=self.studies)
        candidate=next(iter(self.complete()['candidates']))
        protocol_file=self.studies/self.protocol['protocol_id']/'protocol.json'
        protocol=json.loads(protocol_file.read_text())
        protocol['threads']=1
        protocol['content_digest']=study._digest({k:v for k,v in protocol.items() if k!='content_digest'})
        protocol_file.write_text(json.dumps(protocol))
        with self.assertRaisesRegex(ValueError,'frozen protocol'):
            self.review(candidate)

    def test_inherited_generation_descriptor_fences_recovery_until_child_exit(self):
        child=None
        try:
            with self.writer() as writer:
                child=subprocess.Popen([sys.executable,'-c',
                    'import os,sys; os.fstat(int(sys.argv[1])); sys.stdin.read(1)',str(writer.lock_fd)],
                    pass_fds=(writer.lock_fd,),stdin=subprocess.PIPE,stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,text=True)
            self.assertEqual(self.read()['generation_state'],'running')
            with self.assertRaises(ValueError):
                with self.writer():
                    pass
        finally:
            if child is not None:
                child.stdin.close()
                child.wait(timeout=20)
                self.assertEqual(child.stderr.read(),'')
                child.stderr.close()
        self.assertEqual(self.read()['generation_state'],'incomplete')

    def test_candidate_rewrite_cannot_escape_terminal_manifest_binding(self):
        self.complete()
        path=self.root/self.pilot_id/'attempt-00.result.json'
        result=json.loads(path.read_text())
        result['text']='Tampered text with internally updated candidate fields'
        import hashlib
        result['text_digest']=hashlib.sha256(result['text'].encode()).hexdigest()
        result['fingerprint']=study.text_fingerprint(result['text'])
        result['candidate_id']=synthetic._digest([self.pilot_id,result['parent_id'],result['text_digest']])
        result['content_digest']=synthetic._digest({k:v for k,v in result.items() if k!='content_digest'})
        path.write_text(json.dumps(result))
        with self.assertRaisesRegex(ValueError,'Terminal manifest'):
            self.read()

    def test_edited_review_snapshot_does_not_match_its_commit(self):
        candidate=next(iter(self.complete()['candidates']))
        self.review(candidate)
        path=self.root/self.pilot_id/'review-0001.json'
        snapshot=json.loads(path.read_text())
        snapshot['decisions'][candidate]['verdict']='reject'
        snapshot['content_digest']=synthetic._digest({k:v for k,v in snapshot.items() if k!='content_digest'})
        path.write_text(json.dumps(snapshot))
        with self.assertRaisesRegex(ValueError,'commit chain'):
            synthetic.read_reviews(self.root,self.pilot_id,studies_root=self.studies)


if __name__ == '__main__':
    unittest.main()
