"""Offline synthetic CLI/model/supervisor contracts. No Hub weights downloaded."""
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import shutil
import sys
import tempfile
import time
import unittest
import uuid
from unittest.mock import patch,Mock
from types import SimpleNamespace

from signallab import study, synthetic, synthetic_cli as cli
from signallab.demo import demo_reviews
from signallab.core import review_id

ROOT=Path(__file__).resolve().parents[1]


class GeneratorContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scratch=Path(tempfile.mkdtemp(prefix='generator-contract-',dir=ROOT/'.scratch'))
        cls.frame=demo_reviews()
        cls.final=cls.frame.assign(text='reserved generator fixture '+cls.frame.text)
        cls.fake=cls.scratch/'owned_worker_fixture.py'
        cls.fake.write_text('import sys\nsys.path.insert(0,'+repr(str(ROOT))+')\n'+'''import os,sys,socket,threading,time,struct,signal
from signallab.synthetic_worker import receive,send,_watch
mode=sys.argv[1]
channel=socket.socket(fileno=int(sys.argv[2]))
threading.Thread(target=_watch,args=(int(sys.argv[3]),),daemon=True).start()
job=receive(channel)
if mode=='die': os._exit(3)
if mode=='hang': time.sleep(60)
if mode=='oversize':
 channel.sendall(struct.pack('!I',128*1024+1)); time.sleep(60)
send(channel,{'type':'ready',**({'extra':True} if mode=='bad_ready' else {})})
i=0
while True:
 message=receive(channel)
 if message['type']=='stop': send(channel,{'type':'stopped'}); break
 if message['type']=='measure':
  send(channel,{'type':'measured','input_tokens':129 if mode=='skip' else 30})
 if message['type']=='generate':
  if mode=='ignore_term': signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(60)
  if mode=='hang_generate': time.sleep(60)
  if mode=='bad_result': send(channel,{'type':'generated','text':'x','output_tokens':129,'eos':True})
  else: send(channel,{'type':'generated','text':'authored candidate '+str(i),'output_tokens':8,'eos':True})
  i+=1
''')

    def setUp(self):
        self.folder=self.scratch/uuid.uuid4().hex
        self.studies=self.folder/'studies'
        self.prepared=self.folder/'generators'
        self.pilots=self.folder/'pilots'
        self.protocol=study.build_protocol(self.frame,self.final,source='authored development',
            final_source='separate authored final',attest_final_reserved=True,max_parents=2)
        study.save_protocol(self.protocol,self.studies)

    def test_cache_miss_never_downloads_or_creates_pilot(self):
        with patch('signallab.synthetic_cli._download_assets',side_effect=AssertionError('network called')):
            with self.assertRaisesRegex(ValueError,'prepare'):
                cli.generate_pilot(self.protocol['protocol_id'],self.frame,studies_root=self.studies,
                    prepared_root=self.prepared,root=self.pilots)
        self.assertFalse(self.pilots.exists())

    def test_help_is_side_effect_free_and_names_supported_actions(self):
        with contextlib.redirect_stdout(io.StringIO()) as output:
            with self.assertRaises(SystemExit) as result:
                cli.main(['--help'])
        self.assertEqual(result.exception.code,0)
        for name in ('freeze','prepare','generate','recover','inspect'):
            self.assertIn(name,output.getvalue())
        self.assertFalse(self.prepared.exists())

    def test_asset_bounds_config_and_symlinks_reject_before_model_loading(self):
        assets=self.folder/'assets'
        assets.mkdir()
        config=dict(model_type='t5',architectures=['T5ForConditionalGeneration'],is_encoder_decoder=True,
                    vocab_size=8,d_model=16,d_ff=32,num_layers=1,num_decoder_layers=1,num_heads=2,d_kv=8,
                    decoder_start_token_id=0,pad_token_id=0,eos_token_id=1)
        for name,data in {'config.json':json.dumps(config).encode(),'tokenizer.json':b'{}',
                          'tokenizer_config.json':b'{}','model.safetensors':b'fixture bytes'}.items():
            (assets/name).write_bytes(data)
        hashes=cli.verify_assets(assets)
        self.assertEqual(set(hashes),synthetic.REQUIRED_ASSETS)
        bad=dict(config,d_model=True)
        (assets/'config.json').write_text(json.dumps(bad))
        with self.assertRaises(ValueError):
            cli.verify_assets(assets)
        (assets/'config.json').write_text(json.dumps(config))
        (assets/'arbitrary.bin').write_bytes(b'not allowed')
        with self.assertRaises(ValueError):
            cli.verify_assets(assets)

    def test_recovery_never_loads_models_or_resumes_attempts(self):
        assets={name:'a'*64 for name in synthetic.REQUIRED_ASSETS}
        pilot=synthetic.create_pilot(self.protocol['protocol_id'],self.frame,assets,root=self.pilots,studies_root=self.studies)
        with synthetic.generation(self.pilots,pilot,studies_root=self.studies) as writer:
            writer.start(self.protocol['selection']['parent_ids'][0],30)
        with patch('signallab.synthetic_cli._load_generator',side_effect=AssertionError('loaded model')):
            result=cli.recover_pilot(pilot,root=self.pilots,studies_root=self.studies)
        self.assertEqual(result['generation_state'],'interrupted')
        self.assertEqual(result['terminal']['resource_status'],'unknown')
        self.assertEqual(result['attempts'][0]['state'],'interrupted')

    def run_fake(self,mode,*,probe=None,limits=None,clock=None,finalize=False):
        assets={name:'a'*64 for name in synthetic.REQUIRED_ASSETS}
        pilot=synthetic.create_pilot(self.protocol['protocol_id'],self.frame,assets,root=self.pilots,studies_root=self.studies)
        parents=[(rid,'Authored training text') for rid in self.protocol['selection']['parent_ids']]
        kwargs=dict(command=[sys.executable,str(self.fake),mode],probe=probe or (lambda process:1000))
        if limits is not None: kwargs['limits']=limits
        if clock is not None: kwargs['clock']=clock
        with synthetic.generation(self.pilots,pilot,studies_root=self.studies) as writer:
            state,code,resources=cli._run_worker({'fixture':'no model'},self.protocol,writer=writer,parents=parents,**kwargs)
            if finalize:
                writer.finish(state,resources,worker_reaped=True,error_code=code)
        return pilot,state,code,resources

    def test_fake_worker_success_and_length_skip_are_bounded_one_attempt_each(self):
        for mode in ('normal','skip'):
            pilot,state,code,resources=self.run_fake(mode)
            self.assertEqual(state,'complete')
            self.assertIsNone(code)
            self.assertGreater(resources['samples'],0)
            result=synthetic.read_pilot(self.pilots,pilot,studies_root=self.studies)
            self.assertTrue(all(a['state']==('skipped_length' if mode=='skip' else 'generated') for a in result['attempts']))

    def test_malformed_or_dead_worker_becomes_failure_with_preserved_intent(self):
        for mode in ('die','oversize','bad_ready','bad_result'):
            pilot,state,code,_=self.run_fake(mode)
            self.assertEqual(state,'failed')
            self.assertIn(code,('invalid_output','model_load_error'))
            if mode=='bad_result':
                result=synthetic.read_pilot(self.pilots,pilot,studies_root=self.studies)
                self.assertEqual(result['attempts'][0]['result']['error_code'],'invalid_output')

    def test_wall_timeout_and_kill_escalation_only_affect_owned_worker(self):
        limits=dict(self.protocol['policy']['resources'],wall_seconds=0.4,rss_sample_seconds=0.02,termination_grace_seconds=0.1)
        sentinel=subprocess.Popen([sys.executable,'-c','import sys; sys.stdin.read()'],stdin=subprocess.PIPE)
        try:
            for mode in ('hang','ignore_term'):
                started=time.monotonic()
                _,state,code,_=self.run_fake(mode,limits=limits)
                self.assertEqual((state,code),('stopped_budget','wall_limit'))
                self.assertLess(time.monotonic()-started,4)
                self.assertIsNone(sentinel.poll())
        finally:
            sentinel.stdin.close()
            sentinel.wait(timeout=5)

    def test_rss_limit_and_failed_probe_preserve_observed_samples(self):
        _,state,code,_=self.run_fake('hang',probe=lambda process:4*1024**3)
        self.assertEqual((state,code),('stopped_budget','rss_limit'))
        count=0
        def unavailable(process):
            nonlocal count
            count+=1
            if count>1: raise OSError('injected unavailable probe')
            return 2345
        _,state,code,resources=self.run_fake('hang',probe=unavailable)
        self.assertEqual((state,code),('failed','rss_unavailable'))
        self.assertEqual(resources['observed_peak_rss_bytes'],2345)
        self.assertEqual(resources['observed_samples'],1)
        self.assertEqual(synthetic._resource_status(resources,self.protocol),'unknown')
        with self.assertRaises(ValueError):
            synthetic._resource_status(dict(resources,method='macos-ps-rss-kib-v1',peak_rss_bytes=2345,samples=1),self.protocol)

    def test_wall_breach_is_still_failure_when_rss_was_unavailable(self):
        resources=dict(elapsed_seconds=1801,peak_rss_bytes=None,samples=0,method='unavailable',
                       observed_peak_rss_bytes=None,observed_samples=0)
        self.assertEqual(synthetic._resource_status(resources,self.protocol),'fail')

    def test_pilot_binds_prepared_identity_and_execution_environment(self):
        execution={'preparation_id':uuid.uuid4().hex,'prepared_digest':'b'*64,'environment':cli._environment()}
        pilot=synthetic.create_pilot(self.protocol['protocol_id'],self.frame,{n:'a'*64 for n in synthetic.REQUIRED_ASSETS},
            execution=execution,root=self.pilots,studies_root=self.studies)
        result=synthetic.read_pilot(self.pilots,pilot,studies_root=self.studies)
        self.assertEqual(result['pilot']['execution'],execution)
        with self.assertRaises(ValueError):
            synthetic.create_pilot(self.protocol['protocol_id'],self.frame,{n:'a'*64 for n in synthetic.REQUIRED_ASSETS},
                execution=dict(execution,prepared_digest='bad'),root=self.pilots,studies_root=self.studies)

    def test_worker_loss_after_supervisor_death_releases_inherited_generation_lock(self):
        assets={name:'a'*64 for name in synthetic.REQUIRED_ASSETS}
        pilot=synthetic.create_pilot(self.protocol['protocol_id'],self.frame,assets,root=self.pilots,studies_root=self.studies)
        ready=self.folder/'supervisor-ready'
        program='''import sys,json
from pathlib import Path
from signallab import synthetic,study,synthetic_cli as cli
root,studies,pilot,ready,fake=sys.argv[1:]
with synthetic.generation(root,pilot,studies_root=studies) as writer:
 writer.start(writer.pilot['parent_ids'][0],30)
 worker=cli.WorkerSession({'fixture':'blocked'},writer.protocol['policy']['resources'],lock_fd=writer.lock_fd,
  command=[sys.executable,fake,'hang'],probe=lambda process:1000)
 Path(ready).write_text('owned worker started')
 sys.stdin.read()
'''
        supervisor=subprocess.Popen([sys.executable,'-c',program,str(self.pilots),str(self.studies),pilot,str(ready),str(self.fake)],stdin=subprocess.PIPE)
        try:
            deadline=time.monotonic()+20
            while not ready.exists() and supervisor.poll() is None and time.monotonic()<deadline:
                time.sleep(0.02)
            self.assertTrue(ready.exists())
            supervisor.terminate()  # Test-owned supervisor only, never a process found by PID/name.
            supervisor.wait(timeout=5)
            deadline=time.monotonic()+5
            while True:
                try:
                    result=cli.recover_pilot(pilot,root=self.pilots,studies_root=self.studies)
                    break
                except ValueError as exc:
                    if 'busy' not in str(exc) or time.monotonic()>=deadline: raise
                    time.sleep(0.02)
            self.assertEqual(result['generation_state'],'interrupted')
        finally:
            supervisor.stdin.close()
            if supervisor.poll() is None:
                supervisor.terminate()
                supervisor.wait(timeout=5)

    @classmethod
    def tiny_assets(cls):
        path=cls.scratch/'tiny-local-t5'
        if path.exists(): return path
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        from tokenizers.processors import TemplateProcessing
        from transformers import PreTrainedTokenizerFast,T5Config,T5ForConditionalGeneration
        import torch
        path.mkdir()
        backend=Tokenizer(WordLevel({'[PAD]':0,'[EOS]':1,'[UNK]':2,'review':3,'good':4,'bad':5,'fixture':6,'the':7},unk_token='[UNK]'))
        backend.pre_tokenizer=Whitespace()
        backend.post_processor=TemplateProcessing(single='$A [EOS]',special_tokens=[('[EOS]',1)])
        tokenizer=PreTrainedTokenizerFast(tokenizer_object=backend,pad_token='[PAD]',eos_token='[EOS]',unk_token='[UNK]')
        tokenizer.save_pretrained(path)
        torch.manual_seed(7)
        model=T5ForConditionalGeneration(T5Config(vocab_size=8,d_model=16,d_ff=32,num_layers=1,
              num_decoder_layers=1,num_heads=2,d_kv=8,decoder_start_token_id=0,pad_token_id=0,eos_token_id=1))
        model.save_pretrained(path,safe_serialization=True)
        return path

    def fixture_download(self,protocol,destination):
        destination.mkdir(mode=0o700)
        for source in self.tiny_assets().iterdir():
            if source.name in synthetic.ASSET_NAMES:
                shutil.copyfile(source,destination/source.name)
                (destination/source.name).chmod(0o600)
        return cli.verify_assets(destination)

    def test_real_offline_t5_preparation_and_generation_publish_matching_evidence(self):
        with patch('signallab.synthetic_cli._download_assets',side_effect=self.fixture_download):
            prepared=cli.prepare_model(self.protocol['protocol_id'],studies_root=self.studies,prepared_root=self.prepared)
        self.assertEqual(synthetic._resource_status(prepared['resources'],self.protocol),'pass')
        with patch('signallab.synthetic_cli._download_assets',side_effect=AssertionError('network fallback')):
            result=cli.generate_pilot(self.protocol['protocol_id'],self.frame,studies_root=self.studies,
                     prepared_root=self.prepared,root=self.pilots)
        self.assertEqual(result['generation_state'],'complete')
        self.assertEqual(len(result['candidates']),2)
        self.assertEqual(result['pilot']['execution']['prepared_digest'],prepared['content_digest'])
        self.assertGreater(result['terminal']['resources']['samples'],0)
        tokenizer,_,_=cli._load_generator(self.tiny_assets(),cli.verify_assets(self.tiny_assets()),2)
        text_by_id={review_id(r.text,r.label):r.text for r in self.frame.itertuples()}
        for attempt in result['attempts']:
            expected=len(tokenizer(study.PROMPT.format(text=text_by_id[attempt['parent_id']]),truncation=False)['input_ids'])
            self.assertEqual(attempt['intent']['input_tokens'],expected)
            self.assertLessEqual(attempt['result']['output_tokens'],128)
        # Corrupt prepared bytes: fail before allocating another pilot or falling back online.
        count=len(list(self.pilots.iterdir()))
        (self.prepared/prepared['preparation_id']/'assets/config.json').write_text('{}')
        with self.assertRaises(ValueError):
            cli.generate_pilot(self.protocol['protocol_id'],self.frame,studies_root=self.studies,
                     prepared_root=self.prepared,root=self.pilots)
        self.assertEqual(len(list(self.pilots.iterdir())),count)

    def test_incomplete_real_weights_are_rejected_instead_of_random_init(self):
        from safetensors.torch import load_file,save_file
        path=self.folder/'incomplete-model'
        path.mkdir()
        for source in self.tiny_assets().iterdir():
            shutil.copyfile(source,path/source.name)
        weights=load_file(path/'model.safetensors')
        weights.pop('encoder.block.0.layer.0.SelfAttention.q.weight')
        save_file(weights,path/'model.safetensors',metadata={'format':'pt'})
        with self.assertRaisesRegex(ValueError,'incomplete'):
            cli._load_generator(path,cli.verify_assets(path),2)

    def test_downloaded_but_unloadable_preparation_never_becomes_ready(self):
        with patch('signallab.synthetic_cli._download_assets',side_effect=self.fixture_download),\
             patch('signallab.synthetic_cli._run_worker',return_value=('failed','model_load_error',
                  dict(elapsed_seconds=1,peak_rss_bytes=None,samples=0,method='unavailable'))):
            with self.assertRaisesRegex(ValueError,'MODEL_NOT_READY'):
                cli.prepare_model(self.protocol['protocol_id'],studies_root=self.studies,prepared_root=self.prepared)
        self.assertTrue(any(self.prepared.iterdir()))
        self.assertFalse(any(self.prepared.glob('*/prepared.json')))
        with self.assertRaisesRegex(ValueError,'prepare'):
            cli.resolve_prepared(self.protocol,self.prepared)

    def test_cli_freeze_and_text_free_json_inspection_are_real_supported_paths(self):
        dev=self.folder/'development.csv'
        final=self.folder/'final.csv'
        dev.write_text(self.frame.to_csv(index=False))
        final.write_text(self.final.to_csv(index=False))
        roots=['--studies-root',str(self.studies),'--prepared-root',str(self.prepared),'--pilots-root',str(self.pilots)]
        with contextlib.redirect_stdout(io.StringIO()) as output:
            code=cli.main([*roots,'freeze','--csv',str(dev),'--final-csv',str(final),
                '--source','authored dev','--final-source','authored final','--attest-final-reserved'])
        self.assertEqual(code,0)
        self.assertIn('Next:',output.getvalue())
        pilot,_,_,_=self.run_fake('normal',finalize=True)
        with contextlib.redirect_stdout(io.StringIO()) as output:
            code=cli.main([*roots,'inspect','--pilot',pilot,'--json'])
        self.assertEqual(code,0)
        record=json.loads(output.getvalue())
        self.assertEqual(record['generation_state'],'complete')
        self.assertEqual(record['reviews']['pending'],2)
        self.assertTrue(record['no_training_has_run'])
        self.assertNotIn('authored candidate',output.getvalue())
        for text in self.frame.text:
            self.assertNotIn(text,output.getvalue())

    def test_explicit_hub_preparation_pins_revision_and_checks_sizes_before_download(self):
        source=self.tiny_assets()
        files=[SimpleNamespace(rfilename=p.name,size=p.stat().st_size) for p in source.iterdir() if p.name in synthetic.ASSET_NAMES]
        info=SimpleNamespace(sha=study.POLICY['generator']['revision'],siblings=files)
        destination=self.folder/'downloaded-assets'
        with patch('huggingface_hub.HfApi') as api,patch('huggingface_hub.hf_hub_download',side_effect=lambda repo,name,**kw:str(source/name)) as download:
            api.return_value.model_info.return_value=info
            hashes=cli._download_assets(self.protocol,destination)
            self.assertEqual(hashes,cli.verify_assets(destination))
            for call in download.call_args_list:
                self.assertEqual(call.kwargs['revision'],study.POLICY['generator']['revision'])
                self.assertIs(call.kwargs['local_files_only'],False)
            download.reset_mock()
            info.sha='wrong-revision'
            with self.assertRaises(ValueError):
                cli._download_assets(self.protocol,self.folder/'invalid-revision')
            download.assert_not_called()
            info.sha=study.POLICY['generator']['revision']
            next(f for f in files if f.rfilename=='model.safetensors').size=cli.MAX_WEIGHTS+1
            with self.assertRaises(ValueError):
                cli._download_assets(self.protocol,self.folder/'oversize-remote')
            download.assert_not_called()

    def test_platform_preflight_fails_before_preparation_output(self):
        with patch('signallab.synthetic_cli.platform.system',return_value='Windows'):
            with self.assertRaisesRegex(ValueError,'SUPPORTED_PLATFORM'):
                cli.prepare_model(self.protocol['protocol_id'],studies_root=self.studies,prepared_root=self.prepared)
        self.assertFalse(self.prepared.exists())

    def test_asset_replacement_during_actual_model_load_is_rejected(self):
        from transformers import AutoModelForSeq2SeqLM
        path=self.folder/'changing-model'
        shutil.copytree(self.tiny_assets(),path)
        load=AutoModelForSeq2SeqLM.from_pretrained
        def mutate(*args,**kwargs):
            loaded=load(*args,**kwargs)
            with (path/'config.json').open('a') as file:
                file.write(' ')
            return loaded
        with patch('transformers.AutoModelForSeq2SeqLM.from_pretrained',side_effect=mutate):
            with self.assertRaisesRegex(ValueError,'changed during load'):
                cli._load_generator(path,cli.verify_assets(path),2)

    def test_cli_sanitizes_library_errors_and_reports_actionable_cache_miss(self):
        with patch('signallab.synthetic_cli.prepare_model',side_effect=OSError('untrusted library diagnostic')),\
             contextlib.redirect_stderr(io.StringIO()) as output:
            code=cli.main(['prepare','--protocol',self.protocol['protocol_id']])
        self.assertEqual(code,1)
        self.assertNotIn('untrusted library diagnostic',output.getvalue())
        with contextlib.redirect_stderr(io.StringIO()) as output:
            code=cli.main(['--studies-root',str(self.studies),'--prepared-root',str(self.prepared),
                          '--pilots-root',str(self.pilots),'generate','--protocol',self.protocol['protocol_id'],
                          '--csv',str(self.folder/'missing.csv')])
        self.assertEqual(code,1)
        self.assertIn('generate failed',output.getvalue())

    def test_real_tokenizer_overflow_skips_intact_reviews_without_model_calls(self):
        frame=self.frame.assign(text=self.frame.text+' lengthyword '*140)
        protocol=study.build_protocol(frame,self.final,source='long authored fixture',
                   final_source='separate authored final',attest_final_reserved=True,max_parents=2)
        study.save_protocol(protocol,self.studies)
        with patch('signallab.synthetic_cli._download_assets',side_effect=self.fixture_download):
            cli.prepare_model(protocol['protocol_id'],studies_root=self.studies,prepared_root=self.prepared)
        result=cli.generate_pilot(protocol['protocol_id'],frame,studies_root=self.studies,
                    prepared_root=self.prepared,root=self.pilots)
        self.assertEqual(result['generation_state'],'complete')
        self.assertEqual(result['terminal']['counts']['skipped_length'],2)
        self.assertEqual(result['terminal']['counts']['generated'],0)
        for attempt in result['attempts']:
            self.assertIsNone(attempt['intent'])
            self.assertGreater(attempt['result']['input_tokens'],128)

    def test_keyboard_interrupt_preserves_active_intent_and_finalizes_after_cleanup(self):
        samples=0
        def interrupt(process):
            nonlocal samples
            samples+=1
            if samples>1: raise KeyboardInterrupt()
            return 1000
        pilot,state,code,_=self.run_fake('hang_generate',probe=interrupt,finalize=True)
        self.assertEqual((state,code),('interrupted','operator_interrupt'))
        result=synthetic.read_pilot(self.pilots,pilot,studies_root=self.studies)
        self.assertEqual(result['attempts'][0]['state'],'interrupted')
        self.assertEqual(result['terminal']['counts']['generated'],0)

    def test_unreaped_child_never_returns_a_successful_cleanup_attestation(self):
        session=cli.WorkerSession.__new__(cli.WorkerSession)
        session.process=Mock()
        session.process.poll.return_value=None
        session.process.wait.side_effect=subprocess.TimeoutExpired('owned fixture child',0.1)
        session.sock=Mock()
        read_fd,session.life_write=os.pipe()
        session.limits={'termination_grace_seconds':0.1}
        try:
            with self.assertRaisesRegex(cli.GeneratorError,'WORKER_NOT_REAPED'):
                session.close()
        finally:
            os.close(read_fd)
        session.process.terminate.assert_called_once()
        session.process.kill.assert_called_once()
        session.sock.close.assert_called_once()


if __name__=='__main__':
    unittest.main()
