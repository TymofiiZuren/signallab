"""Explicit local synthetic-study commands. Only `prepare` contacts the Hub."""
import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import select
import shlex
import socket
import stat
import struct
import subprocess
import sys
import time
import uuid

from signallab import study, synthetic
from signallab.core import review_id

PREPARED_ROOT=Path(__file__).resolve().parents[1]/'artifacts/generators'
MAX_MESSAGE=128*1024
MAX_ASSET_JSON=8*1024*1024
MAX_WEIGHTS=512*1024*1024


class GeneratorError(ValueError):
    """Safe application-authored diagnostic; library exceptions stay sanitized."""


def _require(value,message):
    if not value:
        raise GeneratorError(message)


def _platform_check():
    _require(platform.system()=='Darwin' and Path('/bin/ps').is_file(),
             'SUPPORTED_PLATFORM: generator supervision is validated on macOS only; no pilot started.')
    try:
        _rss_pid(os.getpid())
    except (OSError,ValueError,subprocess.TimeoutExpired) as exc:
        raise GeneratorError('SUPPORTED_PLATFORM: RSS probe preflight failed; no preparation or pilot started.') from exc


def _json_bytes(data):
    try:
        return json.loads(data,object_pairs_hook=synthetic._unique,parse_constant=synthetic._constant)
    except (UnicodeDecodeError,json.JSONDecodeError,RecursionError) as exc:
        raise ValueError('Invalid bounded generator JSON.') from exc


def _config(value):
    synthetic._tree(value)
    _require(isinstance(value,dict) and value.get('model_type')=='t5' and
             value.get('is_encoder_decoder') is True and value.get('architectures')==['T5ForConditionalGeneration'] and
             'auto_map' not in value,'MODEL_CONFIG: only a standard T5 encoder-decoder is supported.')
    for key,maximum in {'vocab_size':65536,'d_model':1024,'d_ff':4096,'num_layers':24,'num_heads':32,'d_kv':256}.items():
        _require(synthetic._integer(value.get(key),1,maximum),f'MODEL_CONFIG: invalid {key}.')
    _require(synthetic._integer(value.get('num_decoder_layers',value['num_layers']),1,24),'MODEL_CONFIG: invalid decoder layers.')
    for key in ('pad_token_id','eos_token_id','decoder_start_token_id'):
        _require(synthetic._integer(value.get(key),0,value['vocab_size']-1),f'MODEL_CONFIG: invalid {key}.')
    _require(value['pad_token_id']!=value['eos_token_id'],'MODEL_CONFIG: pad and EOS must differ.')


def _scan_assets(path):
    """Hash bounded regular allowlisted files and record their actual identities."""
    try:
        fd=os.open(path,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    except OSError as exc:
        raise ValueError('MODEL_ASSETS: cannot safely open the local asset directory.') from exc
    hashes,identities,documents={},{},{}
    try:
        with os.scandir(fd) as entries:
            names=[]
            for entry in entries:
                _require(len(names)<20,'MODEL_ASSETS: at most 20 files are supported.')
                names.append(entry.name)
        _require(synthetic.REQUIRED_ASSETS<=set(names)<=synthetic.ASSET_NAMES,'MODEL_ASSETS: missing required or unexpected files; no fallback loading.')
        for name in sorted(names):
            try:
                file_fd=os.open(name,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK,dir_fd=fd)
            except OSError as exc:
                raise ValueError('MODEL_ASSETS: symlinks and unreadable files are unsupported.') from exc
            with os.fdopen(file_fd,'rb') as file:
                before=os.fstat(file.fileno())
                limit=MAX_WEIGHTS if name=='model.safetensors' else MAX_ASSET_JSON
                _require(stat.S_ISREG(before.st_mode) and 0<before.st_size<=limit,'MODEL_ASSETS: unsupported asset type/size.')
                checksum=hashlib.sha256()
                total=0
                chunks=[] if name in ('config.json','tokenizer_config.json') else None
                while chunk:=file.read(1024*1024):
                    total+=len(chunk)
                    _require(total<=limit,'MODEL_ASSETS: file grew beyond its size bound.')
                    checksum.update(chunk)
                    if chunks is not None:
                        chunks.append(chunk)
                after=os.fstat(file.fileno())
                identity=lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns)
                _require(identity(before)==identity(after) and total==before.st_size,'MODEL_ASSETS: asset changed during verification.')
                hashes[name]=checksum.hexdigest()
                identities[name]=identity(after)
                if chunks is not None:
                    documents[name]=_json_bytes(b''.join(chunks))
        _config(documents['config.json'])
        token_config=documents['tokenizer_config.json']
        _require(isinstance(token_config,dict) and 'auto_map' not in token_config and
                 token_config.get('tokenizer_class') in (None,'T5Tokenizer','T5TokenizerFast','PreTrainedTokenizerFast'),
                 'MODEL_TOKENIZER: unsupported tokenizer class or remote-code mapping.')
    finally:
        os.close(fd)
    return hashes,identities,documents['config.json']


def verify_assets(path,expected=None):
    hashes,_,_=_scan_assets(path)
    if expected is not None:
        synthetic._assets(expected)
        _require(hashes==expected,'MODEL_ASSETS_CHANGED: checksums differ; preserve this snapshot and prepare a new one.')
    return hashes


def _load_generator(path,assets,threads):
    _require(synthetic._integer(threads,1,8),'MODEL_THREADS: use integer threads 1–8.')
    before,identities,config=_scan_assets(path)
    _require(before==assets,'MODEL_ASSETS_CHANGED: local model bytes differ from the prepared identity.')
    os.environ['HF_HUB_OFFLINE']='1'
    os.environ['TRANSFORMERS_OFFLINE']='1'
    os.environ['HF_HUB_DISABLE_TELEMETRY']='1'
    os.environ['TOKENIZERS_PARALLELISM']='false'
    import torch
    from transformers import AutoModelForSeq2SeqLM,AutoTokenizer,GenerationConfig
    torch.set_num_threads(threads)
    torch.manual_seed(42)
    tokenizer=AutoTokenizer.from_pretrained(path,use_fast=True,local_files_only=True,trust_remote_code=False)
    _require(tokenizer.is_fast,'MODEL_TOKENIZER: pinned fast tokenizer is unavailable; no dependency or slow-tokenizer fallback.')
    _require(tokenizer.pad_token_id==config['pad_token_id'] and tokenizer.eos_token_id==config['eos_token_id'] and
             len(tokenizer)<=config['vocab_size'],'MODEL_TOKENIZER: tokenizer/config token identities disagree.')
    model,info=AutoModelForSeq2SeqLM.from_pretrained(path,local_files_only=True,trust_remote_code=False,
                    use_safetensors=True,torch_dtype=torch.float32,output_loading_info=True)
    _require(not any(info.get(key) for key in ('missing_keys','unexpected_keys','mismatched_keys','error_msgs')),
             'MODEL_WEIGHTS: incomplete or incompatible checkpoint; random initialization is not accepted.')
    _require(model.config.model_type=='t5' and model.config.is_encoder_decoder,'MODEL_CONFIG: unsupported loaded model.')
    model.to(device='cpu',dtype=torch.float32)
    model.eval()
    # Fresh config avoids inherited forced tokens, sampling or other saved overrides.
    settings=GenerationConfig(do_sample=False,num_beams=1,num_return_sequences=1,max_new_tokens=128,
                pad_token_id=config['pad_token_id'],eos_token_id=config['eos_token_id'],
                decoder_start_token_id=config['decoder_start_token_id'],use_cache=True)
    after,after_identities,_=_scan_assets(path)
    _require(after==assets and identities==after_identities,'MODEL_ASSETS_CHANGED: model files changed during load.')
    return tokenizer,model,settings


def _download_assets(protocol,destination):
    """The only network-owning path. Copy allowlisted bytes into a private snapshot."""
    from huggingface_hub import HfApi,hf_hub_download
    generator=protocol['policy']['generator']
    info=HfApi().model_info(generator['model'],revision=generator['revision'],files_metadata=True)
    _require(info.sha==generator['revision'],'MODEL_REVISION: resolved Hub revision differs from the frozen revision.')
    files={item.rfilename:item for item in info.siblings if item.rfilename in synthetic.ASSET_NAMES}
    _require(synthetic.REQUIRED_ASSETS<=set(files),'MODEL_ASSETS: required fast-tokenizer/safetensors assets are missing.')
    for name,item in files.items():
        limit=MAX_WEIGHTS if name=='model.safetensors' else MAX_ASSET_JSON
        _require(type(item.size) is int and 0<item.size<=limit,'MODEL_ASSETS: remote asset size exceeds the local bound.')
    destination.mkdir(mode=0o700)
    for name in sorted(files):
        source=hf_hub_download(generator['model'],name,revision=generator['revision'],local_files_only=False)
        limit=MAX_WEIGHTS if name=='model.safetensors' else MAX_ASSET_JSON
        fd=os.open(destination/name,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
        with open(source,'rb') as incoming,os.fdopen(fd,'wb') as output:
            total=0
            while chunk:=incoming.read(1024*1024):
                total+=len(chunk)
                _require(total<=limit,'MODEL_ASSETS: downloaded asset exceeded its bound.')
                output.write(chunk)
            _require(total==files[name].size,'MODEL_ASSETS: downloaded size differs from pinned metadata.')
            output.flush()
            os.fsync(output.fileno())
    fd=os.open(destination,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    return verify_assets(destination)


def _environment():
    return {'python':platform.python_version(),
            'dependencies':{n:version(n) for n in ('torch','transformers','tokenizers','huggingface-hub')},
            'source_hashes':{n:hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest()
                             for n in ('synthetic_cli.py','synthetic_worker.py')}}


def _prepared_record(root,identifier,protocol):
    with synthetic._directory(root,identifier) as fd:
        record=synthetic._read(fd,'prepared.json')
    synthetic._envelope(record,'synthetic-generator-prepared',{'preparation_id','protocol_id','protocol_digest',
                         'generator','threads','assets','environment','resources'})
    _require(record['preparation_id']==identifier and record['protocol_id']==protocol['protocol_id'] and
             record['protocol_digest']==protocol['content_digest'],'MODEL_PREPARED: snapshot belongs to a different protocol.')
    _require(synthetic._digest(record['generator'])==synthetic._digest(protocol['policy']['generator']) and type(record['threads']) is int and
             record['threads']==protocol['threads'],'MODEL_PREPARED: generator settings changed.')
    _require(record['environment']==_environment(),'MODEL_PREPARED: runtime/source changed; explicitly prepare with this environment again.')
    _require(synthetic._resource_status(record['resources'],protocol)=='pass','MODEL_PREPARED: readiness smoke did not pass its resource guard.')
    path=Path(root)/identifier/'assets'
    verify_assets(path,record['assets'])
    return record,path


def resolve_prepared(protocol,root,identifier=None):
    if identifier is not None:
        synthetic._id(identifier)
        return _prepared_record(root,identifier,protocol)
    if not Path(root).exists():
        raise ValueError('MODEL_NOT_PREPARED: run prepare --protocol '+protocol['protocol_id']+' before cache-only generation.')
    found=[]
    flags=os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW
    fd=os.open(root,flags)
    try:
        with os.scandir(fd) as entries:
            for index,entry in enumerate(entries):
                _require(index<50,'MODEL_LIST_LIMIT: reached 50 entries; use an explicit --prepared ID.')
                if not entry.is_dir(follow_symlinks=False) or not re_full_id(entry.name):
                    continue
                with synthetic._directory(root,entry.name) as item_fd:
                    record=synthetic._read(item_fd,'prepared.json',optional=True)
                if record is not None and isinstance(record,dict) and record.get('protocol_id')==protocol['protocol_id']:
                    found.append(entry.name)
    finally:
        os.close(fd)
    _require(len(found)==1,'MODEL_NOT_PREPARED: run prepare --protocol '+protocol['protocol_id']+
             ' or select a compatible snapshot with --prepared ID (no automatic latest selection).')
    return _prepared_record(root,found[0],protocol)


def re_full_id(value):
    import re
    return re.fullmatch('[0-9a-f]{32}',value) is not None


def _encode_message(value):
    synthetic._tree(value)
    data=json.dumps(value,ensure_ascii=False,separators=(',', ':'),allow_nan=False).encode()
    _require(len(data)<=MAX_MESSAGE,'WORKER_PROTOCOL: message exceeds 128 KiB.')
    return struct.pack('!I',len(data))+data


class WorkerFailure(RuntimeError):
    def __init__(self,state,code):
        super().__init__(code)
        self.state,self.code=state,code


def sample_rss(process):
    return _rss_pid(process.pid)


def _rss_pid(pid):
    result=subprocess.run(['/bin/ps','-o','rss=','-p',str(pid)],capture_output=True,text=True,timeout=1,check=False)
    value=result.stdout.strip()
    if result.returncode or not value.isdecimal():
        raise OSError('RSS unavailable')
    amount=int(value)*1024
    _require(0<amount<=int(1e10),'RSS measurement exceeds supported range.')
    return amount


class WorkerSession:
    """Owned child, bounded IPC and sampled stop guard; no global PID operations."""
    def __init__(self,job,limits,*,lock_fd=None,command=None,probe=sample_rss,clock=time.monotonic):
        self.limits,self.probe,self.clock=limits,probe,clock
        self.started=clock()
        self.peak,self.samples=0,0
        self.measurement_unknown=False
        self.next_sample=self.started
        self.buffer=b''
        self.sock,child=socket.socketpair()
        life_read,self.life_write=os.pipe()
        self.sock.setblocking(False)
        descriptors=[child.fileno(),life_read]+([lock_fd] if lock_fd is not None else [])
        argv=command or [sys.executable,'-m','signallab.synthetic_worker']
        try:
            self.process=subprocess.Popen([*argv,str(child.fileno()),str(life_read),str(lock_fd if lock_fd is not None else -1)],
                pass_fds=tuple(descriptors),stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,
                start_new_session=True)
        except BaseException:
            self.sock.close()
            os.close(self.life_write)
            raise
        finally:
            child.close()
            os.close(life_read)
        self.job=job

    def check(self):
        now=self.clock()
        if now-self.started>=self.limits['wall_seconds']:
            raise WorkerFailure('stopped_budget','wall_limit')
        if self.process.poll() is not None:
            raise WorkerFailure('failed','worker_exit')
        if now>=self.next_sample:
            try:
                rss=self.probe(self.process)
                _require(synthetic._integer(rss,1,int(1e10)),'RSS sample invalid')
            except (OSError,ValueError,subprocess.TimeoutExpired) as exc:
                if self.process.poll() is not None:
                    raise WorkerFailure('failed','worker_exit') from exc
                self.measurement_unknown=True
                raise WorkerFailure('failed','rss_unavailable') from exc
            self.samples+=1
            self.peak=max(self.peak,rss)
            self.next_sample=now+self.limits['rss_sample_seconds']
            if rss>=self.limits['rss_bytes']:
                raise WorkerFailure('stopped_budget','rss_limit')

    def exchange(self,message,expected):
        outgoing=memoryview(_encode_message(message))
        while True:
            self.check()
            readable,writable,_=select.select([self.sock],[self.sock] if outgoing else [],[],
                            min(0.05,self.limits['rss_sample_seconds']))
            if writable:
                try:
                    sent=self.sock.send(outgoing)
                    outgoing=outgoing[sent:]
                except (BrokenPipeError,ConnectionResetError) as exc:
                    raise WorkerFailure('failed','worker_exit') from exc
                except BlockingIOError:
                    continue
            if readable:
                try:
                    chunk=self.sock.recv(65536)
                except ConnectionResetError as exc:
                    raise WorkerFailure('failed','worker_exit') from exc
                except BlockingIOError:
                    continue
                if not chunk:
                    raise WorkerFailure('failed','worker_exit')
                self.buffer+=chunk
                if len(self.buffer)>=4:
                    size=struct.unpack('!I',self.buffer[:4])[0]
                    if not 0<size<=MAX_MESSAGE or len(self.buffer)>MAX_MESSAGE+4:
                        raise WorkerFailure('failed','invalid_output')
                    if len(self.buffer)>=size+4:
                        payload,self.buffer=self.buffer[4:size+4],self.buffer[size+4:]
                        try:
                            value=_json_bytes(payload)
                            synthetic._tree(value)
                        except ValueError as exc:
                            raise WorkerFailure('failed','invalid_output') from exc
                        if not isinstance(value,dict) or value.get('type')!=expected or self.buffer or outgoing:
                            raise WorkerFailure('failed','invalid_output')
                        return value

    def resources(self):
        result={'elapsed_seconds':max(0,self.clock()-self.started),
                'peak_rss_bytes':self.peak if self.samples and not self.measurement_unknown else None,
                'samples':self.samples if not self.measurement_unknown else 0,
                'method':'macos-ps-rss-kib-v1' if self.samples and not self.measurement_unknown else 'unavailable'}
        if self.measurement_unknown:
            result.update(observed_peak_rss_bytes=self.peak if self.samples else None,observed_samples=self.samples)
        return result

    def close(self):
        try:
            if self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=self.limits['termination_grace_seconds'])
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    try:
                        self.process.wait(timeout=self.limits['termination_grace_seconds'])
                    except subprocess.TimeoutExpired as exc:
                        raise GeneratorError('WORKER_NOT_REAPED: owned worker exit is unconfirmed. '
                            'No terminal success was published; preserve the pilot and recover only after its lock releases.') from exc
            else:
                self.process.wait()
        finally:
            self.sock.close()
            os.close(self.life_write)


def _run_worker(job,protocol,*,writer=None,parents=(),command=None,probe=sample_rss,limits=None,clock=time.monotonic,progress=None):
    limits=limits or protocol['policy']['resources']
    session=WorkerSession(job,limits,lock_fd=writer.lock_fd if writer else None,command=command,probe=probe,clock=clock)
    state,code,active='complete',None,None
    loaded=False
    try:
        if progress: progress('Loading verified local model')
        ready=session.exchange(job,'ready')
        if set(ready)!={'type'}:
            raise WorkerFailure('failed','invalid_output')
        loaded=True
        if progress: progress('Local model ready')
        for index,(parent,text) in enumerate(parents,1):
            measured=session.exchange({'type':'measure','text':text},'measured')
            if set(measured)!={'type','input_tokens'} or not synthetic._integer(measured.get('input_tokens'),1,100000):
                raise WorkerFailure('failed','invalid_output')
            count=measured['input_tokens']
            if count>128:
                writer.skip(parent,count)
                if progress: progress(f'Processed {index}/{len(parents)} parents (length skip)')
                continue
            writer.start(parent,count)
            active=parent
            result=session.exchange({'type':'generate'},'generated')
            if set(result)!={'type','text','output_tokens','eos'}:
                raise WorkerFailure('failed','invalid_output')
            try:
                writer.generated(parent,result['text'],result['output_tokens'],result['eos'])
            except ValueError as exc:
                raise WorkerFailure('failed','invalid_output') from exc
            active=None
            if progress: progress(f'Processed {index}/{len(parents)} parents')
        stopped=session.exchange({'type':'stop'},'stopped')
        if set(stopped)!={'type'}:
            raise WorkerFailure('failed','invalid_output')
        session.process.wait(timeout=limits['termination_grace_seconds'])
        if session.process.returncode!=0:
            raise WorkerFailure('failed','worker_exit')
    except WorkerFailure as exc:
        state,code=exc.state,exc.code
        if not loaded and code=='worker_exit':
            code='model_load_error'
    except subprocess.TimeoutExpired:
        state,code='failed','worker_exit'
    except KeyboardInterrupt:
        state,code='interrupted','operator_interrupt'
    finally:
        session.close()
    resources=session.resources()
    if state=='complete' and resources['elapsed_seconds']>=limits['wall_seconds']:
        state,code='stopped_budget','wall_limit'
    if active is not None and state=='failed':
        writer.failed(active,code if code in synthetic.ERROR_CODES else 'worker_error')
    return state,code,resources


def prepare_model(protocol_id,*,studies_root=study.STUDIES_ROOT,prepared_root=PREPARED_ROOT,progress=None):
    _platform_check()
    protocol=study.read_protocol(studies_root,protocol_id)
    root=Path(prepared_root)
    _require(not root.is_symlink(),'MODEL_PREPARED: root cannot be a symlink.')
    root.mkdir(mode=0o700,parents=True,exist_ok=True)
    identifier=uuid.uuid4().hex
    fd=os.open(root,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    try:
        os.mkdir(identifier,mode=0o700,dir_fd=fd)
        os.fsync(fd)
    finally:
        os.close(fd)
    folder=root/identifier
    if progress: progress(f'Preparation {identifier}: downloading allowlisted pinned assets')
    assets=_download_assets(protocol,folder/'assets')
    environment=_environment()
    job={'assets_path':str((folder/'assets').absolute()),'assets':assets,'threads':protocol['threads'],
         'generator':protocol['policy']['generator'],'environment':environment}
    state,code,resources=_run_worker(job,protocol,progress=progress)
    _require(state=='complete',f'MODEL_NOT_READY: bounded local load failed ({code}); preserve preparation {identifier} and explicitly retry prepare.')
    verify_assets(folder/'assets',assets)
    _require(_environment()==environment,'MODEL_PREPARED: code/runtime changed during preparation; explicitly retry.')
    record=synthetic._document('synthetic-generator-prepared',preparation_id=identifier,
            protocol_id=protocol_id,protocol_digest=protocol['content_digest'],generator=protocol['policy']['generator'],
            threads=protocol['threads'],assets=assets,environment=environment,resources=resources)
    with synthetic._directory(root,identifier) as fd:
        synthetic._publish(fd,'prepared.json',record)
    return record


def generate_pilot(protocol_id,frame,*,studies_root=study.STUDIES_ROOT,prepared_root=PREPARED_ROOT,
                   root=synthetic.SYNTHETIC_ROOT,prepared_id=None,progress=None,on_created=None):
    _platform_check()
    protocol=study.read_protocol(studies_root,protocol_id)
    splits=study.attach_development(protocol,frame)
    prepared,path=resolve_prepared(protocol,prepared_root,prepared_id)
    lookup={review_id(row.text,row.label):row.text for row in splits.train.itertuples()}
    parents=[(parent,lookup[parent]) for parent in protocol['selection']['parent_ids']]
    pilot_id=synthetic.create_pilot(protocol_id,frame,prepared['assets'],root=root,studies_root=studies_root,
            execution={'preparation_id':prepared['preparation_id'],'prepared_digest':prepared['content_digest'],'environment':prepared['environment']})
    if on_created: on_created(pilot_id)
    with synthetic.generation(root,pilot_id,studies_root=studies_root) as writer:
        state,code,resources=_run_worker({'assets_path':str(path.absolute()),'assets':prepared['assets'],
                'threads':protocol['threads'],'generator':protocol['policy']['generator'],'environment':prepared['environment']},
                protocol,writer=writer,parents=parents,progress=progress)
        writer.finish(state,resources,worker_reaped=True,error_code=code)
    return synthetic.read_pilot(root,pilot_id,studies_root=studies_root)


def recover_pilot(pilot_id,*,root=synthetic.SYNTHETIC_ROOT,studies_root=study.STUDIES_ROOT):
    with synthetic.generation(root,pilot_id,studies_root=studies_root) as writer:
        writer.finish('interrupted',{'elapsed_seconds':0,'peak_rss_bytes':None,'samples':0,'method':'unavailable'},worker_reaped=True)
    return synthetic.read_pilot(root,pilot_id,studies_root=studies_root)


def worker_main(channel):
    """Called only after the lightweight worker bootstrap starts its watchdog."""
    from signallab.synthetic_worker import receive,send
    job=receive(channel)
    _require(set(job)=={'assets_path','assets','threads','generator','environment'} and
             job['generator']==study.POLICY['generator'] and job['environment']==_environment(),
             'Invalid worker job.')
    tokenizer,model,settings=_load_generator(job['assets_path'],job['assets'],job['threads'])
    import torch
    send(channel,{'type':'ready'})
    encoded=None
    while True:
        message=receive(channel)
        if message=={'type':'stop'}:
            _require(job['environment']==_environment(),'Worker source/runtime changed during execution.')
            send(channel,{'type':'stopped'})
            return
        if isinstance(message,dict) and set(message)=={'type','text'} and message['type']=='measure':
            _require(isinstance(message['text'],str) and 0<len(message['text'])<=20000,'Invalid parent text.')
            prompt=job['generator']['prompt_template'].format(text=message['text'])
            encoded=tokenizer(prompt,return_tensors='pt',padding=False,truncation=False,return_token_type_ids=False)
            send(channel,{'type':'measured','input_tokens':int(encoded['input_ids'].shape[1])})
        elif message=={'type':'generate'}:
            _require(encoded is not None and encoded['input_ids'].shape[1]<=128,'Missing or overlength measured input.')
            with torch.inference_mode():
                output=model.generate(**encoded,generation_config=settings)
            tokens=output[0,1:].tolist()  # T5 output begins with its decoder-start token.
            text=tokenizer.decode(tokens,skip_special_tokens=True,clean_up_tokenization_spaces=False)
            send(channel,{'type':'generated','text':text,'output_tokens':len(tokens),'eos':settings.eos_token_id in tokens})
            encoded=None
        else:
            raise ValueError('Invalid worker message order.')


def _next_command(args,action,*tokens):
    command=['uv','run','--locked','python','-m','signallab.synthetic_cli']
    for flag,value,default in (('--studies-root',args.studies_root,study.STUDIES_ROOT),
                              ('--prepared-root',args.prepared_root,PREPARED_ROOT),
                              ('--pilots-root',args.pilots_root,synthetic.SYNTHETIC_ROOT)):
        if value.absolute()!=Path(default).absolute():
            command.extend((flag,str(value.absolute())))
    return shlex.join([*command,action,*tokens])


def _summary(result,root,studies_root):
    return synthetic.read_handoff(root,result['pilot']['pilot_id'],studies_root=studies_root)


def _print_summary(summary,json_mode):
    if json_mode:
        print(json.dumps(summary,sort_keys=True))
        return
    print(f"Pilot {summary['pilot_id']} · protocol {summary['protocol_id']}")
    print(f"Generation: {summary['generation_state']} · resources: {summary['resource_status']} · reason: {summary['termination_code'] or 'not terminal'}")
    if summary['counts'] is not None:
        print('Counts: '+json.dumps(summary['counts'],sort_keys=True))
        print('Reviews: '+json.dumps(summary['reviews'],sort_keys=True))
    print('Qualified for sealing: '+str(summary['qualified'])+' · unmet gates: '+', '.join(summary['unmet_gates']))
    print('Seals: '+json.dumps(summary['seals'],sort_keys=True))
    print('No training has run. Qualification concerns this reviewed pilot, not classifier improvement.')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__,epilog='No training runs here. Only prepare downloads; generate is cache-only. See README: Synthetic generation CLI.')
    parser.add_argument('--studies-root',type=Path,default=study.STUDIES_ROOT)
    parser.add_argument('--prepared-root',type=Path,default=PREPARED_ROOT)
    parser.add_argument('--pilots-root',type=Path,default=synthetic.SYNTHETIC_ROOT)
    commands=parser.add_subparsers(dest='action',required=True)
    freeze=commands.add_parser('freeze',help='Freeze development/final inputs before synthesis.')
    freeze.add_argument('--csv',type=Path,required=True)
    freeze.add_argument('--final-csv',type=Path,required=True)
    freeze.add_argument('--source',required=True)
    freeze.add_argument('--final-source',required=True)
    freeze.add_argument('--attest-final-reserved',action='store_true',required=True)
    freeze.add_argument('--max-parents',type=int,default=50)
    freeze.add_argument('--threads',type=int,default=2)
    prepare=commands.add_parser('prepare',help='Explicitly download pinned assets and verify bounded local readiness.')
    prepare.add_argument('--protocol',required=True)
    generate=commands.add_parser('generate',help='Run one cache-only training-parent pilot, no retries.')
    generate.add_argument('--protocol',required=True)
    generate.add_argument('--csv',type=Path,required=True)
    generate.add_argument('--prepared',help='Select an exact prepared snapshot ID, never an arbitrary checkpoint.')
    recover=commands.add_parser('recover',help='Finalize interrupted evidence only; never resume generation.')
    recover.add_argument('--pilot',required=True)
    inspect=commands.add_parser('inspect',help='Read text-free pilot status without model loading.')
    inspect.add_argument('--pilot',required=True)
    inspect.add_argument('--json',action='store_true')
    seal=commands.add_parser('seal',help='Seal a qualified current review revision; no training.')
    seal.add_argument('--pilot',required=True)
    seal.add_argument('--csv',type=Path,required=True)
    seal.add_argument('--expected-revision',type=int,required=True)
    seal.add_argument('--seal-id',help='Reuse this full request ID only for an unchanged retry.')
    seal.add_argument('--json',action='store_true')
    bundle=commands.add_parser('inspect-bundle',help='Validate a current seal and exact source; print text-free handoff counts.')
    bundle.add_argument('--pilot',required=True)
    bundle.add_argument('--seal',required=True)
    bundle.add_argument('--csv',type=Path,required=True)
    bundle.add_argument('--json',action='store_true')
    args=parser.parse_args(argv)
    progress=lambda message:print(message,file=sys.stderr)
    try:
        if args.action=='freeze':
            record=study.freeze_protocol(args.csv,args.final_csv,source=args.source,final_source=args.final_source,
                    attest_final_reserved=args.attest_final_reserved,max_parents=args.max_parents,threads=args.threads,root=args.studies_root)
            print('Protocol '+record['protocol_id']+' saved. Next: '+_next_command(args,'prepare','--protocol',record['protocol_id']))
        elif args.action=='prepare':
            record=prepare_model(args.protocol,studies_root=args.studies_root,prepared_root=args.prepared_root,progress=progress)
            print('Ready for cache-only generation. Prepared '+record['preparation_id']+' for protocol '+record['protocol_id']+
                  '. Next (replace the CSV placeholder): '+_next_command(args,'generate','--protocol',record['protocol_id'],
                   '--prepared',record['preparation_id'],'--csv','<original-development.csv>'))
        elif args.action=='seal':
            identifier=synthetic._id(args.seal_id) if args.seal_id is not None else uuid.uuid4().hex
            progress('Seal request '+identifier+'; retain this ID for an unchanged retry.')
            record=synthetic.seal_pilot(args.pilots_root,args.pilot,study.load_study_csv(args.csv),
                    expected_revision=args.expected_revision,seal_id=identifier,studies_root=args.studies_root)
            if args.json:
                print(json.dumps(record,sort_keys=True))
            else:
                print(f"Seal {record['seal_id']} saved for review revision {record['review_revision']}. No training has run.")
                print('Inspect the validated handoff: '+_next_command(args,'inspect-bundle','--pilot',args.pilot,
                      '--seal',identifier,'--csv',str(args.csv)))
        elif args.action=='inspect-bundle':
            bundle=synthetic.build_training_bundle(args.pilots_root,args.pilot,args.seal,study.load_study_csv(args.csv),studies_root=args.studies_root)
            summary=dict(schema_version=1,protocol_id=bundle.protocol_id,pilot_id=bundle.pilot_id,seal_id=bundle.seal_id,
                         review_revision=bundle.review_revision,seal_digest=bundle.seal_digest,
                         original_partitions={name:len(getattr(bundle.splits,name)) for name in ('train','validation','test')},
                         training_rows=len(bundle.training),accepted_synthetic=int((bundle.lineage.source_kind=='synthetic').sum()),
                         training_class_counts={str(label):int((bundle.training.label==label).sum()) for label in (0,1)},
                         no_training_has_run=True)
            print(json.dumps(summary,sort_keys=True,indent=None if args.json else 2))
        else:
            if args.action=='generate':
                result=generate_pilot(args.protocol,study.load_study_csv(args.csv),studies_root=args.studies_root,
                     prepared_root=args.prepared_root,root=args.pilots_root,prepared_id=args.prepared,progress=progress,
                     on_created=lambda identifier:progress('Pilot '+identifier+' allocated. If interrupted: '+
                         _next_command(args,'recover','--pilot',identifier)))
            elif args.action=='recover':
                result=recover_pilot(args.pilot,root=args.pilots_root,studies_root=args.studies_root)
            else:
                result=synthetic.read_pilot(args.pilots_root,args.pilot,studies_root=args.studies_root)
            _print_summary(_summary(result,args.pilots_root,args.studies_root),getattr(args,'json',False))
            if args.action=='generate' and result['generation_state']!='complete':
                return 1
    except (OSError,ValueError,RuntimeError) as exc:
        # Exception strings from model/Hub libraries can contain URLs or text.
        code=str(exc) if isinstance(exc,GeneratorError) else type(exc).__name__
        print(f'{args.action} failed ({code}). Existing evidence is preserved. Check inputs/permissions; see README: Synthetic generation CLI.',file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print('Interrupted. Existing evidence is preserved; use the printed pilot ID with recover. See README: Synthetic generation CLI.',file=sys.stderr)
        return 130
    return 0


if __name__=='__main__':
    raise SystemExit(main())
