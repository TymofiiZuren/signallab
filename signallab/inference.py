"""CPU inference for trusted local binary BERT/DistilBERT safetensors artifacts."""
import json
from threading import RLock
from pathlib import Path

import torch
from safetensors import SafetensorError
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MAX_LENGTH = 128


def validate_batch_size(batch_size):
    if type(batch_size) is not int or not 1 <= batch_size <= 128:
        raise ValueError('Batch size must be an integer between 1 and 128. Use --batch-size 8 for the default.')


def encode_reviews(tokenizer, texts):
    return tokenizer(texts, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors='pt')


def class_probabilities(model, tokenizer, texts, batch_size=8):
    validate_batch_size(batch_size)
    texts = list(texts)
    model.eval()
    values = []
    with torch.inference_mode():
        for offset in range(0, len(texts), batch_size):
            logits = model(**encode_reviews(tokenizer, texts[offset:offset + batch_size])).logits
            if logits.ndim != 2 or logits.shape[1] != 2 or not torch.isfinite(logits).all():
                raise ValueError('Classifier returned invalid binary scores. Use a compatible trained model.')
            values.append(logits.softmax(dim=-1).cpu())
    return torch.cat(values) if values else torch.empty((0, 2))


def predict_review(model, tokenizer, text):
    if not isinstance(text, str) or not text.strip() or len(text) > 20000:
        raise ValueError('Enter a non-empty review of at most 20,000 characters.')
    text = text.strip()
    truncated = len(tokenizer.encode(text, add_special_tokens=True, truncation=False)) > MAX_LENGTH
    scores = class_probabilities(model, tokenizer, [text])[0]
    return {'label': int(scores.argmax()), 'scores': scores.tolist(), 'truncated': truncated, 'token_limit': MAX_LENGTH}


def artifact_identity(path):
    """Bound local artifacts and detect replacements before reusing a loaded model."""
    path = Path(path)
    if not path.is_dir() or path.is_symlink():
        raise ValueError('Saved model directory is missing or unsafe. Train into a new local output directory.')
    required = {'config.json', 'tokenizer_config.json', 'tokenizer.json', 'model.safetensors'}
    entries = list(path.iterdir())
    if len(entries) > 20 or not required.issubset({entry.name for entry in entries}):
        raise ValueError('Saved model needs local config, fast tokenizer and model.safetensors files. Rerun training into a new directory.')
    identity = []
    for entry in sorted(entries):
        if entry.is_symlink() or not entry.is_file():
            raise ValueError('Saved model files must be regular local files without symlinks.')
        info = entry.stat()
        limit = 512 * 1024 * 1024 if entry.name == 'model.safetensors' else 8 * 1024 * 1024
        if info.st_size > limit:
            raise ValueError('Saved model exceeds the supported local artifact size.')
        identity.append((entry.name, info.st_ino, info.st_size, info.st_mtime_ns))
    return tuple(identity)


def load_local_classifier(path):
    path = Path(path)
    artifact_identity(path)
    try:
        config = json.loads((path/'config.json').read_text())
    except (OSError, ValueError) as exc:
        raise ValueError('Cannot read config.json. Preserve the files and choose another completed classifier.') from exc
    if not isinstance(config, dict):
        raise ValueError('config.json must contain a model configuration object.')
    if config.get('model_type') not in ('bert', 'distilbert') or config.get('id2label') != {'0':'NEGATIVE', '1':'POSITIVE'} or config.get('label2id') != {'NEGATIVE':0, 'POSITIVE':1}:
        raise ValueError('Saved classifier has an incompatible model or label mapping. Train a binary NEGATIVE/POSITIVE model using signallab.train.')
    for name, fallback, maximum in (('hidden_size',768,1024), ('dim',768,1024), ('num_hidden_layers',6,16), ('n_layers',6,16), ('intermediate_size',3072,4096), ('hidden_dim',3072,4096), ('vocab_size',30522,100000), ('max_position_embeddings',512,4096), ('num_attention_heads',12,32), ('n_heads',12,32)):
        value = config.get(name, fallback)
        if type(value) is not int or not 1 <= value <= maximum:
            raise ValueError(f'Saved model configuration exceeds the supported {name} bounds.')
    try:
        model, loading = AutoModelForSequenceClassification.from_pretrained(path, local_files_only=True, trust_remote_code=False, use_safetensors=True, output_loading_info=True)
    except (OSError, ValueError, SafetensorError) as exc:
        raise ValueError('Cannot decode model.safetensors/config.json. Preserve the files and choose another completed classifier.') from exc
    if any(loading.get(key) for key in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs')):
        raise ValueError('Saved classifier weights are incomplete or incompatible. Resave a fully trained classifier.')
    try:
        tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False, use_fast=True)
    except MemoryError:
        raise
    except Exception as exc:
        # The installed Rust fast-tokenizer decoder raises plain Exception for
        # malformed serialization. Translate only this dependency call, not inference.
        raise ValueError('Cannot decode tokenizer.json/tokenizer_config.json. Preserve the files and choose another completed classifier.') from exc
    if not tokenizer.is_fast:
        raise ValueError('A local fast tokenizer is required. Resave the tokenizer with the trained classifier.')
    model.to('cpu').eval()
    return model, tokenizer


# One shared CPU classifier at a time. The lock covers tokenizer use and cache eviction.
_prediction_lock = RLock()
_cached_classifier = None


def predict_saved(path, identity, text):
    global _cached_classifier
    path = Path(path)
    with _prediction_lock:
        if artifact_identity(path) != identity:
            raise ValueError('Saved model changed. Select the model again before predicting.')
        key = (str(path.resolve()), identity)
        if _cached_classifier is None or _cached_classifier[0] != key:
            _cached_classifier = None
            model, tokenizer = load_local_classifier(path)
            if artifact_identity(path) != identity:
                raise ValueError('Saved model changed while loading. Retry after training finishes.')
            _cached_classifier = (key, model, tokenizer)
        result = predict_review(_cached_classifier[1], _cached_classifier[2], text)
        if artifact_identity(path) != identity:
            _cached_classifier = None
            raise ValueError('Saved model changed during prediction. Retry with the completed model.')
        return result
