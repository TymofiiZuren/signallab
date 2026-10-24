"""Actual offline model loading and evaluation/UI inference parity."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd
from transformers import BertConfig, BertForSequenceClassification, BertTokenizerFast

from signallab.inference import load_local_classifier, predict_review, class_probabilities
from signallab.train import score

ROOT = Path(__file__).resolve().parents[1]


class InferenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=ROOT / '.scratch')
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)
        vocab = self.path / 'vocab.txt'
        vocab.write_text('\n'.join(['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]', 'good', 'bad', 'product']))
        self.tokenizer = BertTokenizerFast(vocab_file=str(vocab))
        self.tokenizer.save_pretrained(self.path)
        self.model = BertForSequenceClassification(BertConfig(vocab_size=len(self.tokenizer), hidden_size=16, num_hidden_layers=1, num_attention_heads=2, intermediate_size=24, num_labels=2, id2label={0:'NEGATIVE',1:'POSITIVE'}, label2id={'NEGATIVE':0,'POSITIVE':1}))
        self.model.save_pretrained(self.path, safe_serialization=True)

    def test_saved_prediction_matches_evaluation_without_network(self):
        with patch('socket.socket.connect', side_effect=AssertionError('Network access forbidden')):
            loaded, tokenizer = load_local_classifier(self.path)
            result = predict_review(loaded, tokenizer, 'good product')
            frame = pd.DataFrame({'text':['good product'], 'label':[1]})
            report = score(loaded, frame, tokenizer, 1)
            self.assertEqual(result['label'], report['predictions'][0])
            self.assertAlmostEqual(sum(result['scores']), 1, places=6)
            self.assertEqual(len(result['scores']), 2)
            self.assertFalse(loaded.training)

    def test_text_validation_and_truncation_notice(self):
        for text in ('', '  ', 'x'*20001):
            with self.assertRaises(ValueError):
                predict_review(self.model, self.tokenizer, text)
        result = predict_review(self.model, self.tokenizer, 'good '*200)
        self.assertTrue(result['truncated'])
        self.assertEqual(result['token_limit'], 128)

    def test_incompatible_labels_and_missing_safe_weights_rejected(self):
        config = json.loads((self.path/'config.json').read_text())
        config['id2label'] = {'0':'POSITIVE','1':'NEGATIVE'}
        (self.path/'config.json').write_text(json.dumps(config))
        with self.assertRaisesRegex(ValueError, 'label'):
            load_local_classifier(self.path)
        with self.assertRaises(ValueError):
            load_local_classifier(self.path/'missing')

    def test_missing_classifier_weights_are_not_randomly_initialised(self):
        from safetensors.torch import save_file
        state = {name: tensor for name, tensor in self.model.state_dict().items() if not name.startswith("classifier.")}
        save_file(state, str(self.path / "model.safetensors"), metadata={"format":"pt"})
        with self.assertRaisesRegex(ValueError, "weights"):
            load_local_classifier(self.path)

    def test_saved_cache_reuses_one_model_and_rejects_changed_artifacts(self):
        from signallab.inference import artifact_identity, predict_saved
        identity = artifact_identity(self.path)
        with patch("signallab.inference._cached_classifier", None):
            with patch("signallab.inference.load_local_classifier", wraps=load_local_classifier) as loader:
                predict_saved(self.path, identity, "good")
                predict_saved(self.path, identity, "bad")
                self.assertEqual(loader.call_count, 1)
                (self.path / "new-file.txt").write_text("changed")
                with self.assertRaisesRegex(ValueError, "changed"):
                    predict_saved(self.path, identity, "good")

    def test_batched_probabilities_keep_class_order_and_input_order(self):
        values = class_probabilities(self.model, self.tokenizer, ['good', 'bad'], batch_size=1)
        self.assertEqual(tuple(values.shape), (2,2))
        for i, text in enumerate(['good', 'bad']):
            result = predict_review(self.model, self.tokenizer, text)
            self.assertAlmostEqual(result['scores'][0], float(values[i,0]), places=6)

    def test_malformed_local_files_are_recoverable(self):
        originals = {name: (self.path / name).read_bytes() for name in ("config.json", "model.safetensors", "tokenizer.json")}
        for name, content in (("config.json", b"[]"), ("config.json", b"null"), ("config.json", b"{"), ("model.safetensors", b"broken"), ("tokenizer.json", b"{")):
            with self.subTest(name=name, content=content):
                (self.path / name).write_bytes(content)
                try:
                    with self.assertRaises(ValueError):
                        load_local_classifier(self.path)
                    self.assertEqual((self.path / name).read_bytes(), content)
                finally:
                    (self.path / name).write_bytes(originals[name])
