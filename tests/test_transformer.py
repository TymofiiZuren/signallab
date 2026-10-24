"""Exercise real optimisation/checkpoint/report paths offline with a tiny BERT."""
import tempfile
import unittest
from pathlib import Path

from transformers import BertConfig, BertForSequenceClassification, BertTokenizerFast

from signallab.demo import demo_reviews
from signallab.train import run_transformer

ROOT = Path(__file__).resolve().parents[1]


class TransformerContractTests(unittest.TestCase):
    def test_two_runs_save_models_and_evaluate_same_holdout(self):
        with tempfile.TemporaryDirectory(dir=ROOT / ".scratch") as folder:
            base = Path(folder)
            checkpoint = base / "tiny-bert"
            checkpoint.mkdir()
            vocab = base / "vocab.txt"
            vocab.write_text("\n".join(["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "the", "is", "good", "bad", "product", "in", "my", "experience"]))
            tokenizer = BertTokenizerFast(vocab_file=str(vocab))
            tokenizer.save_pretrained(checkpoint)
            model = BertForSequenceClassification(BertConfig(vocab_size=len(tokenizer), hidden_size=16, num_hidden_layers=1, num_attention_heads=2, intermediate_size=24, num_labels=2))
            model.save_pretrained(checkpoint)
            report = run_transformer(demo_reviews(), base / "run", str(checkpoint), epochs=1, batch_size=12)
            self.assertEqual(len(report["runs"]), 2)
            self.assertEqual(report["split_sizes"], {"train": 36, "validation": 12, "test": 12})
            self.assertEqual(report["runs"][1]["training_rows"], 54)
            self.assertEqual(report["runs"][0]["test"]["n"], 12)
            for partition in ("validation", "test"):
                evidence = [run[partition]["evidence"] for run in report["runs"]]
                self.assertEqual([r["review_id"] for r in evidence[0]], [r["review_id"] for r in evidence[1]])
                self.assertEqual(len(evidence[0]), 12)
                self.assertNotIn("text", evidence[0][0])
            self.assertTrue((base / "run/real/model.safetensors").exists())
            self.assertTrue((base / "run/report.json").exists())
            with self.assertRaises(FileExistsError):
                run_transformer(demo_reviews(), base / "run", str(checkpoint), epochs=1)


class BatchPreflightTests(unittest.TestCase):
    def test_invalid_batch_rejected_before_output_or_model_loading(self):
        from unittest.mock import patch
        from uuid import uuid4
        for batch in (0, -1, 129, True, 1.5, "8"):
            with self.subTest(batch=batch):
                output = ROOT / ".scratch" / ("invalid-batch-" + uuid4().hex)
                with patch("signallab.train.fit_model", side_effect=AssertionError("Training started")) as fit:
                    with self.assertRaisesRegex(ValueError, "Batch size"):
                        run_transformer(demo_reviews(), output, epochs=1, batch_size=batch)
                    fit.assert_not_called()
                    self.assertFalse(output.exists())

    def test_direct_fit_uses_same_preflight_and_valid_boundaries(self):
        from unittest.mock import patch
        from signallab.train import fit_model
        from signallab.inference import validate_batch_size
        for batch in (1, 128):
            validate_batch_size(batch)
        for batch in (129, False, "8"):
            with patch("signallab.train.AutoTokenizer.from_pretrained") as loader:
                with self.assertRaisesRegex(ValueError, "Batch size"):
                    fit_model(demo_reviews(), demo_reviews(), "unused", batch_size=batch)
                loader.assert_not_called()
