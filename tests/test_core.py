import io
import unittest

import pandas as pd

from signallab.core import (
    augment_training, dataset_id, evaluate, load_csv, make_splits,
    train_baseline, validate_reviews,
)


def reviews():
    return pd.DataFrame([
        {"text": f"{'Excellent reliable' if label else 'Broken disappointing'} item number {i}", "label": label}
        for label in (0, 1) for i in range(20)
    ])


class DataContractTests(unittest.TestCase):
    def test_csv_maps_sentiment_names(self):
        frame = reviews()
        frame["label"] = frame.label.map({0: "negative", 1: "positive"})
        loaded = load_csv(io.StringIO(frame.to_csv(index=False)))
        self.assertEqual(set(loaded.label), {0, 1})

    def test_rejects_missing_invalid_and_conflicting_labels(self):
        for mutation in (None, "neutral", 0.5):
            frame = reviews()
            frame["label"] = frame.label.astype(object)
            frame.loc[0, "label"] = mutation
            with self.assertRaises(ValueError):
                validate_reviews(frame)
        frame = pd.concat([reviews(), pd.DataFrame([{"text": " EXCELLENT reliable item number 0 ", "label": 0}])])
        with self.assertRaisesRegex(ValueError, "conflicting"):
            validate_reviews(frame)

    def test_rejects_missing_text_and_small_classes(self):
        frame = reviews()
        frame.loc[0, "text"] = None
        with self.assertRaisesRegex(ValueError, "text"):
            validate_reviews(frame)
        with self.assertRaisesRegex(ValueError, "each class"):
            validate_reviews(reviews().iloc[:22])

    def test_deduplicates_before_split_and_is_reproducible(self):
        frame = reviews()
        frame = pd.concat([frame, frame.iloc[:5]])
        clean = validate_reviews(frame)
        self.assertEqual(len(clean), 40)
        a, b = make_splits(clean), make_splits(clean)
        sets = [set(part.text) for part in (a.train, a.validation, a.test)]
        self.assertFalse(sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
        self.assertEqual(sum(map(len, sets)), 40)
        self.assertEqual(a.test.text.tolist(), b.test.text.tolist())
        self.assertEqual(dataset_id(clean), dataset_id(clean.sample(frac=1)))

    def test_augmentation_uses_only_training_and_retains_labels(self):
        split = make_splits(validate_reviews(reviews()))
        augmented = augment_training(split.train, 0.5)
        self.assertEqual(len(augmented), len(split.train) + int(len(split.train) * 0.5))
        generated = augmented.iloc[len(split.train):]
        for row in generated.itertuples():
            original = row.text.removeprefix("In my experience, ")
            match = split.train[split.train.text == original]
            self.assertEqual(len(match), 1)
            self.assertEqual(row.label, match.iloc[0].label)
        self.assertFalse(set(generated.text) & set(split.test.text))
        pd.testing.assert_frame_equal(augment_training(split.train, 0), split.train)

    def test_template_variants_cannot_cross_the_split_boundary(self):
        frame = reviews()
        variant = frame.iloc[[0]].copy()
        variant["text"] = "In my experience, " + variant.text
        clean = validate_reviews(pd.concat([frame, variant]))
        self.assertEqual(len(clean), len(frame))

    def test_evaluation_reports_errors_and_both_classes(self):
        split = make_splits(validate_reviews(reviews()))
        model = train_baseline(split.train)
        report = evaluate(model, split.test)
        self.assertEqual(len(report["predictions"]), len(split.test))
        self.assertEqual(len(report["confusion_matrix"]), 2)
        self.assertGreaterEqual(report["macro_f1"], 0)
        self.assertLessEqual(report["macro_f1"], 1)
        self.assertEqual(report["n"], len(split.test))


if __name__ == "__main__":
    unittest.main()
