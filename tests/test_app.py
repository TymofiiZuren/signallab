import unittest
import tempfile
from unittest.mock import patch

from signallab.demo import demo_reviews
from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "app.py"


class AppFlowTests(unittest.TestCase):
    def test_explore_train_predict_and_export(self):
        app = AppTest.from_file(str(APP), default_timeout=30).run()
        self.assertFalse(app.exception)
        self.assertIn("Explore reviews", [h.value for h in app.title])
        app.sidebar.radio[0].set_value("Experiments").run()
        self.assertFalse(app.exception)
        next(b for b in app.button if b.label == "Run baseline comparison").click().run()
        self.assertFalse(app.exception)
        self.assertTrue(app.success)
        self.assertTrue(app.dataframe)
        app.sidebar.radio[0].set_value("Predict").run()
        app.text_area[0].set_value("The product is excellent and works perfectly.")
        next(b for b in app.button if b.label == "Analyse review").click().run()
        self.assertFalse(app.exception)
        self.assertTrue(app.success)
        app.text_area[0].set_value("   ")
        next(b for b in app.button if b.label == "Analyse review").click().run()
        self.assertTrue(app.error)
        self.assertFalse(app.success, "Invalid submissions must not retain the previous prediction")
        app.sidebar.radio[0].set_value("Methodology").run()
        self.assertFalse(app.exception)

    def test_dataset_change_discards_models_and_pending_upload_is_clear(self):
        app = AppTest.from_file(str(APP), default_timeout=30).run()
        app.sidebar.radio[0].set_value("Experiments").run()
        next(b for b in app.button if b.label == "Run baseline comparison").click().run()
        self.assertIn("models", app.session_state)
        if "Local review dataset" in app.sidebar.selectbox[0].options:
            app.sidebar.selectbox[0].set_value("Local review dataset").run()
            self.assertNotIn("models", app.session_state)
        app.sidebar.selectbox[0].set_value("Upload CSV").run()
        self.assertFalse(app.exception)
        self.assertTrue(app.info)
        self.assertIn("Bring your reviews", [h.value for h in app.title])

    def test_analysis_validation_slices_and_comparison(self):
        app = AppTest.from_file(str(APP), default_timeout=30).run()
        app.sidebar.radio[0].set_value("Analysis").run()
        self.assertFalse(app.exception)
        self.assertTrue(app.info)
        app.sidebar.radio[0].set_value("Experiments").run()
        next(b for b in app.button if b.label == "Run baseline comparison").click().run()
        app.sidebar.radio[0].set_value("Analysis").run()
        self.assertFalse(app.exception)
        self.assertIn("Investigate model errors", [h.value for h in app.title])
        self.assertEqual(next(s for s in app.selectbox if s.label == "Evaluation partition").value, "validation")
        self.assertGreaterEqual(len(app.dataframe), 3)
        next(s for s in app.selectbox if s.label == "Show reviews").set_value("Model disagreements").run()
        self.assertFalse(app.exception)

    def test_saved_history_survives_fresh_session_and_missing_upload(self):
        with tempfile.TemporaryDirectory(dir=APP.parent / ".scratch") as folder:
            with patch("signallab.runs.RUNS_ROOT", Path(folder) / "runs"):
                app = AppTest.from_file(str(APP), default_timeout=30).run()
                app.sidebar.radio[0].set_value("Experiments").run()
                next(b for b in app.button if b.label == "Run baseline comparison").click().run()
                next(b for b in app.button if b.label == "Save experiment").click().run()
                self.assertFalse(app.exception)
                self.assertTrue(list((Path(folder) / "runs").glob("*/report.json")))
                fresh = AppTest.from_file(str(APP), default_timeout=30).run()
                fresh.sidebar.radio[0].set_value("Saved runs").run()
                next(b for b in fresh.button if b.label == "Refresh saved runs").click().run()
                self.assertFalse(fresh.exception)
                self.assertTrue(fresh.dataframe)
                self.assertTrue(any("Authored demo results" in warning.value for warning in fresh.warning))
                fresh.sidebar.selectbox[0].set_value("Upload CSV").run()
                self.assertFalse(fresh.exception)
                self.assertIn("Saved experiment history", [h.value for h in fresh.title])
                fresh.sidebar.selectbox[0].set_value("Authored demo").run()
                fresh.sidebar.radio[0].set_value("Analysis").run()
                next(s for s in fresh.selectbox if s.label == "Analysis source").set_value("Saved experiment").run()
                self.assertFalse(fresh.exception)
                self.assertGreaterEqual(len(fresh.dataframe), 3)
                self.assertNotIn("models", fresh.session_state)
                fresh.sidebar.radio[0].set_value("Predict").run()
                self.assertFalse(fresh.exception)
                self.assertTrue(fresh.info)

    def test_save_failure_keeps_session_results(self):
        with patch("signallab.runs.save_record", side_effect=OSError("disk unavailable")):
            app = AppTest.from_file(str(APP), default_timeout=30).run()
            app.sidebar.radio[0].set_value("Experiments").run()
            next(b for b in app.button if b.label == "Run baseline comparison").click().run()
            next(b for b in app.button if b.label == "Save experiment").click().run()
            self.assertFalse(app.exception)
            self.assertTrue(app.error)
            self.assertIn("report", app.session_state)
            self.assertIn("models", app.session_state)

    def test_case_only_input_change_invalidates_session_models(self):
        frame = demo_reviews()
        with patch("signallab.demo.demo_reviews", return_value=frame):
            app = AppTest.from_file(str(APP), default_timeout=30).run()
            app.sidebar.radio[0].set_value("Experiments").run()
            next(b for b in app.button if b.label == "Run baseline comparison").click().run()
            self.assertIn("models", app.session_state)
        changed = frame.copy()
        changed.loc[0, "text"] = changed.loc[0, "text"].upper()
        with patch("signallab.demo.demo_reviews", return_value=changed):
            app.run()
        self.assertFalse(app.exception)
        self.assertNotIn("models", app.session_state)
        self.assertNotIn("report", app.session_state)


    def test_save_retains_completed_ratio_after_slider_changes(self):
        with tempfile.TemporaryDirectory(dir=APP.parent / ".scratch") as folder:
            with patch("signallab.runs.RUNS_ROOT", Path(folder) / "runs"):
                app = AppTest.from_file(str(APP), default_timeout=30).run()
                app.sidebar.radio[0].set_value("Experiments").run()
                next(b for b in app.button if b.label == "Run baseline comparison").click().run()
                app.slider[0].set_value(1.0).run()
                next(b for b in app.button if b.label == "Save experiment").click().run()
                self.assertFalse(app.exception)
                import json
                record = json.loads(next((Path(folder) / "runs").glob("*/report.json")).read_text())
                self.assertEqual(record["augmentation"]["ratio"], .5)
                self.assertTrue(app.warning)

    def test_old_session_report_gets_rerun_guidance_before_save(self):
        app = AppTest.from_file(str(APP), default_timeout=30).run()
        app.sidebar.radio[0].set_value("Experiments").run()
        next(b for b in app.button if b.label == "Run baseline comparison").click().run()
        app.session_state["report"].pop("input_id")
        next(b for b in app.button if b.label == "Save experiment").click().run()
        self.assertFalse(app.exception)
        self.assertTrue(any("Run a new baseline comparison" in error.value for error in app.error))
        self.assertIn("models", app.session_state)

    def test_saved_prediction_does_not_need_training_or_upload(self):
        with patch("signallab.prediction_ui.saved_model_context", return_value={"identity": (("model", 1, 1, 1),), "dataset_id": "a"*64, "checkpoint": "local-test", "ratio": .5, "seed": 42}):
            with patch("signallab.inference.predict_saved", return_value={"label": 1, "scores": [.1,.9], "truncated": True, "token_limit": 128}):
                app = AppTest.from_file(str(APP), default_timeout=30).run()
                app.sidebar.radio[0].set_value("Predict").run()
                next(r for r in app.radio if r.label == "Prediction source").set_value("Saved transformer").run()
                app.sidebar.selectbox[0].set_value("Upload CSV").run()
                self.assertFalse(app.exception)
                self.assertTrue(app.text_area)
                app.text_area[0].set_value("A good product.")
                next(b for b in app.button if b.label == "Analyse review").click().run()
                self.assertFalse(app.exception)
                self.assertTrue(app.success)
                self.assertTrue(any("128" in warning.value for warning in app.warning))
                next(s for s in app.selectbox if s.label == "Saved model").set_value("augmented").run()
                self.assertFalse(app.success, "Switching models must clear the prior prediction")
                self.assertNotIn("models", app.session_state)

    def test_saved_prediction_load_failure_clears_old_result(self):
        with patch("signallab.prediction_ui.saved_model_context", return_value={"identity": (("model", 1, 1, 1),), "dataset_id": "a"*64, "checkpoint": "local-test", "ratio": .5, "seed": 42}):
            with patch("signallab.inference.predict_saved", side_effect=ValueError("incompatible classifier")):
                app = AppTest.from_file(str(APP), default_timeout=30).run()
                app.sidebar.radio[0].set_value("Predict").run()
                next(r for r in app.radio if r.label == "Prediction source").set_value("Saved transformer").run()
                app.text_area[0].set_value("A good product.")
                next(b for b in app.button if b.label == "Analyse review").click().run()
                self.assertFalse(app.exception)
                self.assertTrue(app.error)
                self.assertFalse(app.success)

    def test_prediction_requires_training(self):
        app = AppTest.from_file(str(APP), default_timeout=30).run()
        app.sidebar.radio[0].set_value("Predict").run()
        self.assertTrue(app.info)
        self.assertFalse(app.button)

    def test_navigation_never_displays_another_prediction_source(self):
        with patch("signallab.prediction_ui.saved_model_context", return_value={"identity": (("model",1,1,1),), "dataset_id": "a"*64, "checkpoint": "local", "ratio": .5, "seed":42}):
            with patch("signallab.inference.predict_saved", return_value={"label":1,"scores":[.1,.9],"truncated":False,"token_limit":128}):
                app = AppTest.from_file(str(APP), default_timeout=30).run()
                app.sidebar.radio[0].set_value("Experiments").run()
                next(b for b in app.button if b.label == "Run baseline comparison").click().run()
                app.sidebar.radio[0].set_value("Predict").run()
                next(r for r in app.radio if r.label == "Prediction source").set_value("Saved transformer").run()
                app.text_area[0].set_value("A good product.")
                next(b for b in app.button if b.label == "Analyse review").click().run()
                self.assertTrue(app.success)
                app.sidebar.radio[0].set_value("Explore").run()
                app.sidebar.radio[0].set_value("Predict").run()
                self.assertFalse(app.exception)
                self.assertFalse(app.success, "Returning to baseline must not show a saved-transformer result")

    def test_case_study_without_upload_and_analysis_partition_handoff(self):
        from signallab.runs import build_record, save_record
        from signallab.core import run_comparison
        frame = demo_reviews()
        report, _, splits = run_comparison(frame)
        report['source'] = 'Authored demo'
        with tempfile.TemporaryDirectory(dir=APP.parent / '.scratch') as folder:
            root = Path(folder)/'runs'
            record = build_record(report,frame,splits)
            save_record(record,root)
            with patch('signallab.runs.RUNS_ROOT',root):
                app=AppTest.from_file(str(APP),default_timeout=30).run()
                app.sidebar.selectbox[0].set_value('Upload CSV').run()
                app.sidebar.radio[0].set_value('Case Study').run()
                self.assertFalse(app.exception)
                next(b for b in app.button if b.label == 'Refresh saved runs').click().run()
                self.assertFalse(app.exception)
                self.assertEqual(len(app.get('download_button')),1)
                self.assertNotIn('models',app.session_state)
                self.assertTrue(any('Authored demo' in w.value for w in app.warning))
                next(s for s in app.selectbox if s.label == 'Evaluation partition').set_value('test').run()
                self.assertTrue(any('Repeated inspection' in w.value for w in app.warning))
                next(s for s in app.selectbox if s.label == 'Original review source').set_value('Authored demo').run()
                next(t for t in app.toggle if t.label == 'Attach original reviews').set_value(True).run()
                self.assertFalse(app.exception)
                self.assertTrue(app.text)
                app.sidebar.selectbox[0].set_value('Authored demo').run()
                next(b for b in app.button if b.label == 'Inspect this partition in Analysis').click().run()
                self.assertFalse(app.exception)
                self.assertEqual(app.session_state['saved_analysis_run'],record['run_id'])
                self.assertEqual(next(s for s in app.selectbox if s.label=='Evaluation partition').value,'test')

    def test_case_study_replaced_removed_record_and_failed_attachment(self):
        import json
        from signallab.runs import build_record,save_record
        from signallab.core import run_comparison
        frame=demo_reviews()
        report,_,splits=run_comparison(frame)
        record=build_record(report,frame,splits)
        with tempfile.TemporaryDirectory(dir=APP.parent/'.scratch') as folder:
            root=Path(folder)/'runs';save_record(record,root)
            with patch('signallab.runs.RUNS_ROOT',root):
                app=AppTest.from_file(str(APP),default_timeout=30).run()
                app.sidebar.radio[0].set_value('Case Study').run()
                next(b for b in app.button if b.label=='Refresh saved runs').click().run()
                before=app.get('download_button')[0].proto.url
                record['manifest']['python']='other-runtime'
                path=root/record['run_id']/'report.json'
                path.write_text(json.dumps(record))
                app.run()
                self.assertFalse(app.exception)
                self.assertNotEqual(before,app.get('download_button')[0].proto.url)
                with patch('signallab.case_study_ui.case_study_examples',side_effect=ValueError('Exact original dataset required')):
                    next(t for t in app.toggle if t.label=='Attach original reviews').set_value(True).run()
                    self.assertTrue(app.error)
                    self.assertEqual(len(app.get('download_button')),1)
                with patch('signallab.runs.read_record', side_effect=FileNotFoundError('removed')):
                    app.run()
                    self.assertTrue(app.error)
                    self.assertFalse(app.get('download_button'))
                path.write_text('[]')
                app.run()
                self.assertFalse(app.exception)
                self.assertTrue(app.error)
                self.assertEqual(len(app.get('download_button')),0)
                self.assertFalse(app.dataframe)

    def test_case_study_empty_and_unreadable_history(self):
        with patch('signallab.runs.discover_runs',return_value=([],False)):
            app=AppTest.from_file(str(APP),default_timeout=30).run()
            app.sidebar.radio[0].set_value('Case Study').run()
            next(b for b in app.button if b.label=='Refresh saved runs').click().run()
            self.assertFalse(app.exception)
            self.assertFalse(app.get('download_button'))

            self.assertTrue(app.info)
        with patch('signallab.runs.discover_runs',side_effect=OSError('denied')):
            next(b for b in app.button if b.label=='Refresh saved runs').click().run()
            self.assertFalse(app.exception)
            self.assertTrue(app.error)
            self.assertFalse(app.get('download_button'))

    def test_case_study_uploaded_csv_parser_keeps_numeric_story_on_failure(self):
        from io import BytesIO
        from signallab.core import run_comparison
        from signallab.runs import build_record, save_record
        frame = demo_reviews()
        report, _, splits = run_comparison(frame)
        changed = frame.copy()
        changed.loc[0, 'text'] = changed.loc[0, 'text'].upper()
        cases = (
            ('matching', frame.to_csv(index=False).encode(), None, False),
            ('mismatch', changed.to_csv(index=False).encode(), None, True),
            ('malformed', b'wrong,columns\n1,2\n', None, True),
            ('oversize', b'', 10*1024*1024+1, True),
        )
        with tempfile.TemporaryDirectory(dir=APP.parent/'.scratch') as folder:
            root = Path(folder)/'runs'
            save_record(build_record(report, frame, splits), root)
            with patch('signallab.runs.RUNS_ROOT', root):
                for name, data, size, should_error in cases:
                    with self.subTest(name=name):
                        uploaded = BytesIO(data)
                        uploaded.size = len(data) if size is None else size
                        app = AppTest.from_file(str(APP), default_timeout=30).run()
                        app.sidebar.radio[0].set_value('Case Study').run()
                        next(b for b in app.button if b.label == 'Refresh saved runs').click().run()
                        next(s for s in app.selectbox if s.label == 'Original review source').set_value('Upload CSV').run()
                        with patch('signallab.case_study_ui.st.file_uploader', return_value=uploaded):
                            next(t for t in app.toggle if t.label == 'Attach original reviews').set_value(True).run()
                        self.assertFalse(app.exception)
                        self.assertEqual(bool(app.error), should_error)
                        self.assertEqual(len(app.get('download_button')), 1)
                        self.assertEqual(bool(app.text), not should_error)

    def test_actual_corrupt_classifier_shows_recovery_without_old_result(self):
        import json
        from transformers import BertConfig, BertForSequenceClassification, BertTokenizerFast
        with tempfile.TemporaryDirectory(dir=APP.parent/'.scratch') as folder:
            root=Path(folder); model_path=root/'real';model_path.mkdir()
            vocab=root/'vocab.txt'
            vocab.write_text('\n'.join(['[PAD]','[UNK]','[CLS]','[SEP]','[MASK]','good','product']))
            tokenizer=BertTokenizerFast(vocab_file=str(vocab));tokenizer.save_pretrained(model_path)
            model=BertForSequenceClassification(BertConfig(vocab_size=len(tokenizer),hidden_size=16,num_hidden_layers=1,num_attention_heads=2,intermediate_size=24,num_labels=2,id2label={0:'NEGATIVE',1:'POSITIVE'},label2id={'NEGATIVE':0,'POSITIVE':1}))
            model.save_pretrained(model_path,safe_serialization=True)
            (root/'report.json').write_text(json.dumps({'schema_version':1,'training':{'max_length':128},'dataset_id':'a'*64,'checkpoint':'local','augmentation':{'ratio':.5},'seed':42}))
            originals={name:(model_path/name).read_bytes() for name in ('config.json','model.safetensors','tokenizer.json')}
            with patch('signallab.prediction_ui.TRANSFORMER_ROOT',root),patch('signallab.inference._cached_classifier',None):
                app=AppTest.from_file(str(APP),default_timeout=30).run()
                app.sidebar.radio[0].set_value('Predict').run()
                next(r for r in app.radio if r.label=='Prediction source').set_value('Saved transformer').run()
                for name,value in (('config.json',b'[]'),('model.safetensors',b'broken'),('tokenizer.json',b'{')):
                    with self.subTest(file=name):
                        (model_path/name).write_bytes(value)
                        app.run()
                        app.text_area[0].set_value('good product')
                        next(b for b in app.button if b.label=='Analyse review').click().run()
                        self.assertFalse(app.exception)
                        self.assertTrue(app.error)
                        self.assertFalse(app.success)
                        self.assertEqual((model_path/name).read_bytes(),value)
                        (model_path/name).write_bytes(originals[name])


if __name__ == "__main__":
    unittest.main()
