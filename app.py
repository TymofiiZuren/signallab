"""Run with: uv run streamlit run app.py"""
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from signallab.core import LABELS, dataset_id, load_csv, run_comparison, validate_reviews
from signallab import runs as saved_runs
from signallab.demo import demo_reviews

ROOT = Path(__file__).resolve().parent
COLORS = {"Negative": "#F6821F", "Positive": "#8CB5F6"}

def refresh_saved_runs():
    try:
        with st.spinner("Reading local saved experiments…"):
            entries, truncated = saved_runs.discover_runs()
        st.session_state.update(saved_run_entries=entries, saved_runs_truncated=truncated)
    except (OSError, ValueError):
        st.session_state.pop("saved_run_entries", None)
        st.error("Cannot read saved experiments. Check permissions on artifacts/runs, then refresh again.")


def saved_run_label(entry):
    saved_at = datetime.fromisoformat(entry["created_at"]).astimezone(timezone.utc)
    return f"{saved_at:%d %b %Y %H:%M} UTC · {entry['run_id'][:8]}"


def open_saved_analysis(run_id, partition="validation"):
    st.session_state.update(workspace="Analysis", analysis_source="Saved experiment", saved_analysis_run=run_id, analysis_partition=partition)


st.set_page_config(page_title="SignalLab · sentiment experiments", page_icon=":material/analytics:", layout="wide")
st.html(ROOT / "static/workspace.css")
with st.sidebar:
    st.html('<div class="signal-brand"><span class="signal-mark" aria-hidden="true"><i></i><i></i><i></i></span><span>SignalLab</span></div>')
st.sidebar.caption("Customer review research lab")
page = st.sidebar.radio("Workspace", ["Explore", "Experiments", "Synthetic Review Lab", "Analysis", "Saved runs", "Case Study", "Predict", "Methodology"], key="workspace")
st.sidebar.divider()
options = ["Authored demo", "Upload CSV"]
if (ROOT / "data/reviews.csv").exists():
    options.append("Local review dataset")
source = st.sidebar.selectbox("Dataset", options)
if page != "Synthetic Review Lab":
    for key in ("synthetic_editor", "synthetic_notice", "synthetic_attachment_source", "synthetic_next", "synthetic_follow_pending",
                "synthetic_conflict_draft", "synthetic_reload_candidate", "synthetic_review_cache",
                "synthetic_seal_request", "synthetic_seal_notice"):
        st.session_state.pop(key, None)
if page == "Synthetic Review Lab":
    from signallab.synthetic_ui import render_synthetic_lab

    render_synthetic_lab()
    st.divider()
    st.caption("SignalLab · Local synthetic evidence and explicit human review.")
    st.stop()

# Saved summaries do not depend on a dataset or an uploaded file being present.
if page == "Case Study":
    from signallab.case_study_ui import render_case_study

    render_case_study(refresh_saved_runs, saved_run_label, open_saved_analysis)
    st.divider()
    st.caption("SignalLab · Local exploratory evidence · Review text stays out of Markdown.")
    st.stop()

if page == "Saved runs":
    with st.container(key="page_intro"):
        st.title("Saved experiment history")
        st.write("Reopen measured results without retraining. Attach the original dataset when you want to inspect review text.")
    st.caption("Local baseline snapshots · numeric evidence and review IDs · no saved models or review text")
    if st.button("Refresh saved runs", type="primary"):
        refresh_saved_runs()
    entries = st.session_state.get("saved_run_entries")
    if entries is None:
        st.info("Choose Refresh saved runs to browse local snapshots. Save a completed comparison in Experiments to create your first one.")
    elif not entries:
        st.info("No saved experiments found. Run a baseline comparison in Experiments, then choose Save experiment.")
    else:
        st.dataframe(pd.DataFrame([{"Status": entry["status"], "Saved at": saved_run_label(entry).split(" · ")[0] if entry["created_at"] else "—",
                                    "Run": entry["run_id"][:8], "Exact dataset": entry["input_id"][:8], "Details": entry["detail"]}
                                   for entry in entries]), hide_index=True, width="stretch")
        if st.session_state.get("saved_runs_truncated"):
            st.warning("Refresh reached the 50-entry scan limit. This is a bounded local listing, not a complete or globally newest history.")
        ready = [entry for entry in entries if entry["status"] == "Ready"]
        if ready:
            selected_id = st.selectbox("Saved experiment", [entry["run_id"] for entry in ready],
                                       format_func=lambda value: saved_run_label(next(entry for entry in ready if entry["run_id"] == value)))
            try:
                record = saved_runs.read_record(saved_runs.RUNS_ROOT, selected_id)
                st.caption(f"Run {selected_id} · seed {record['seed']} · completed synthetic ratio {record['augmentation']['ratio']:.0%}")
                recorded_source = record["manifest"].get("data_source", "Source not recorded")
                st.caption(f"Recorded dataset source: {recorded_source}")
                if recorded_source == "Authored demo":
                    st.warning("Authored demo results illustrate the workflow; they are not evidence of real-world performance.")
                elif recorded_source == "Source not recorded":
                    st.warning("Dataset source was not recorded in this snapshot. Check its provenance before treating scores as research evidence.")
                partition = st.selectbox("Saved evaluation partition", ["validation", "test"])
                if partition == "test":
                    st.warning("Exploratory test results. Repeated inspection makes this a development set; reserve fresh reviews for final claims.")
                st.dataframe(pd.DataFrame([{"Model": run["name"], "Reviews": run[partition]["n"], "Macro F1": run[partition]["macro_f1"], "Accuracy": run[partition]["accuracy"]} for run in record["runs"]]), hide_index=True, width="stretch")
                st.button("Inspect in Analysis", on_click=open_saved_analysis, args=(selected_id,))
                st.caption("Analysis needs the exact original dataset. These records contain no model; Predict can use a session-trained baseline or a separately saved transformer.")
                with st.expander("Experiment provenance"):
                    st.json(record["manifest"])
                st.download_button("Download saved experiment JSON", json.dumps(record, indent=2), f"signallab-{selected_id}.json", "application/json")
            except (OSError, ValueError, TypeError, KeyError):
                st.error("This saved experiment changed or cannot be read. Refresh the list; preserve invalid files and save a new experiment.")
    st.divider()
    st.caption("SignalLab · Saved evidence stays on this computer.")
    st.stop()

if page == "Predict":
    from signallab.prediction_ui import clear_prediction, reconcile_prediction, render_saved_prediction

    with st.container(key="page_intro"):
        st.title("Try a review")
        st.write("Choose a session-trained baseline or a saved transformer, then investigate a new review.")
    prediction_source = st.radio("Prediction source", ["Session baseline", "Saved transformer"], horizontal=True,
                                 key="prediction_source", on_change=clear_prediction)
    if prediction_source == "Saved transformer":
        render_saved_prediction()
        st.divider()
        st.caption("SignalLab · Local saved-model inference · Probabilities are uncalibrated.")
        st.stop()

try:
    if source == "Upload CSV":
        uploaded = st.sidebar.file_uploader("Reviews CSV", type="csv", help="UTF-8 CSV: text,label. Labels: 0/negative or 1/positive. Maximum 10 MB.")
        if uploaded is None:
            st.title("Bring your reviews")
            st.info("Upload a CSV with text and label columns. Use at least 8 distinct reviews per class.")
            st.code('text,label\n"The headphones sound excellent.",positive\n"The battery stopped charging.",negative', language="csv")
            st.stop()
        if uploaded.size > 10 * 1024 * 1024:
            raise ValueError("CSV exceeds 10 MB. Upload a smaller file.")
        raw_count = None
        frame = load_csv(uploaded)
    elif source == "Local review dataset":
        raw_count = None
        frame = load_csv(ROOT / "data/reviews.csv")
    else:
        raw_count = len(demo_reviews())
        frame = validate_reviews(demo_reviews())
except (ValueError, OSError) as exc:
    st.title("Check your dataset")
    st.error(str(exc))
    st.stop()

fingerprint = dataset_id(frame)
input_id = saved_runs.exact_dataset_id(frame)
if st.session_state.get("input_id") != input_id:
    for key in ("report", "models", "splits", "prediction"):
        st.session_state.pop(key, None)
    st.session_state.update(dataset_id=fingerprint, input_id=input_id)
st.sidebar.caption(f"{len(frame):,} distinct reviews · dataset {fingerprint[:8]}")
st.sidebar.caption("English · binary sentiment · seed 42")
if source == "Authored demo":
    st.warning("Authored demo data — 60 illustrative reviews, not real customers. Scores demonstrate the workflow; they are not evidence of real-world performance.")
else:
    st.caption("Reviews are processed locally. Exact duplicates are removed before splitting. Check your dataset's provenance and labels before interpreting results.")

if page == "Explore":
    with st.container(key="page_intro"):
        st.title("Explore reviews")
        st.write("Understand the data before asking a model to learn from it.")
    left, right = st.columns(2)
    counts = frame.label.map(LABELS).value_counts().rename_axis("Sentiment").reset_index(name="Reviews")
    with left:
        st.subheader("Class balance")
        chart = px.bar(counts, x="Sentiment", y="Reviews", color="Sentiment", color_discrete_map=COLORS, text="Reviews")
        chart.update_layout(showlegend=False, margin=dict(t=24, b=0, l=0, r=16), height=280,
                            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                            font=dict(family="Signal Sans, sans-serif", color="#F5F2ED", size=14))
        chart.update_traces(textfont_color="#111111", marker_line_width=0)
        chart.update_yaxes(gridcolor="#333333", zeroline=False)
        st.plotly_chart(chart, width="stretch")
    with right:
        st.subheader("Review length")
        lengths = frame.assign(Words=frame.text.str.split().str.len(), Sentiment=frame.label.map(LABELS))
        chart = px.histogram(lengths, x="Words", color="Sentiment", barmode="overlay", color_discrete_map=COLORS, nbins=15, opacity=0.75)
        chart.update_layout(margin=dict(t=24, b=0, l=0, r=16), height=280, yaxis_title="Reviews",
                            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                            font=dict(family="Signal Sans, sans-serif", color="#F5F2ED", size=14),
                            legend=dict(orientation="h", y=1.2, x=0))
        chart.update_yaxes(gridcolor="#333333", zeroline=False)
        st.plotly_chart(chart, width="stretch")
    st.subheader("Read the evidence")
    a, b = st.columns([1, 2])
    sentiment = a.selectbox("Sentiment", ["All", "Positive", "Negative"])
    query = b.text_input("Find words in reviews", placeholder="Try battery, delivery, quality…")
    visible = frame.copy()
    if sentiment != "All":
        visible = visible[visible.label == (1 if sentiment == "Positive" else 0)]
    if query:
        visible = visible[visible.text.str.contains(query, case=False, regex=False)]
    st.caption(f"{len(visible):,} matching reviews")
    if visible.empty:
        st.info("No reviews match. Clear the search or change the sentiment filter.")
    else:
        st.dataframe(visible.assign(Sentiment=visible.label.map(LABELS))[["text", "Sentiment"]], hide_index=True, width="stretch")
    st.download_button("Download dataset CSV", frame.to_csv(index=False), "signallab-reviews.csv", "text/csv")

elif page == "Experiments":
    with st.container(key="page_intro"):
        st.title("Compare experiments")
        st.write("Does adding synthetic training text help on untouched reviews? Run both baselines on the same split.")
    st.info("This first experiment uses a simple control: prefix sampled training reviews with ‘In my experience, ’. It adds examples, not new information. An improvement is not guaranteed.")
    ratio = st.slider("Synthetic examples per real training example", 0.0, 1.0, 0.5, 0.25)
    st.caption("60% training · 20% validation · 20% held-out test. Rounded to whole reviews; class balance preserved. TF-IDF is fitted on training text only.")
    if st.button("Run baseline comparison", type="primary"):
        try:
            with st.spinner("Splitting reviews, fitting two models and evaluating…"):
                report, models, splits = run_comparison(frame, ratio)
            report["source"] = source
            st.session_state.update(report=report, models=models, splits=splits,
                                    training_generation=st.session_state.get("training_generation", 0) + 1)
            st.session_state.pop("prediction", None)
            st.success("Comparison complete. Both models were evaluated on the same held-out reviews.")
        except ValueError as exc:
            st.error(f"Training failed: {exc}")
    report = st.session_state.get("report")
    if report:
        if report["augmentation"]["ratio"] != ratio:
            st.warning("The slider has changed. Results below belong to the previous run; run again to update them.")
        st.subheader("Measured results")
        st.caption(f"Completed run: synthetic ratio {report['augmentation']['ratio']:.0%} · {report['split_sizes']}")
        table = pd.DataFrame([{
            "Model": r["name"], "Training rows": r["training_rows"],
            "Validation macro F1": r["validation"]["macro_f1"],
            "Test macro F1": r["test"]["macro_f1"], "Test accuracy": r["test"]["accuracy"],
        } for r in report["runs"]])
        st.dataframe(table, hide_index=True, width="stretch", column_config={key: st.column_config.NumberColumn(format="%.3f") for key in ("Validation macro F1", "Test macro F1", "Test accuracy")})
        delta = report["runs"][1]["test"]["macro_f1"] - report["runs"][0]["test"]["macro_f1"]
        st.write(f"Observed change in test macro F1: **{delta:+.3f}**. A single split does not establish a reliable gain.")
        selected = st.selectbox("Inspect model", [r["name"] for r in report["runs"]])
        run = next(r for r in report["runs"] if r["name"] == selected)
        a, b = st.columns(2)
        with a:
            st.subheader("Confusion matrix")
            matrix = pd.DataFrame(run["test"]["confusion_matrix"], index=["Actual negative", "Actual positive"], columns=["Predicted negative", "Predicted positive"])
            st.dataframe(matrix, width="stretch")
        with b:
            st.subheader("Precision and recall")
            st.dataframe(pd.DataFrame(run["test"]["classes"]).T.loc[["Negative", "Positive"], ["precision", "recall", "f1-score", "support"]], width="stretch")
        st.subheader("Where the model went wrong")
        test = st.session_state.splits.test.copy()
        test["prediction"] = run["test"]["predictions"]
        mistakes = test[test.label != test.prediction]
        if mistakes.empty:
            st.info("No errors on this small holdout. That does not establish performance on new domains.")
        else:
            st.dataframe(mistakes.assign(Actual=mistakes.label.map(LABELS), Predicted=mistakes.prediction.map(LABELS))[["text", "Actual", "Predicted"]], hide_index=True, width="stretch")
        if st.button("Save experiment"):
            try:
                snapshot = saved_runs.build_record(report, frame, st.session_state.splits)
                saved_id = saved_runs.save_record(snapshot)
                st.success(f"Experiment saved locally as {saved_id[:8]}. Open Saved runs and refresh to reopen it.")
                st.caption("Saved the completed experiment's settings, not the pending slider value. Review text and trained models were not saved.")
                st.session_state.pop("saved_run_entries", None)
            except (ValueError, TypeError, KeyError):
                st.error("This session report cannot be saved with the current contract. Run a new baseline comparison, then save again. Your existing session results remain available for download.")
            except OSError:
                st.error("Could not save this experiment. Your session results are still available. Check artifacts/runs permissions and free disk space, then retry or download the JSON.")
        st.download_button("Download experiment JSON", json.dumps(report, indent=2), "signallab-experiment.json", "application/json")
    else:
        st.caption("No experiment has run for this dataset yet. Results will appear here after training.")
    st.divider()
    st.subheader("PyTorch + Hugging Face")
    st.write("Fine-tune DistilBERT locally, comparing real-only and augmented training with the same data contract. Training runs in the terminal so the dashboard stays responsive.")
    st.code("uv run python -m signallab.train --csv data/reviews.csv --epochs 2 --ratio 0.5", language="bash")
    path = ROOT / "artifacts/transformer/report.json"
    if path.exists():
        try:
            transformer = json.loads(path.read_text())
            if transformer.get("dataset_id") != fingerprint:
                st.info("The saved transformer experiment belongs to a different dataset. Select that dataset or run training again.")
            else:
                st.caption(f"Saved transformer experiment · checkpoint {transformer['checkpoint']} · ratio {transformer['augmentation']['ratio']} · seed {transformer['seed']}")
                st.dataframe(pd.DataFrame([{"Model": r["name"], "Validation macro F1": r["validation"]["macro_f1"], "Test macro F1": r["test"]["macro_f1"], "Training rows": r["training_rows"]} for r in transformer["runs"]]), hide_index=True, width="stretch")
                st.download_button("Download transformer report", json.dumps(transformer, indent=2), "signallab-transformer.json", "application/json")
        except (OSError, ValueError, KeyError, TypeError):
            st.error("The saved transformer report is invalid. Rerun the training command to produce a fresh report.")
    else:
        st.caption("No saved transformer run yet. No transformer scores are displayed until training completes.")

elif page == "Analysis":
    from signallab.analysis import analysis_rows, slice_metrics

    with st.container(key="page_intro"):
        st.title("Investigate model errors")
        st.write("Find the reviews behind the scores. Compare mistakes on the same evaluated examples.")
    origin = st.selectbox("Analysis source", ["Current session", "Saved experiment"], key="analysis_source")
    report = None
    analysis_splits = None
    if origin == "Current session":
        report = st.session_state.get("report")
        analysis_splits = st.session_state.get("splits")
        if not report:
            st.info("Run a baseline comparison in Experiments first, or choose Saved experiment and refresh the history.")
    else:
        if st.button("Refresh saved runs"):
            refresh_saved_runs()
        ready = [entry for entry in st.session_state.get("saved_run_entries", []) if entry["status"] == "Ready"]
        if not ready:
            st.info("Refresh saved runs to find completed baseline snapshots. Save a comparison in Experiments if you have none yet.")
        else:
            choices = [entry["run_id"] for entry in ready]
            if st.session_state.get("saved_analysis_run") not in choices:
                st.session_state.pop("saved_analysis_run", None)
            saved_id = st.selectbox("Saved analysis experiment", choices, key="saved_analysis_run",
                                    format_func=lambda value: saved_run_label(next(entry for entry in ready if entry["run_id"] == value)))
            try:
                saved_report = saved_runs.read_record(saved_runs.RUNS_ROOT, saved_id)
                analysis_splits = saved_runs.attach_reviews(saved_report, frame)
                report = saved_report
                st.caption(f"Saved run {saved_id[:8]} · completed ratio {report['augmentation']['ratio']:.0%} · review text attached from the exact selected dataset")
            except (OSError, ValueError, TypeError, KeyError) as exc:
                st.error(f"Cannot inspect this saved experiment: {exc}")
    if report:
        partition = st.selectbox("Evaluation partition", ["validation", "test"], key="analysis_partition")
        if partition == "test":
            st.warning("These test results are exploratory. Repeated inspection makes this a development set; reserve fresh reviews for final claims.")
        else:
            st.caption("Validation is the default for diagnosis. Use a fresh final cohort after choosing your experiment settings.")
        runs = list(report["runs"])
        saved = ROOT / "artifacts/transformer/report.json"
        if saved.exists():
            try:
                transformer = json.loads(saved.read_text())
                if transformer.get("dataset_id") == fingerprint:
                    if all(r.get(partition, {}).get("evidence_version") == 1 for r in transformer["runs"]):
                        runs.extend(transformer["runs"])
                    else:
                        st.caption("Legacy transformer results lack exact review IDs. Rerun training into a new output directory before using them for detailed analysis.")
            except (OSError, ValueError, KeyError, TypeError):
                st.warning("Saved transformer analysis is unavailable: its report could not be read.")
        names = [run["name"] for run in runs]
        selected = st.selectbox("Analyse model", names)
        comparison = st.selectbox("Compare against", [name for name in names if name != selected])
        try:
            evaluated = getattr(analysis_splits, partition)
            rows = analysis_rows(next(r for r in runs if r["name"] == selected)[partition], evaluated)
            other = analysis_rows(next(r for r in runs if r["name"] == comparison)[partition], evaluated)
            other_predictions = other.set_index("review_id").prediction
            rows["Comparison prediction"] = rows.review_id.map(other_predictions).map(LABELS)
            rows["Disagreement"] = rows.Predicted != rows["Comparison prediction"]
            a, b, c = st.columns(3)
            a.metric("Evaluated reviews", len(rows))
            b.metric("Model errors", int((~rows.correct).sum()))
            c.metric("Model disagreements", int(rows.Disagreement.sum()))
            st.subheader("Where performance changes")
            st.caption("Fixed word-count bands, not model tokens. Small slices are descriptive and may be unstable; review the counts alongside accuracy.")
            left, right = st.columns(2)
            with left:
                st.write("By actual sentiment")
                st.dataframe(slice_metrics(rows, "Actual"), hide_index=True, width="stretch")
            with right:
                st.write("By review length")
                st.dataframe(slice_metrics(rows, "Length"), hide_index=True, width="stretch")
            st.subheader("Read individual predictions")
            show = st.selectbox("Show reviews", ["Model errors", "Model disagreements", "All evaluated reviews"])
            sentiment = st.selectbox("Actual sentiment", ["All", "Negative", "Positive"])
            query = st.text_input("Search evaluated reviews")
            visible = rows.copy()
            if show == "Model errors":
                visible = visible[~visible.correct]
            elif show == "Model disagreements":
                visible = visible[visible.Disagreement]
            if sentiment != "All":
                visible = visible[visible.Actual == sentiment]
            if query:
                visible = visible[visible.text.str.contains(query, case=False, regex=False)]
            st.caption(f"{len(visible)} of {len(rows)} evaluated reviews · comparison: {comparison}")
            if visible.empty:
                st.info("No reviews match these filters. Try all evaluated reviews or clear the search.")
            else:
                st.dataframe(visible[["text", "Actual", "Predicted", "Comparison prediction", "Words"]], hide_index=True, width="stretch")
            st.download_button("Download filtered analysis CSV", visible.to_csv(index=False), "signallab-analysis.csv", "text/csv")
            st.caption("CSV exports include review text. Experiment JSON exports contain exact review IDs and labels without text.")
        except (ValueError, KeyError, TypeError) as exc:
            st.error(f"Cannot align this experiment with the selected reviews: {exc}")

elif page == "Predict":
    models = st.session_state.get("models")
    if not models:
        st.info("Run a baseline comparison in Experiments first. Predictions use the model trained on your selected dataset.")
    else:
        model_name = st.selectbox("Model", list(models), on_change=clear_prediction)
        reconcile_prediction(("baseline", input_id, model_name, st.session_state.get("training_generation")))
        with st.form("prediction_form"):
            text = st.text_area("Review", placeholder="The sound is excellent, but the battery barely lasts an hour.", max_chars=20000)
            submitted = st.form_submit_button("Analyse review", type="primary")
        if submitted:
            st.session_state.pop("prediction", None)
            if not text.strip():
                st.error("Enter a review before analysing it.")
            else:
                model = models[model_name]
                probabilities = model.predict_proba([text])[0]
                label = int(model.classes_[probabilities.argmax()])
                st.session_state.prediction = {"label": label, "scores": probabilities.tolist(), "model": model_name}
        if "prediction" in st.session_state:
            result = st.session_state.prediction
            st.success(f"Predicted sentiment: {LABELS[result['label']]}")
            st.caption(f"Last submitted review · {result['model']}")
            st.dataframe(pd.DataFrame({"Sentiment": list(LABELS.values()), "Model probability": result["scores"]}), hide_index=True, width="stretch")
            st.caption("These probabilities are not calibrated confidence scores. A binary model must choose positive or negative, even for neutral, sarcastic or mixed reviews.")

else:
    with st.container(key="page_intro"):
        st.title("How the experiment works")
        st.write("SignalLab makes the learning process inspectable: data quality, a simple baseline, controlled augmentation, then a transformer comparison.")
    st.subheader("The experiment contract")
    st.markdown("""
1. Validate binary labels and non-empty text. Reject conflicting labels and remove case/whitespace duplicates and known prefix-template variants.
2. Sort consistently and stratify a fixed 60/20/20 split with seed 42.
3. Generate prefix-template examples from the training split only. Keep validation and test text unchanged.
4. Fit TF-IDF and logistic regression on training data. Fine-tune DistilBERT through the separate PyTorch command.
5. Select transformer checkpoints using validation macro F1, then evaluate the selected checkpoint on test reviews.
6. Inspect both classes, the confusion matrix and individual mistakes. Export the complete measured report.
""")
    st.subheader("What the scores can tell you")
    st.write("Macro F1 gives both classes equal weight. Precision asks how often a predicted class is correct; recall asks how many examples of a class are found. Accuracy alone can hide poor minority-class performance.")
    st.write("The same test partition supports an exploratory comparison. Repeatedly adjusting settings after looking at its scores makes it a development set in practice. Reserve an additional unseen dataset for final portfolio claims. Use multiple seeds and uncertainty estimates before claiming a stable augmentation benefit.")
    st.subheader("Limits worth documenting")
    st.markdown("""
- The bundled dataset is authored and tiny. It is suitable for demonstrating functionality only.
- Prefix augmentation is a deliberately weak baseline, not an LLM-generated dataset or independent new evidence.
- Deduplication catches case and whitespace duplicates, not semantic paraphrases or repeated customers. Group by customer/product when those identifiers are available.
- Pretrained models may have seen public benchmark text. Document the checkpoint and dataset before making generalisation claims.
- English binary sentiment cannot represent neutral or mixed opinions. Do not use the demo for consequential decisions.
- Uploaded text stays in this local process. Dataset/model downloads contact Hugging Face; saved reports contain predictions and dataset hashes, not review text. Downloaded CSV exports contain the selected reviews.
""")
    st.subheader("Bring real data")
    st.code("uv run python -m signallab.fetch_data --limit 1000\nuv run streamlit run app.py", language="bash")
    st.write("The download command creates a balanced sample from the public Amazon Polarity training split. Review its dataset card and usage conditions before sharing the data. Select ‘Local review dataset’ after downloading.")
    st.link_button("Read the dataset card", "https://huggingface.co/datasets/fancyzhx/amazon_polarity")


st.divider()
st.caption("SignalLab · Customer review research · Built to question the results.")
