# Image Captioning & VQA Assistant

Rebuilt from the verified annotation manifest and restored source images.

## Setup

```bash
python -m pip install -e ".[dev]"
```

## Validate dataset

```bash
python scripts/validate_dataset.py
```

## Run app

```bash
python -m streamlit run app.py
```

## Optional speech dependencies

```bash
python -m pip install -e ".[speech,dev]"
```

Speech to text uses the Vietnamese-specific Gipformer 1.5 68M INT8 model through
Sherpa-ONNX. Its quantized model files are downloaded from Hugging Face and cached
locally the first time STT is used; subsequent recognition runs locally. Piper uses
the medium-quality Northern Vietnamese female voice `vi_VN-vais1000-medium` and the
`en_US-amy-low` voice for English output.
Piper voices need their `.onnx` and `.onnx.json` files installed separately; the
Python package does not include voice weights. Download voice files into `voices/`:

```bash
python -m piper.download_voices --download-dir voices vi_VN-vais1000-medium en_US-amy-low
```

The Streamlit app records or uploads audio for Vietnamese transcription and provides
Piper playback controls. Piper's English catalog currently has no `x_low` voice, so
English uses the `low` voice.

## Confidence calibration

Build and manually review the draft answerability set, then run inference and calibration:

```bash
python scripts/build_answerability_validation.py --limit 30
python scripts/run_confidence_validation.py dataset/answerability_validation.json
python scripts/calibrate_confidence.py artifacts/confidence_predictions.jsonl
```

The app automatically loads `artifacts/confidence_calibration.json` on its next start. Until that
file exists, the UI labels confidence thresholds as experimental and does not show a percentage.

## Benchmark/evaluation

`dataset/annotations_300.json` contains the independent reference answers: 100
images and 300 short QA pairs. The application does not look up these answers
when answering a user. Instead, the benchmark sends each image and question to
BLIP, stores its predictions, and the evaluator compares those predictions with
the JSON references. Keep the references human-reviewed; copying BLIP answers
into the manifest would make the evaluation circular.

```bash
python scripts/normalize_vqa_references.py
python scripts/run_decode_benchmark.py --limit 2 --repeats 1
python scripts/evaluate_predictions.py artifacts/predictions.jsonl
python scripts/build_error_report.py artifacts/predictions.jsonl
```

The image restore mapping is recorded in `dataset/restoration_manifest.json`.

## Development roadmap

See [`docs/application-roadmap.md`](docs/application-roadmap.md) for confidence/abstention and
lightweight domain fine-tuning milestones.
