# Image Captioning & VQA Assistant

Rebuilt from the verified annotation manifest and restored source images.

## Setup

```bash
python -m pip install -e ".[dev]"
```

Install the optional local speech-to-text and online text-to-speech adapters:

```bash
python -m pip install -e ".[dev,speech]"
```

## Validate dataset

```bash
python scripts/validate_dataset.py
```

## Run app

```bash
python app.py
```

The first voice transcription downloads the `large-v3` Whisper model, which is larger and slower,
especially on CPU. Short recordings are decoded without VAD so quiet words and pauses are not cut.
Select the spoken language explicitly when possible; Vietnamese is the UI default and is more
accurate than automatic detection for short questions. Generated TTS audio is temporary and is
deleted when the conversation resets or the process exits. Edge TTS requires internet access.
Alternatively, configure `PIPER_EXECUTABLE`,
`PIPER_VI_MODEL`, and/or `PIPER_EN_MODEL` before starting the app.

Benchmark a reviewed audio clip close to 10 seconds:

```bash
python scripts/benchmark_speech.py path/to/10-second-audio.wav --language vi
```

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

```bash
python scripts/run_decode_benchmark.py --limit 2 --repeats 1
python scripts/evaluate_predictions.py artifacts/predictions.jsonl
python scripts/build_error_report.py artifacts/predictions.jsonl
```

The image restore mapping is recorded in `dataset/restoration_manifest.json`.

## Development roadmap

See [`docs/application-roadmap.md`](docs/application-roadmap.md) for the planned voice assistant,
confidence/abstention, and lightweight domain fine-tuning milestones.
