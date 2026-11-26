# Image Captioning and VQA Assistant

## Dataset

100 reviewed images, 300 QA pairs, three QA types per image. Reference answers are intentionally short: object labels, scene categories, and compact visible-object lists. Images were restored from the referenced Hugging Face archive using source archive entry order; see `dataset/restoration_manifest.json`.

## Models

- Caption: `Salesforce/blip-image-captioning-base`
- VQA: `Salesforce/blip-vqa-base`

## Evaluation

The JSON manifest is independent ground truth, not an answer lookup used by the application. `run_decode_benchmark.py` sends its images and questions to BLIP and records predictions; `evaluate_predictions.py` then compares those predictions with the manifest to calculate BLEU, ROUGE-L, BERTScore, VQA exact match/token F1, and latency. Because image selection order was recovered from archive order, visually spot-check every normalized reference before final submission.
