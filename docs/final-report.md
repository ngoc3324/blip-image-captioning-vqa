# Image Captioning and VQA Assistant

## Dataset

300 reviewed images, 900 QA pairs, three QA types per image. Images were restored from the referenced Hugging Face archive using source archive entry order; see `dataset/restoration_manifest.json`.

## Models

- Caption: `Salesforce/blip-image-captioning-base`
- VQA: `Salesforce/blip-vqa-base`

## Evaluation

Run the benchmark and evaluator to populate BLEU, ROUGE-L, BERTScore, VQA exact match/token F1, latency, and error categories. Because image selection order was recovered from archive order, visually spot-check the mapping before final submission.
