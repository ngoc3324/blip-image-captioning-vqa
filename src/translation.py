"""Lazy local Vietnamese-English translation for the English-centric BLIP VQA model."""

from functools import lru_cache


@lru_cache(maxsize=2)
def _translator(direction: str):
    try:
        import torch
        from transformers import MarianMTModel, MarianTokenizer
    except ImportError as exc:
        raise RuntimeError("Cần cài torch và transformers để dịch câu hỏi VQA tiếng Việt.") from exc
    model_name = {
        "vi-en": "Helsinki-NLP/opus-mt-vi-en",
        "en-vi": "Helsinki-NLP/opus-mt-en-vi",
    }[direction]
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    model.eval()
    return torch, tokenizer, model


def _translate(text: str, direction: str) -> str:
    if not text or not text.strip():
        return text
    torch, tokenizer, model = _translator(direction)
    inputs = tokenizer([text.strip()], return_tensors="pt", truncation=True, max_length=256)
    with torch.inference_mode():
        generated = model.generate(**inputs, num_beams=4, max_new_tokens=96)
    translated = tokenizer.decode(generated[0], skip_special_tokens=True).strip()
    if not translated:
        raise RuntimeError(f"Model translation {direction} returned empty text")
    return translated


def translate_vi_en(text: str) -> str:
    return _translate(text, "vi-en")


def translate_en_vi(text: str) -> str:
    return _translate(text, "en-vi")
