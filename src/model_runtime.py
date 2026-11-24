"""Lazy BLIP dependencies and generation controls."""
from dataclasses import dataclass
from typing import Any
import random

@dataclass(frozen=True)
class GenerationConfig:
    num_beams: int = 1
    max_new_tokens: int = 30
    temperature: float = 1.0
    do_sample: bool = False
    seed: int = 42
    top_p: float | None = None
    top_k: int | None = None

    def kwargs(self) -> dict[str, Any]:
        validate_generation_config(self)
        result = {"num_beams": self.num_beams, "max_new_tokens": self.max_new_tokens, "do_sample": self.do_sample}
        if self.do_sample:
            result.update(temperature=self.temperature)
            if self.top_p is not None: result["top_p"] = self.top_p
            if self.top_k is not None: result["top_k"] = self.top_k
        return result


@dataclass(frozen=True)
class GenerationEvidence:
    text: str
    mean_logprob: float | None = None
    beam_margin: float | None = None
    alternative_text: str | None = None


def generate_with_evidence(model, processor, inputs, config: GenerationConfig, torch) -> GenerationEvidence:
    """Generate text and expose sequence evidence without changing the public text API."""
    kwargs = config.kwargs()
    candidate_count = min(2, config.num_beams) if config.num_beams > 1 and not config.do_sample else 1
    kwargs.update(return_dict_in_generate=True, output_scores=True, num_return_sequences=candidate_count)
    with torch.inference_mode():
        output = model.generate(**inputs, **kwargs)
    sequences = output.sequences
    text = processor.decode(sequences[0], skip_special_tokens=True).strip()
    alternative = None
    if len(sequences) > 1:
        alternative = processor.decode(sequences[1], skip_special_tokens=True).strip() or None

    mean_logprob = None
    try:
        transition = model.compute_transition_scores(
            sequences,
            output.scores,
            getattr(output, "beam_indices", None),
            normalize_logits=True,
        )
        row = transition[0]
        valid = row[row < 0]
        if valid.numel():
            mean_logprob = float(valid.mean().item())
    except (AttributeError, IndexError, RuntimeError, TypeError):
        pass

    beam_margin = None
    sequence_scores = getattr(output, "sequences_scores", None)
    if sequence_scores is not None and len(sequence_scores) > 1:
        beam_margin = float((sequence_scores[0] - sequence_scores[1]).item())
    return GenerationEvidence(
        text=text,
        mean_logprob=mean_logprob,
        beam_margin=beam_margin,
        alternative_text=alternative,
    )

def validate_generation_config(config: GenerationConfig) -> None:
    if config.num_beams < 1 or config.max_new_tokens < 1 or config.temperature <= 0: raise ValueError("Invalid generation settings")
    if not config.do_sample and config.temperature != 1.0: raise ValueError("temperature requires do_sample=True")
    if config.top_p is not None and not 0 < config.top_p <= 1: raise ValueError("top_p must be in (0,1]")
    if config.top_k is not None and config.top_k < 1: raise ValueError("top_k must be positive")

def set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    except ImportError: pass

def device(requested: str | None = None) -> str:
    if requested and requested != "auto": return requested
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError: return "cpu"

def dependencies():
    try:
        import torch
        from PIL import Image
        from transformers import BlipForConditionalGeneration, BlipForQuestionAnswering, BlipProcessor
    except ImportError as exc:
        raise RuntimeError("Install torch, transformers, and Pillow") from exc
    return torch, Image, (BlipForConditionalGeneration, BlipForQuestionAnswering, BlipProcessor)
