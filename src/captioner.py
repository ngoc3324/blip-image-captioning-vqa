"""BLIP captioning adapter."""
from pathlib import Path
from typing import Any
from .model_runtime import GenerationConfig, dependencies, device, generate_with_evidence, set_seed

def load_rgb(image: Any) -> Any:
    _, Image, _ = dependencies()
    if isinstance(image, (str, Path)): image = Image.open(image)
    if not isinstance(image, Image.Image): raise TypeError("image must be a path or PIL image")
    return image.convert("RGB")

class BlipCaptioner:
    def __init__(self, model_name="Salesforce/blip-image-captioning-base", requested_device=None):
        torch, _, classes = dependencies(); model_class, _, processor_class = classes
        self.torch, self.device, self.model_name = torch, device(requested_device), model_name
        self.processor = processor_class.from_pretrained(model_name)
        self.model = model_class.from_pretrained(model_name).to(self.device); self.model.eval()
    def caption(self, image, config=None):
        return self.caption_with_evidence(image, config).text
    def caption_with_evidence(self, image, config=None):
        config = config or GenerationConfig(); set_seed(config.seed)
        inputs = self.processor(images=load_rgb(image), return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        return generate_with_evidence(self.model, self.processor, inputs, config, self.torch)
