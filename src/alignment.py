"""Optional lazy CLIP image-text alignment signal."""

from pathlib import Path


class ClipAligner:
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", requested_device: str | None = None):
        self.model_name = model_name
        self.requested_device = requested_device
        self._model = None
        self._processor = None
        self._torch = None
        self._device = None

    def _load(self):
        if self._model is not None:
            return
        try:
            import torch
            from PIL import Image
            from transformers import CLIPModel, CLIPProcessor
        except ImportError as exc:
            raise RuntimeError("CLIP requires torch, Pillow, and transformers") from exc
        self._torch = torch
        self._image_class = Image.Image
        self._device = self.requested_device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._processor = CLIPProcessor.from_pretrained(self.model_name)
        self._model = CLIPModel.from_pretrained(self.model_name).to(self._device)
        self._model.eval()

    def similarity(self, image, text: str) -> float:
        self._load()
        from PIL import Image

        if isinstance(image, (str, Path)):
            image = Image.open(image)
        if not isinstance(image, self._image_class):
            raise TypeError("image must be a path or PIL image")
        inputs = self._processor(text=[text], images=[image.convert("RGB")], return_tensors="pt", padding=True)
        inputs = {name: value.to(self._device) for name, value in inputs.items()}
        with self._torch.inference_mode():
            image_features = self._model.get_image_features(pixel_values=inputs["pixel_values"])
            text_features = self._model.get_text_features(
                input_ids=inputs["input_ids"], attention_mask=inputs.get("attention_mask")
            )
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            cosine = float((image_features * text_features).sum().item())
        return min(1.0, max(0.0, (cosine + 1) / 2))
