"""BLIP visual question answering adapter."""
from .captioner import load_rgb
from .model_runtime import GenerationConfig, dependencies, device, generate_with_evidence, set_seed

class BlipVQA:
    def __init__(self, model_name="Salesforce/blip-vqa-base", requested_device=None):
        torch, _, classes = dependencies(); _, model_class, processor_class = classes
        self.torch, self.device, self.model_name = torch, device(requested_device), model_name
        self.processor = processor_class.from_pretrained(model_name)
        self.model = model_class.from_pretrained(model_name).to(self.device); self.model.eval()
    def answer(self, image, question, config=None):
        return self.answer_with_evidence(image, question, config).text
    def answer_with_evidence(self, image, question, config=None):
        if not isinstance(question, str) or not question.strip(): raise ValueError("question must not be empty")
        config = config or GenerationConfig(); set_seed(config.seed)
        inputs = self.processor(images=load_rgb(image), text=question.strip(), return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        return generate_with_evidence(self.model, self.processor, inputs, config, self.torch)
