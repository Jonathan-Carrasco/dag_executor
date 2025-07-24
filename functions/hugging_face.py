from transformers import pipeline
from langchain_core.language_models import BaseLLM

class HuggingFaceLLM(BaseLLM):
    def __init__(self, model_id="sshleifer/tiny-gpt2"):
        super().__init__()
        self.generator = pipeline("text-generation", model=model_id)

    def _call(self, prompt: str, stop: list = None) -> str:
        result = self.generator(prompt, max_new_tokens=50)
        return result[0]["generated_text"]

