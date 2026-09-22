import os
from functools import lru_cache
from typing import Optional

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


DEFAULT_MODEL = os.getenv("MODEL_ID", "distilgpt2")
MODEL_LIST = [
    model.strip()
    for model in os.getenv("MODEL_LIST", DEFAULT_MODEL).split(",")
    if model.strip()
]


class ModelRunner:
    def __init__(self):
        self.default_model = DEFAULT_MODEL

    def available_models(self):
        return MODEL_LIST

    @lru_cache(maxsize=4)
    def _pipeline(self, model_name: str):
        if model_name not in MODEL_LIST:
            raise ValueError(
                f"Model '{model_name}' is not installed. "
                f"Choose one of: {', '.join(MODEL_LIST)}"
            )

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)

        return pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
        )

    def generate(
        self,
        prompt: str,
        model_name: Optional[str],
        max_new_tokens: int,
        temperature: float,
    ):
        selected = model_name or self.default_model
        generator = self._pipeline(selected)

        result = generator(
            prompt,
            max_new_tokens=max(1, min(max_new_tokens, 1024)),
            temperature=max(0.1, min(temperature, 2.0)),
            do_sample=True,
            pad_token_id=generator.tokenizer.eos_token_id,
        )

        return {
            "model": selected,
            "prompt": prompt,
            "text": result[0]["generated_text"],
        }
