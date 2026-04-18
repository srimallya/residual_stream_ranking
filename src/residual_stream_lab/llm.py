from __future__ import annotations

from pathlib import Path

import numpy as np
from llama_cpp import Llama


class GGUFRunner:
    def __init__(
        self,
        model_path: str,
        n_ctx: int = 4096,
        n_batch: int = 512,
        verbose: bool = False,
    ) -> None:
        self.model_path = Path(model_path)
        self.generator = Llama(
            model_path=str(self.model_path),
            n_ctx=n_ctx,
            n_batch=n_batch,
            embedding=False,
            logits_all=False,
            verbose=verbose,
        )
        self.embedder = Llama(
            model_path=str(self.model_path),
            n_ctx=n_ctx,
            n_batch=n_batch,
            embedding=True,
            logits_all=False,
            verbose=verbose,
        )

    def embed(self, text: str) -> np.ndarray:
        values = self.embedder.embed(text, normalize=True)
        vector = values[0] if values and isinstance(values[0], list) else values
        return np.asarray(vector, dtype=np.float32)

    def complete(
        self,
        prompt: str,
        max_tokens: int = 32,
        temperature: float = 0.0,
    ) -> str:
        response = self.generator(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=1.0,
            repeat_penalty=1.0,
            stop=["\n", "</answer>"],
        )
        return response["choices"][0]["text"].strip()

    def answer_question(
        self,
        memory: str,
        question: str,
        max_tokens: int = 32,
        temperature: float = 0.0,
    ) -> str:
        response = self.generator.create_chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You answer questions from provided memory only. "
                        "If the answer is missing, reply with UNKNOWN. "
                        "Return only the final short answer."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Memory:\n{memory}\n\nQuestion: {question}",
                },
            ],
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=1.0,
        )
        return response["choices"][0]["message"]["content"].strip()
