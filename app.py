import os
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from model_runner import ModelRunner

app = FastAPI(title="Python Model Runner", version="1.0.0")
runner = ModelRunner()


class RunRequest(BaseModel):
    prompt: str
    model: Optional[str] = None
    max_new_tokens: int = 128
    temperature: float = 0.7


@app.get("/")
def root():
    return {
        "service": "python-model-runner",
        "status": "ok",
        "default_model": runner.default_model,
        "installed_models": runner.available_models(),
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.get("/models")
def models():
    return {
        "default": runner.default_model,
        "models": runner.available_models(),
    }


@app.post("/run")
def run(request: RunRequest):
    if not request.prompt.strip():
        raise HTTPException(status_code=400, detail="prompt is required")
    try:
        return runner.generate(
            prompt=request.prompt,
            model_name=request.model,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
