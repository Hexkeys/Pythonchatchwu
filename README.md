# Python Model Runner for Render

A FastAPI service for running Hugging Face text-generation models on Render.

## Deploy

1. Push this repository to GitHub.
2. In Render, create a new Web Service from the repository.
3. Render can use `render.yaml` automatically.
4. The first deployment downloads the configured Hugging Face model.

## Install/select models

Set these Render environment variables:

- `MODEL_ID`: default model, for example `distilgpt2`
- `MODEL_LIST`: comma-separated allowlist of models, for example:
  `distilgpt2,gpt2`

Models are downloaded automatically from Hugging Face the first time they are used.

The API intentionally does not run arbitrary `pip install` commands from HTTP requests.

## API

### List models

GET `/models`

### Run a model

POST `/run`

Example JSON:

```json
{
  "model": "distilgpt2",
  "prompt": "Hello from Render",
  "max_new_tokens": 64,
  "temperature": 0.7
}
```

### Health

GET `/health`
