"""
FastAPI serving for the fine-tuned financial sentiment model.

Endpoints:
  POST /predict   — classify sentiment of a financial sentence
  GET  /health    — liveness check (model loaded?)
"""

import logging
import os
import re
import sys
from contextlib import asynccontextmanager

import torch
import yaml
from fastapi import FastAPI, HTTPException
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from api.schemas import HealthResponse, SentimentRequest, SentimentResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

with open("config/config.yaml") as f:
    CONFIG = yaml.safe_load(f)

LABELS = CONFIG["dataset"]["labels"]
LABEL2ID = {label: i for i, label in enumerate(LABELS)}

# Shared state — loaded once at startup
MODEL_STATE: dict = {"model": None, "tokenizer": None, "device": None}


def _get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_path = CONFIG["api"]["model_path"]
    device = _get_device()
    logger.info(f"Loading model from {model_path} on {device}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float32,
            device_map={"": device},
        )
        model.eval()
        MODEL_STATE["model"] = model
        MODEL_STATE["tokenizer"] = tokenizer
        MODEL_STATE["device"] = device
        logger.info("Model loaded successfully.")
    except Exception as exc:
        logger.error(f"Failed to load model: {exc}")
    yield
    MODEL_STATE.clear()


app = FastAPI(
    title="Financial Sentiment Classifier",
    description=(
        "Fine-tuned Llama 3.2 1B (LoRA) for financial news sentiment classification. "
        "Labels: negative, neutral, positive."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


def _classify(text: str) -> tuple[str, int, float]:
    """
    Run inference and return (label_text, label_id, confidence).

    Confidence is computed from the log-probabilities of the first generated token.
    The three candidate tokens are 'negative', 'neutral', 'positive'.
    """
    model = MODEL_STATE["model"]
    tokenizer = MODEL_STATE["tokenizer"]
    device = MODEL_STATE["device"]

    prompt = (
        f"### Instruction:\n"
        f"Classify the sentiment of the following financial news sentence.\n"
        f"Reply with one word: negative, neutral, or positive.\n\n"
        f"### Input:\n{text}\n\n"
        f"### Response:\n"
    )
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256).to(device)

    with torch.no_grad():
        # Get logits for the next token (the label token)
        outputs = model(**inputs)
        next_token_logits = outputs.logits[:, -1, :]  # shape: (1, vocab_size)

        # Map each label word to its token id
        label_token_ids = {
            label: tokenizer.encode(label, add_special_tokens=False)[0]
            for label in LABELS
        }
        label_ids_tensor = torch.tensor(list(label_token_ids.values()), device=device)
        label_logits = next_token_logits[0, label_ids_tensor]
        probs = torch.softmax(label_logits, dim=0)

    best_idx = int(probs.argmax())
    best_label = LABELS[best_idx]
    confidence = float(probs[best_idx])
    return best_label, LABEL2ID[best_label], round(confidence, 4)


@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health():
    return HealthResponse(
        status="ok",
        model_loaded=MODEL_STATE.get("model") is not None,
        model_path=CONFIG["api"]["model_path"],
    )


@app.post("/predict", response_model=SentimentResponse, tags=["inference"])
def predict(request: SentimentRequest):
    if MODEL_STATE.get("model") is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    try:
        label_text, label_id, confidence = _classify(request.text)
        return SentimentResponse(
            text=request.text,
            label=label_text,
            label_id=label_id,
            confidence=confidence,
        )
    except Exception as exc:
        logger.error(f"Inference error: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))
