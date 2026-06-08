"""
Stage 3 — Evaluate fine-tuned model vs. base model on the test split.

Computes accuracy and macro F1 for both models side by side.
Results are logged to MLflow.

How classification works:
  The model generates text. We extract the first word of the response
  and map it to a label. If the model outputs "positive", label = 2.
  This works because we trained with the instruction format that ends
  with "### Response:\n<label>".
"""

import json
import logging
import os
import re
from pathlib import Path
import socket
import subprocess
import time
from urllib.parse import urlparse

import mlflow
import torch
import yaml
from sklearn.metrics import accuracy_score, classification_report, f1_score
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

with open("config/config.yaml") as f:
    CONFIG = yaml.safe_load(f)

LABELS = CONFIG["dataset"]["labels"]
LABEL2ID = {label: i for i, label in enumerate(LABELS)}


def _can_connect(host: str, port: int, timeout: float = 0.5) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _start_local_mlflow_server(port: int) -> subprocess.Popen | None:
    cmd = ["mlflow", "server", "--host", "127.0.0.1", "--port", str(port)]
    try:
        return subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as exc:
        logger.warning(f"Failed to start local MLflow server: {exc}")
        return None


def _wait_for_port(host: str, port: int, timeout_seconds: float = 8.0) -> bool:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        if _can_connect(host, port, timeout=0.3):
            return True
        time.sleep(0.25)
    return False


def _stop_local_mlflow_server(proc: subprocess.Popen | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    try:
        proc.terminate()
        proc.wait(timeout=5)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


def configure_mlflow() -> tuple[str | None, bool, subprocess.Popen | None]:
    preferred_uri = os.getenv("AZURE_MLFLOW_TRACKING_URI", CONFIG["mlflow"]["tracking_uri"])
    experiment_name = CONFIG["mlflow"]["experiment_name"]

    if preferred_uri.startswith("http://localhost:"):
        parsed = urlparse(preferred_uri)
        port = parsed.port or 5001
        if not _can_connect("localhost", port):
            logger.warning(f"MLflow server offline at {preferred_uri}. Starting local MLflow server...")
            proc = _start_local_mlflow_server(port)
            if proc and _wait_for_port("localhost", port):
                mlflow.set_tracking_uri(preferred_uri)
                mlflow.set_experiment(experiment_name)
                return preferred_uri, True, proc
            _stop_local_mlflow_server(proc)
            return None, False, None

    try:
        mlflow.set_tracking_uri(preferred_uri)
        mlflow.set_experiment(experiment_name)
        return preferred_uri, True, None
    except Exception as exc:
        logger.warning(f"MLflow setup failed for '{preferred_uri}' ({exc}). Disabling MLflow.")
        return None, False, None


def resolve_lora_dir(config_output_dir: str) -> str:
    """
    Resolve LoRA adapter directory.

    Training may save adapters under checkpoint-* folders. If adapter_config.json
    is not present in output_dir, pick the checkpoint with the highest step.
    """
    out = Path(config_output_dir)
    if (out / "adapter_config.json").exists():
        return str(out)

    candidates = sorted(
        [p for p in out.glob("checkpoint-*") if (p / "adapter_config.json").exists()],
        key=lambda p: int(p.name.split("-")[-1]),
    )
    if candidates:
        resolved = str(candidates[-1])
        logger.warning(
            f"LoRA adapter not found at '{config_output_dir}'. Using latest checkpoint: {resolved}"
        )
        return resolved

    raise FileNotFoundError(
        f"No adapter_config.json found in '{config_output_dir}' or checkpoint-* subfolders."
    )


def get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_test_set() -> list[dict]:
    with open("data/test.jsonl") as f:
        return [json.loads(line) for line in f]


def predict_label(model, tokenizer, sentence: str, device: str) -> str:
    """
    Generate a prediction for a sentence.
    Returns the predicted label string (negative | neutral | positive).
    """
    prompt = (
        f"### Instruction:\n"
        f"Classify the sentiment of the following financial news sentence.\n"
        f"Reply with one word: negative, neutral, or positive.\n\n"
        f"### Input:\n{sentence}\n\n"
        f"### Response:\n"
    )
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=5,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode only the newly generated tokens (skip prompt)
    generated = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    first_word = re.split(r"[\s\n.,!]", generated.strip().lower())[0]

    return first_word if first_word in LABEL2ID else "neutral"


def evaluate_model(model, tokenizer, test_samples: list[dict], device: str, model_name: str) -> dict:
    logger.info(f"Evaluating {model_name} on {len(test_samples)} samples...")
    true_labels = [s["label"] for s in test_samples]
    pred_labels = []

    for i, sample in enumerate(test_samples):
        # Extract original sentence from instruction format
        sentence = sample["text"].split("### Input:\n")[1].split("\n\n### Response:")[0].strip()
        pred = predict_label(model, tokenizer, sentence, device)
        pred_labels.append(LABEL2ID.get(pred, 1))  # default to neutral on unknown

        if (i + 1) % 20 == 0:
            logger.info(f"  {i + 1}/{len(test_samples)} done")

    acc = accuracy_score(true_labels, pred_labels)
    f1 = f1_score(true_labels, pred_labels, average="macro")
    report = classification_report(true_labels, pred_labels, target_names=LABELS)

    logger.info(f"\n{model_name} Results:\n  Accuracy: {acc:.4f} | Macro F1: {f1:.4f}")
    logger.info(f"\n{report}")
    return {"accuracy": acc, "macro_f1": f1}


def main():
    device = get_device()
    test_samples = load_test_set()
    model_name = CONFIG["base_model"]["name"]

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Evaluate base model ───────────────────────────────────────────────────
    logger.info("Loading base model for baseline evaluation...")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float32, device_map={"": device}
    )
    base_metrics = evaluate_model(base_model, tokenizer, test_samples, device, "BASE MODEL")
    del base_model
    if device == "mps":
        torch.mps.empty_cache()

    # ── Evaluate fine-tuned model ─────────────────────────────────────────────
    logger.info("Loading fine-tuned model...")
    from peft import PeftModel
    ft_base = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float32, device_map={"": device}
    )
    lora_dir = resolve_lora_dir(CONFIG["training"]["output_dir"])
    ft_model = PeftModel.from_pretrained(ft_base, lora_dir)
    ft_metrics = evaluate_model(ft_model, tokenizer, test_samples, device, "FINE-TUNED MODEL")

    # ── Log to MLflow ─────────────────────────────────────────────────────────
    tracking_uri, tracking_enabled, mlflow_proc = configure_mlflow()

    if not tracking_enabled:
        logger.info("MLflow disabled for this run — skipping metric logging.")
        return

    try:
        with mlflow.start_run(run_name="evaluation"):
            mlflow.log_metrics({
                "base_accuracy": base_metrics["accuracy"],
                "base_macro_f1": base_metrics["macro_f1"],
                "finetuned_accuracy": ft_metrics["accuracy"],
                "finetuned_macro_f1": ft_metrics["macro_f1"],
                "accuracy_gain": ft_metrics["accuracy"] - base_metrics["accuracy"],
                "f1_gain": ft_metrics["macro_f1"] - base_metrics["macro_f1"],
            })
    finally:
        _stop_local_mlflow_server(mlflow_proc)

    logger.info(
        f"\n\nSummary:\n"
        f"  Base model     — Accuracy: {base_metrics['accuracy']:.4f} | F1: {base_metrics['macro_f1']:.4f}\n"
        f"  Fine-tuned     — Accuracy: {ft_metrics['accuracy']:.4f} | F1: {ft_metrics['macro_f1']:.4f}\n"
        f"  Accuracy gain  : {ft_metrics['accuracy'] - base_metrics['accuracy']:+.4f}\n"
        f"  F1 gain        : {ft_metrics['macro_f1'] - base_metrics['macro_f1']:+.4f}"
    )


if __name__ == "__main__":
    main()
