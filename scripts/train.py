"""
Stage 2 — LoRA fine-tuning of Llama 3.2 1B on financial_phrasebank.

Key decisions:
  - LoRA (not QLoRA): 1B model fits in M1 unified memory without quantization.
    QLoRA is needed for 7B+ models. For 1B, full LoRA gives better gradient flow.
  - target_modules q_proj + v_proj: standard choice for attention-based LoRA.
    Covers the most impactful weight matrices without adding all linear layers.
  - MPS device: Apple Silicon GPU via PyTorch Metal backend.
    Falls back to CPU if MPS is not available (e.g. CI/CD runner).
  - Instruction format: Llama is a generative model. We frame classification
    as text generation — the model learns to output "positive"/"neutral"/"negative".
"""

import logging
import os
import socket
import subprocess
import time
from urllib.parse import urlparse

import mlflow
import torch
import yaml
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

with open("config/config.yaml") as f:
    CONFIG = yaml.safe_load(f)


def _can_connect(host: str, port: int, timeout: float = 0.5) -> bool:
    """Fast connectivity probe to avoid long MLflow retry loops."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _start_local_mlflow_server(port: int) -> subprocess.Popen | None:
    """Start a local MLflow server process for this training run."""
    cmd = [
        "mlflow",
        "server",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
    ]
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return proc
    except Exception as exc:
        logger.warning(f"Failed to start local MLflow server: {exc}")
        return None


def _wait_for_port(host: str, port: int, timeout_seconds: float = 8.0) -> bool:
    """Wait until host:port accepts TCP connections."""
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        if _can_connect(host, port, timeout=0.3):
            return True
        time.sleep(0.25)
    return False


def _stop_local_mlflow_server(proc: subprocess.Popen | None) -> None:
    """Stop MLflow process started by this script, if still running."""
    if proc is None:
        return
    if proc.poll() is not None:
        return
    try:
        proc.terminate()
        proc.wait(timeout=5)
        logger.info("Local MLflow server stopped.")
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


def configure_mlflow() -> tuple[str | None, bool, subprocess.Popen | None]:
    """
    Configure MLflow tracking URI.

    Behavior:
      1) Prefer AZURE_MLFLOW_TRACKING_URI when set; else config mlflow.tracking_uri.
      2) If URI points to localhost and server is down, fallback immediately to local file store.
      3) If remote setup fails for any reason, fallback to local file store.

        Returns:
            (tracking_uri_used_or_none, tracking_enabled, mlflow_process_started_by_this_script)
    """
    preferred_uri = os.getenv("AZURE_MLFLOW_TRACKING_URI", CONFIG["mlflow"]["tracking_uri"])
    experiment_name = CONFIG["mlflow"]["experiment_name"]

    # Short-circuit the common local failure case to avoid noisy retries.
    if preferred_uri.startswith("http://localhost:"):
        try:
            parsed = urlparse(preferred_uri)
            port = parsed.port or 5001
            if not _can_connect("localhost", port):
                logger.warning(f"MLflow server offline at {preferred_uri}. Starting local MLflow server...")
                proc = _start_local_mlflow_server(port)
                if proc and _wait_for_port("localhost", port):
                    mlflow.set_tracking_uri(preferred_uri)
                    mlflow.set_experiment(experiment_name)
                    logger.info(f"Local MLflow server started on {preferred_uri}.")
                    return preferred_uri, True, proc

                logger.warning(
                    f"Could not start local MLflow on {preferred_uri}. Disabling MLflow for this run."
                )
                _stop_local_mlflow_server(proc)
                return None, False, None
        except Exception:
            pass

    # Try preferred URI first.
    try:
        mlflow.set_tracking_uri(preferred_uri)
        mlflow.set_experiment(experiment_name)
        return preferred_uri, True, None
    except Exception as exc:
        logger.warning(
            f"MLflow setup failed for '{preferred_uri}' ({exc}). Disabling MLflow for this run."
        )
        return None, False, None


def get_device() -> str:
    """Select best available device: MPS (Apple Silicon) > CUDA > CPU."""
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_jsonl(path: str) -> Dataset:
    import json
    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line))
    return Dataset.from_list(records)


def main():
    device = get_device()
    logger.info(f"Training device: {device}")

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    model_name = CONFIG["base_model"]["name"]
    logger.info(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── Base model ────────────────────────────────────────────────────────────
    logger.info("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,   # float32 required for MPS stability
        device_map={"": device},
    )
    model.config.use_cache = False   # required for gradient checkpointing

    # ── LoRA config ───────────────────────────────────────────────────────────
    lora_cfg = CONFIG["lora"]
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["alpha"],
        lora_dropout=lora_cfg["dropout"],
        target_modules=lora_cfg["target_modules"],
        bias="none",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # ── Dataset ───────────────────────────────────────────────────────────────
    train_ds = load_jsonl("data/train.jsonl")
    val_ds = load_jsonl("data/val.jsonl")
    logger.info(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    # ── Training args ─────────────────────────────────────────────────────────
    tr_cfg = CONFIG["training"]
    training_args = SFTConfig(
        output_dir=tr_cfg["output_dir"],
        num_train_epochs=tr_cfg["num_epochs"],
        per_device_train_batch_size=tr_cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=tr_cfg["per_device_eval_batch_size"],
        gradient_accumulation_steps=tr_cfg["gradient_accumulation_steps"],
        learning_rate=tr_cfg["learning_rate"],
        warmup_ratio=tr_cfg["warmup_ratio"],
        lr_scheduler_type=tr_cfg["lr_scheduler_type"],
        logging_steps=tr_cfg["logging_steps"],
        eval_strategy=tr_cfg["eval_strategy"],
        save_strategy=tr_cfg["save_strategy"],
        load_best_model_at_end=tr_cfg["load_best_model_at_end"],
        fp16=tr_cfg["fp16"],
        bf16=tr_cfg["bf16"],
        dataset_text_field="text",
        max_length=CONFIG["base_model"]["max_length"],
        packing=False,
        report_to="none",            # MLflow logging is done manually below
    )

    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
    )

    # ── MLflow tracking ───────────────────────────────────────────────────────
    tracking_uri, tracking_enabled, mlflow_proc = configure_mlflow()
    try:
        if tracking_enabled:
            logger.info(f"MLflow tracking uri: {tracking_uri}")
            with mlflow.start_run():
                mlflow.log_params({
                    "base_model": model_name,
                    "lora_r": lora_cfg["r"],
                    "lora_alpha": lora_cfg["alpha"],
                    "lora_target_modules": ",".join(lora_cfg["target_modules"]),
                    "epochs": tr_cfg["num_epochs"],
                    "learning_rate": tr_cfg["learning_rate"],
                    "device": device,
                    "train_samples": len(train_ds),
                    "val_samples": len(val_ds),
                })

                logger.info("Starting fine-tuning...")
                trainer.train()

                # Log final eval loss
                eval_results = trainer.evaluate()
                mlflow.log_metrics({"eval_loss": eval_results["eval_loss"]})
                logger.info(f"Eval loss: {eval_results['eval_loss']:.4f}")

                # Save LoRA checkpoint path as artifact tag
                mlflow.set_tag("lora_checkpoint", tr_cfg["output_dir"])
        else:
            logger.info("MLflow disabled for this run. Starting fine-tuning...")
            trainer.train()
            eval_results = trainer.evaluate()
            logger.info(f"Eval loss: {eval_results['eval_loss']:.4f}")
    finally:
        _stop_local_mlflow_server(mlflow_proc)

    logger.info(f"LoRA checkpoint saved to: {tr_cfg['output_dir']}")


if __name__ == "__main__":
    main()
