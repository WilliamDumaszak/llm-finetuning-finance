"""
Fine-tuning pipeline for financial sentiment classification.

Model  : Llama 3.2 1B (meta-llama/Llama-3.2-1B)
Dataset: financial_phrasebank (4,840 labeled financial sentences)
Method : LoRA via HuggingFace PEFT + SFTTrainer (TRL)
Device : MPS (Apple Silicon) | CUDA | CPU

Stages (run in order):
  1. python scripts/prepare_data.py        — download and format dataset
  2. python scripts/train.py               — LoRA fine-tuning
  3. python scripts/evaluate.py            — accuracy + F1 vs base model
  4. python scripts/export_model.py        — merge LoRA weights and save
  5. uvicorn api.main:app                  — serve fine-tuned model
"""
