# LLM Fine-tuning Finance

End-to-end LoRA fine-tuning project for financial sentiment classification, from dataset preparation to API serving and Azure-ready deployment.

## Overview

This project fine-tunes a 1B LLM for 3-class sentiment classification on financial news (`negative`, `neutral`, `positive`).

It includes:
- Data preparation with instruction-style formatting
- LoRA training with TRL
- Evaluation against the base model
- LoRA merge/export for production inference
- FastAPI serving with confidence score
- Docker packaging
- Azure deployment automation (Terraform + GitHub Actions)

## Architecture

```text
financial_phrasebank
  -> scripts/prepare_data.py
  -> scripts/train.py (LoRA)
  -> scripts/evaluate.py (base vs tuned)
  -> scripts/export_model.py (merge)
  -> api/main.py (/health, /predict)
  -> Docker/Azure deployment
```

## Repository Structure

```text
llm-finetuning-finance/
  config/
    config.yaml
  scripts/
    prepare_data.py
    train.py
    evaluate.py
    export_model.py
    deploy_azure.sh
  api/
    main.py
    schemas.py
  infra/
    bootstrap_backend.sh
    terraform/
  tests/
    test_api.py
  artifacts/
  data/
  Dockerfile
  .env.example
```

## Local Run

### 1) Setup

```bash
cd llm-finetuning-finance
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Prepare Data

```bash
python scripts/prepare_data.py
```

### 3) Train (LoRA)

```bash
python scripts/train.py
```

### 4) Evaluate

```bash
python scripts/evaluate.py
```

### 5) Export Merged Model

```bash
python scripts/export_model.py
```

### 6) Run API

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

### 7) Test API

```bash
curl -s http://localhost:8000/health

curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text":"The company reported record quarterly profits and raised guidance."}'
```

## Docker

```bash
docker build -t llm-finetuning-finance:local .
docker run --rm -p 8001:8000 llm-finetuning-finance:local
```

Then test:

```bash
curl -s http://127.0.0.1:8001/health
```

## Azure Deployment

### Option A: Local CLI + Terraform

```bash
./infra/bootstrap_backend.sh
./scripts/deploy_azure.sh latest
```

### Option B: GitHub Actions (recommended)

Use workflow `.github/workflows/azure-infra-deploy.yml`.

Supported auth modes:
- OIDC: `AZURE_CLIENT_ID`, `AZURE_TENANT_ID`, `AZURE_SUBSCRIPTION_ID`
- Legacy JSON secret: `AZURE_CREDENTIALS`

## Core Design Decisions

- LoRA instead of full fine-tuning to reduce memory and cost
- Instruction-style labels so the generative model can classify via constrained outputs
- MLflow auto-fallback behavior to avoid training failure when local tracking is unavailable
- Automatic checkpoint resolution for evaluation and export

## Notes

- If running on macOS with Apple Silicon, training uses MPS when available.
- If port `8000` is occupied, map to another host port (`8001:8000`).
- Azure access can fail due to Conditional Access policies on unmanaged devices.

## License

This project follows the repository root license.
