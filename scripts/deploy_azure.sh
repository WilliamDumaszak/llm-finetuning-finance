#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TF_DIR="$PROJECT_DIR/infra/terraform"
IMAGE_REPO="llm-finetuning-finance"
IMAGE_TAG="${1:-latest}"

if ! command -v az >/dev/null 2>&1; then
  echo "Azure CLI (az) não encontrado. Instale e tente novamente."
  exit 1
fi

if ! command -v terraform >/dev/null 2>&1; then
  echo "Terraform não encontrado. Instale e tente novamente."
  exit 1
fi

echo "[1/5] Validando login Azure..."
az account show >/dev/null

if ! az extension show --name containerapp >/dev/null 2>&1; then
  echo "Instalando extensão containerapp..."
  az extension add --name containerapp >/dev/null
fi

# Bootstrap do backend Terraform (idempotente)
BOOTSTRAP="$PROJECT_DIR/infra/bootstrap_backend.sh"
if [[ -f "$BOOTSTRAP" ]]; then
  echo "[bootstrap] Verificando backend Terraform..."
  bash "$BOOTSTRAP"
fi

echo "[2/5] Provisionando infraestrutura com Terraform..."
cd "$TF_DIR"
terraform init -upgrade
terraform apply -auto-approve

ACR_NAME="$(terraform output -raw acr_name)"
ACR_LOGIN_SERVER="$(terraform output -raw acr_login_server)"
IMAGE_REF="$ACR_LOGIN_SERVER/$IMAGE_REPO:$IMAGE_TAG"

echo "[3/5] Build e push da imagem para ACR: $IMAGE_REF"
az acr build \
  --registry "$ACR_NAME" \
  --image "$IMAGE_REPO:$IMAGE_TAG" \
  "$PROJECT_DIR"

echo "[4/5] Atualizando Container App com a nova imagem..."
terraform apply -auto-approve -var="image_reference=$IMAGE_REF"

APP_URL="$(terraform output -raw container_app_url)"

echo "[5/5] Deploy concluído."
echo "URL: $APP_URL"
echo "Health: $APP_URL/health"
