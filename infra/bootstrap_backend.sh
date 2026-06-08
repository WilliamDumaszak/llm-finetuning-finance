#!/usr/bin/env bash
# Bootstrap: cria o Azure Storage Account que guarda o estado remoto do Terraform.
# Execute UMA VEZ antes de qualquer terraform init/apply.
#
# Uso:
#   chmod +x infra/bootstrap_backend.sh
#   ./infra/bootstrap_backend.sh
set -euo pipefail

LOCATION="eastus"
RG_NAME="rg-llmfinetuning-tfstate"
SA_NAME="tfstatellmfin$(openssl rand -hex 4)"   # nome único
CONTAINER="tfstate"

echo "[1/4] Criando resource group para o estado Terraform..."
az group create --name "$RG_NAME" --location "$LOCATION" >/dev/null

echo "[2/4] Criando storage account $SA_NAME..."
az storage account create \
  --name "$SA_NAME" \
  --resource-group "$RG_NAME" \
  --location "$LOCATION" \
  --sku Standard_LRS \
  --kind StorageV2 \
  --min-tls-version TLS1_2 \
  --allow-blob-public-access false \
  >/dev/null

echo "[3/4] Criando container $CONTAINER..."
az storage container create \
  --name "$CONTAINER" \
  --account-name "$SA_NAME" \
  >/dev/null

echo "[4/4] Atualizando providers.tf com o nome do storage account..."
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROVIDERS_FILE="$SCRIPT_DIR/terraform/providers.tf"

sed -i.bak "s/storage_account_name = .*/storage_account_name = \"$SA_NAME\"/" "$PROVIDERS_FILE"
rm -f "${PROVIDERS_FILE}.bak"

echo ""
echo "Backend criado com sucesso."
echo "  Resource group: $RG_NAME"
echo "  Storage account: $SA_NAME"
echo "  Container: $CONTAINER"
echo ""
echo "Próximo passo:"
echo "  cd infra/terraform && terraform init"
