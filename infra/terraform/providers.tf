terraform {
  required_version = ">= 1.6.0"

  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 4.0"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.6"
    }
  }

  # ── Backend remoto ──────────────────────────────────────────────────────────
  # Usado automaticamente quando AZURE_STORAGE_ACCOUNT_NAME e
  # AZURE_STORAGE_CONTAINER_NAME estão definidos via env vars ou
  # quando o bootstrap_backend.sh já criou o storage account.
  # Para usar backend local em dev, comente o bloco backend.
  backend "azurerm" {
    resource_group_name  = "rg-llmfinetuning-tfstate"
    storage_account_name = "tfstatellmfin736faa10"
    container_name       = "tfstate"
    key                  = "llm-finetuning-finance.tfstate"
  }
}

provider "azurerm" {
  features {}
}
