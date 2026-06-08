variable "project_name" {
  description = "Base name for Azure resources."
  type        = string
  default     = "llmfinetuning"
}

variable "environment" {
  description = "Environment suffix (dev, stg, prod)."
  type        = string
  default     = "dev"
}

variable "location" {
  description = "Azure region."
  type        = string
  default     = "eastus"
}

variable "image_reference" {
  description = "Full image reference used by Container App."
  type        = string
  default     = "mcr.microsoft.com/azuredocs/containerapps-helloworld:latest"
}

variable "container_cpu" {
  description = "CPU cores for Container App."
  type        = number
  default     = 1
}

variable "container_memory" {
  description = "Memory for Container App (Gi format, e.g. 2Gi)."
  type        = string
  default     = "2Gi"
}

variable "min_replicas" {
  description = "Minimum number of container replicas."
  type        = number
  default     = 1
}

variable "max_replicas" {
  description = "Maximum number of container replicas."
  type        = number
  default     = 3
}
