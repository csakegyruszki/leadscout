#!/usr/bin/env bash
set -Eeuo pipefail

# Keep defaults simple while allowing CI or an operator to override names.
RESOURCE_GROUP="${RESOURCE_GROUP:-leadscout-prototype-rg}"
LOCATION="${LOCATION:-germanywestcentral}"
DEPLOYMENT_NAME="${DEPLOYMENT_NAME:-leadscout-infra}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

teardown() {
  local answer
  read -r -p "Delete resource group ${RESOURCE_GROUP}? Type 'delete' to confirm: " answer
  if [[ "${answer}" != "delete" ]]; then
    printf 'Teardown cancelled.\n'
    return 0
  fi
  az group delete --name "${RESOURCE_GROUP}" --yes --no-wait
}

if [[ "${1:-}" == "teardown" ]]; then
  teardown
  exit 0
fi

: "${OPENROUTER_API_KEY:?Set OPENROUTER_API_KEY in the environment.}"
: "${OPENSANCTIONS_API_KEY:?Set OPENSANCTIONS_API_KEY in the environment.}"

if [[ -z "${GIT_SHA:-}" ]]; then
  GIT_SHA="$(git -C "${REPO_ROOT}" rev-parse --short HEAD)"
fi

# Re-running these commands converges the resource group and ARM deployment.
az group create \
  --name "${RESOURCE_GROUP}" \
  --location "${LOCATION}" \
  --output none

az deployment group create \
  --name "${DEPLOYMENT_NAME}" \
  --resource-group "${RESOURCE_GROUP}" \
  --parameters "${SCRIPT_DIR}/main.bicepparam" \
  --parameters location="${LOCATION}" \
  --parameters openRouterApiKey="${OPENROUTER_API_KEY}" \
  --parameters openSanctionsApiKey="${OPENSANCTIONS_API_KEY}" \
  --output none

ACR_NAME="$(az deployment group show --name "${DEPLOYMENT_NAME}" --resource-group "${RESOURCE_GROUP}" --query properties.outputs.acrName.value -o tsv)"
VAULT_NAME="$(az deployment group show --name "${DEPLOYMENT_NAME}" --resource-group "${RESOURCE_GROUP}" --query properties.outputs.vaultName.value -o tsv)"
APP_NAME="$(az deployment group show --name "${DEPLOYMENT_NAME}" --resource-group "${RESOURCE_GROUP}" --query properties.outputs.containerAppName.value -o tsv)"

# These explicit writes also provide an idempotent rotation path without printing values.
az keyvault secret set \
  --vault-name "${VAULT_NAME}" \
  --name OPENROUTER-API-KEY \
  --value "${OPENROUTER_API_KEY}" \
  --output none
az keyvault secret set \
  --vault-name "${VAULT_NAME}" \
  --name OPENSANCTIONS-API-KEY \
  --value "${OPENSANCTIONS_API_KEY}" \
  --output none

cd "${REPO_ROOT}"
ACR_LOGIN_SERVER="$(az acr show --name "${ACR_NAME}" --query loginServer -o tsv)"

# Two build paths, because the one this script originally assumed does not work on
# every subscription. ACR Tasks (cloud build, no local Docker) is the default, but
# Azure for Students forbids it - so the live 2026-09-19 deployment was in fact built
# by hand with a local Docker daemon, which is why this step had never run and its
# --file pointed at infra/Dockerfile, a path that does not exist (the Dockerfile is at
# the repo root). Both are fixed here: the path, and the missing local path.
#   BUILD_MODE=acr    (default) ACR Tasks
#   BUILD_MODE=local  local Docker daemon + a short-lived ACR token, no docker login
#                     credentials stored on disk
BUILD_MODE="${BUILD_MODE:-acr}"
if [[ "${BUILD_MODE}" == "local" ]]; then
  # --build-arg GIT_SHA is what makes GET /health report the running commit; without it
  # the image says "unknown" and a stale revision is indistinguishable from a fresh one.
  docker build --build-arg "GIT_SHA=${GIT_SHA}" \
    --file "${REPO_ROOT}/Dockerfile" --tag "${ACR_LOGIN_SERVER}/leadscout:${GIT_SHA}" "${REPO_ROOT}"
  # --expose-token yields a refresh token scoped to this registry; piping it to
  # --password-stdin keeps it out of the process list and the shell history.
  az acr login --name "${ACR_NAME}" --expose-token --output tsv --query accessToken \
    | docker login "${ACR_LOGIN_SERVER}" --username 00000000-0000-0000-0000-000000000000 --password-stdin
  docker push "${ACR_LOGIN_SERVER}/leadscout:${GIT_SHA}"
  docker logout "${ACR_LOGIN_SERVER}"
else
  az acr build \
    --registry "${ACR_NAME}" \
    --image "leadscout:${GIT_SHA}" \
    --build-arg "GIT_SHA=${GIT_SHA}" \
    --file "${REPO_ROOT}/Dockerfile" \
    .
fi

az containerapp update \
  --name "${APP_NAME}" \
  --resource-group "${RESOURCE_GROUP}" \
  --image "${ACR_LOGIN_SERVER}/leadscout:${GIT_SHA}" \
  --output none

APP_FQDN="$(az containerapp show --name "${APP_NAME}" --resource-group "${RESOURCE_GROUP}" --query properties.configuration.ingress.fqdn -o tsv)"
curl --fail --show-error --silent "https://${APP_FQDN}/health"
printf '\nDeployment healthy: https://%s\n' "${APP_FQDN}"
