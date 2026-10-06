# Azure prototype infrastructure

`main.bicep` creates a Consumption Container Apps environment/app, Log Analytics, Basic ACR, a managed identity, RBAC-enabled Standard Key Vault, and Standard_LRS blob/Azure Files storage.
The `out` file share is mounted at `/app/out`; the separate `out` blob container is available for later exports or archival.
The app scales from zero to two replicas and receives both API keys only through managed-identity Key Vault references.

## Cost at zero traffic
Container Apps Consumption is normally covered by its monthly free grant when idle; the environment, identity, and RBAC assignments have no fixed charge.
Key Vault has no fixed instance charge but operations are metered and should be pennies at prototype volume.
ACR Basic is the main fixed paid item (roughly US$5/month, region/currency dependent).
Log Analytics is metered (typically $0 at zero ingestion within its included allowance), while Standard_LRS Storage/Azure Files costs depend on stored GiB and transactions (usually cents for a tiny prototype).
Budget roughly US$5–10/month at near-zero traffic; the figures above are region/currency dependent — `main.bicep`'s `location` parameter defaults to `germanywestcentral`, but the actually deployed stack runs in `polandcentral` (see "What was actually deployed" below), so verify current prices for whichever region a given run targets in the Azure calculator.

## Deploy and review
Run `infra/deploy.sh` from a repo with `OPENROUTER_API_KEY` and `OPENSANCTIONS_API_KEY` exported; secrets are never stored in `main.bicepparam`.
With no environment overrides, `deploy.sh` defaults to `RESOURCE_GROUP=leadscout-prototype-rg` and `LOCATION=germanywestcentral`.
On a subscription that forbids ACR Tasks (see below), build locally instead:
`BUILD_MODE=local RESOURCE_GROUP=leadscout-rg LOCATION=polandcentral infra/deploy.sh` — it needs a running Docker daemon and pushes with a short-lived `az acr login --expose-token` token, which is never written to disk.
The actually deployed stack used exactly this local-build override — `RESOURCE_GROUP=leadscout-rg`, `LOCATION=polandcentral`, not the script's defaults; see "What was actually deployed" below for why.
To give someone read-only access, invite them as a guest Microsoft account, then grant resource-group read access, using whichever resource group name that deployment actually used:
`az role assignment create --assignee user@example.com --role Reader --scope /subscriptions/<subscription-id>/resourceGroups/leadscout-rg` (or `leadscout-prototype-rg` if deployed with the script's defaults).
Use `infra/deploy.sh teardown` when finished; deletion requires typing `delete`.

The same OCI image is portable to AWS App Runner/ECS or OCI Container Instances; replace only the Azure-specific identity, secret, and persistent-volume wiring.

## What was actually deployed (2026-09-19)

Azure for Students restricts regions by policy (allowed: italynorth, spaincentral, belgiumcentral, polandcentral, denmarkeast), so the stack runs in **polandcentral**, in resource group **`leadscout-rg`** — not `main.bicep`'s default `germanywestcentral` or `deploy.sh`'s default `leadscout-prototype-rg`. ACR Tasks (cloud build) are not permitted on this subscription; the image is built with Docker in WSL and pushed with a short-lived `az acr login --expose-token` token. Key Vault purge protection cannot be explicitly disabled (property omitted). Verified live: `GET /health` → 200; `POST /leads` (Artizan Bakery) → 200 in 27 s with all providers and provenance running. The app scales to zero; first request after idle takes ~20 s.
