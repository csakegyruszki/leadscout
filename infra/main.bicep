targetScope = 'resourceGroup'

@description('Azure region for every regional resource.')
param location string = 'germanywestcentral'

@description('Container App name and naming prefix.')
param appName string = 'leadscout'

@description('Initial OCI image; deploy.sh replaces it with the newly built ACR image.')
param image string

@description('Comma-separated model routing configuration.')
param leadscoutModels string

@description('Sender address written into generated sales messages.')
param salesRepEmail string

@description('Minimum lead-fit score accepted by the application.')
param fitThreshold string

@secure()
@description('Bootstrap value stored in Key Vault; never put this in a parameter file.')
param openRouterApiKey string

@secure()
@description('Bootstrap value stored in Key Vault; never put this in a parameter file.')
param openSanctionsApiKey string

@secure()
param braveApiKey string = ''

@secure()
param cloudflareApiToken string = ''

@secure()
param githubToken string = ''

param cloudflareAccountId string = ''

param leadscoutProfile string = 'demo'

var suffix = uniqueString(resourceGroup().id)
var workspaceName = '${appName}-logs-${suffix}'
var environmentName = '${appName}-env-${suffix}'
var identityName = '${appName}-id-${suffix}'
var acrName = take(replace('${appName}${suffix}', '-', ''), 50)
var vaultName = take('${appName}-kv-${suffix}', 24)
var storageName = take(replace('${appName}${suffix}', '-', ''), 24)
var acrPullRoleDefinitionId = subscriptionResourceId('Microsoft.Authorization/roleDefinitions', '7f951dda-4ed3-4680-a7ca-43fe172d538d')
var keyVaultSecretsUserRoleDefinitionId = subscriptionResourceId('Microsoft.Authorization/roleDefinitions', '4633458b-17de-408a-b874-0445c86b69e6')

resource workspace 'Microsoft.OperationalInsights/workspaces@2022-10-01' = {
  name: workspaceName
  location: location
  properties: {
    sku: {
      name: 'PerGB2018'
    }
    retentionInDays: 30
  }
}

resource environment 'Microsoft.App/managedEnvironments@2024-03-01' = {
  name: environmentName
  location: location
  properties: {
    appLogsConfiguration: {
      destination: 'log-analytics'
      logAnalyticsConfiguration: {
        customerId: workspace.properties.customerId
        sharedKey: workspace.listKeys().primarySharedKey
      }
    }
  }
}

resource identity 'Microsoft.ManagedIdentity/userAssignedIdentities@2023-01-31' = {
  name: identityName
  location: location
}

resource acr 'Microsoft.ContainerRegistry/registries@2023-07-01' = {
  name: acrName
  location: location
  sku: {
    name: 'Basic'
  }
  properties: {
    adminUserEnabled: false
    publicNetworkAccess: 'Enabled'
  }
}

resource vault 'Microsoft.KeyVault/vaults@2023-07-01' = {
  name: vaultName
  location: location
  properties: {
    tenantId: subscription().tenantId
    sku: {
      family: 'A'
      name: 'standard'
    }
    enableRbacAuthorization: true
    softDeleteRetentionInDays: 7
    publicNetworkAccess: 'Enabled'
  }
}

// Creating the secrets here lets the app's Key Vault references resolve on its first revision.
resource openRouterSecret 'Microsoft.KeyVault/vaults/secrets@2023-07-01' = {
  parent: vault
  name: 'OPENROUTER-API-KEY'
  properties: {
    value: openRouterApiKey
  }
}

resource openSanctionsSecret 'Microsoft.KeyVault/vaults/secrets@2023-07-01' = {
  parent: vault
  name: 'OPENSANCTIONS-API-KEY'
  properties: {
    value: openSanctionsApiKey
  }
}

resource braveSecret 'Microsoft.KeyVault/vaults/secrets@2023-07-01' = {
  parent: vault
  name: 'BRAVE-API-KEY'
  properties: {
    value: braveApiKey
  }
}

resource cloudflareSecret 'Microsoft.KeyVault/vaults/secrets@2023-07-01' = {
  parent: vault
  name: 'CLOUDFLARE-API-TOKEN'
  properties: {
    value: cloudflareApiToken
  }
}

resource githubSecret 'Microsoft.KeyVault/vaults/secrets@2023-07-01' = {
  parent: vault
  name: 'GITHUB-TOKEN'
  properties: {
    value: githubToken
  }
}

resource storage 'Microsoft.Storage/storageAccounts@2023-05-01' = {
  name: storageName
  location: location
  sku: {
    name: 'Standard_LRS'
  }
  kind: 'StorageV2'
  properties: {
    allowBlobPublicAccess: false
    minimumTlsVersion: 'TLS1_2'
    supportsHttpsTrafficOnly: true
  }
}

resource blobService 'Microsoft.Storage/storageAccounts/blobServices@2023-05-01' = {
  parent: storage
  name: 'default'
}

resource blobOut 'Microsoft.Storage/storageAccounts/blobServices/containers@2023-05-01' = {
  parent: blobService
  name: 'out'
  properties: {
    publicAccess: 'None'
  }
}

resource fileService 'Microsoft.Storage/storageAccounts/fileServices@2023-05-01' = {
  parent: storage
  name: 'default'
}

resource fileOut 'Microsoft.Storage/storageAccounts/fileServices/shares@2023-05-01' = {
  parent: fileService
  name: 'out'
  properties: {
    accessTier: 'TransactionOptimized'
    enabledProtocols: 'SMB'
    shareQuota: 5
  }
}

// The environment owns the storage binding so app revisions can share durable output.
resource environmentStorage 'Microsoft.App/managedEnvironments/storages@2024-03-01' = {
  parent: environment
  name: 'out'
  properties: {
    azureFile: {
      accountName: storage.name
      accountKey: storage.listKeys().keys[0].value
      accessMode: 'ReadWrite'
      shareName: fileOut.name
    }
  }
}

resource acrPull 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  scope: acr
  name: guid(acr.id, identity.id, acrPullRoleDefinitionId)
  properties: {
    principalId: identity.properties.principalId
    principalType: 'ServicePrincipal'
    roleDefinitionId: acrPullRoleDefinitionId
  }
}

resource vaultSecretsUser 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  scope: vault
  name: guid(vault.id, identity.id, keyVaultSecretsUserRoleDefinitionId)
  properties: {
    principalId: identity.properties.principalId
    principalType: 'ServicePrincipal'
    roleDefinitionId: keyVaultSecretsUserRoleDefinitionId
  }
}

resource app 'Microsoft.App/containerApps@2024-03-01' = {
  name: appName
  location: location
  identity: {
    type: 'UserAssigned'
    userAssignedIdentities: {
      '${identity.id}': {}
    }
  }
  properties: {
    managedEnvironmentId: environment.id
    configuration: {
      activeRevisionsMode: 'Single'
      ingress: {
        external: true
        targetPort: 8000
        transport: 'auto'
        traffic: [
          {
            latestRevision: true
            weight: 100
          }
        ]
      }
      registries: [
        {
          server: acr.properties.loginServer
          identity: identity.id
        }
      ]
      secrets: [
        {
          name: 'openrouter-api-key'
          keyVaultUrl: '${vault.properties.vaultUri}secrets/${openRouterSecret.name}'
          identity: identity.id
        }
        {
          name: 'opensanctions-api-key'
          keyVaultUrl: '${vault.properties.vaultUri}secrets/${openSanctionsSecret.name}'
          identity: identity.id
        }
        {
          name: 'brave-api-key'
          keyVaultUrl: '${vault.properties.vaultUri}secrets/${braveSecret.name}'
          identity: identity.id
        }
        {
          name: 'cloudflare-api-token'
          keyVaultUrl: '${vault.properties.vaultUri}secrets/${cloudflareSecret.name}'
          identity: identity.id
        }
        {
          name: 'github-token'
          keyVaultUrl: '${vault.properties.vaultUri}secrets/${githubSecret.name}'
          identity: identity.id
        }
      ]
    }
    template: {
      containers: [
        {
          name: appName
          image: image
          env: [
            {
              name: 'OPENROUTER_API_KEY'
              secretRef: 'openrouter-api-key'
            }
            {
              name: 'OPENSANCTIONS_API_KEY'
              secretRef: 'opensanctions-api-key'
            }
            {
              name: 'BRAVE_API_KEY'
              secretRef: 'brave-api-key'
            }
            {
              name: 'CLOUDFLARE_API_TOKEN'
              secretRef: 'cloudflare-api-token'
            }
            {
              name: 'GITHUB_TOKEN'
              secretRef: 'github-token'
            }
            {
              name: 'CLOUDFLARE_ACCOUNT_ID'
              value: cloudflareAccountId
            }
            {
              name: 'LEADSCOUT_PROFILE'
              value: leadscoutProfile
            }
            {
              name: 'LEADSCOUT_MODELS'
              value: leadscoutModels
            }
            {
              name: 'SALES_REP_EMAIL'
              value: salesRepEmail
            }
            {
              name: 'FIT_THRESHOLD'
              value: fitThreshold
            }
          ]
          resources: {
            cpu: json('0.5')
            memory: '1Gi'
          }
          volumeMounts: [
            {
              volumeName: 'out'
              mountPath: '/app/out'
            }
          ]
        }
      ]
      scale: {
        minReplicas: 0
        maxReplicas: 2
      }
      volumes: [
        {
          name: 'out'
          storageName: environmentStorage.name
          storageType: 'AzureFile'
        }
      ]
    }
  }
  dependsOn: [
    acrPull
    vaultSecretsUser
    openRouterSecret
    openSanctionsSecret
  ]
}

output appFqdn string = app.properties.configuration.ingress.fqdn
output containerAppName string = app.name
output acrLoginServer string = acr.properties.loginServer
output acrName string = acr.name
output vaultName string = vault.name
output storageAccountName string = storage.name
