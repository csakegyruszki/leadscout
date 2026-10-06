using './main.bicep'

param location = 'polandcentral'
param appName = 'leadscout'
param image = 'mcr.microsoft.com/azuredocs/containerapps-helloworld:latest'
param leadscoutModels = 'openrouter:deepseek/deepseek-chat-v3.1,cloudflare:@cf/meta/llama-3.3-70b-instruct-fp8-fast,openrouter:deepseek/deepseek-v4-flash-0731:free,openrouter:nvidia/nemotron-3-super-120b-a12b:free'
param salesRepEmail = 'sales@example.invalid'
param fitThreshold = '60'

// Secure parameters are intentionally supplied from environment variables by deploy.sh.
