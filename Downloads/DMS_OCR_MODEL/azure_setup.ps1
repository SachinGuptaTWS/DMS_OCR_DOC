# Azure Setup Script for DMS OCR (Development/Testing)

# Login to Azure (will open browser for authentication)
az login

# Set your subscription
$subscriptionId = "b6b02c6a-3981-420c-9018-270723d41506"
az account set --subscription $subscriptionId

# Create a resource group
$resourceGroup = "dms-ocr-dev-rg"
$location = "eastus"  # Using East US as it has good availability of free tier services

# Create resource group
Write-Host "Creating Resource Group..."
az group create --name $resourceGroup --location $location

# Create Storage Account for file storage
$storageAccount = "dmsocrstorage$(Get-Random -Minimum 1000 -Maximum 9999)"
Write-Host "Creating Storage Account: $storageAccount"
az storage account create `
    --name $storageAccount `
    --resource-group $resourceGroup `
    --location $location `
    --sku Standard_LRS `
    --kind StorageV2 `
    --allow-blob-public-access false

# Create a container in the storage account
$containerName = "documents"
Write-Host "Creating container: $containerName"
az storage container create `
    --name $containerName `
    --account-name $storageAccount `
    --auth-mode login `
    --public-access off

# Create App Service Plan (Basic B1 tier - low cost)
$appServicePlan = "dmsocr-plan"
Write-Host "Creating Basic B1 App Service Plan..."
az appservice plan create `
    --name $appServicePlan `
    --resource-group $resourceGroup `
    --sku B1 `
    --is-linux

# Create Web App for backend
$webAppName = "dmsocr-backend-$(Get-Random -Minimum 1000 -Maximum 9999)"
Write-Host "Creating Web App: $webAppName"
az webapp create `
    --name $webAppName `
    --resource-group $resourceGroup `
    --plan $appServicePlan `
    --runtime "PYTHON:3.11"

# Configure Python version
Write-Host "Configuring Python version..."
az webapp config appsettings set `
    --name $webAppName `
    --resource-group $resourceGroup `
    --settings "SCM_DO_BUILD_DURING_DEPLOYMENT=true" `
    "PYTHON_VERSION=3.11" `
    "WEBSITE_NODE_DEFAULT_VERSION=18.x"

# Get storage connection string
Write-Host "Getting Storage Connection String..."
$connectionString = az storage account show-connection-string `
    --name $storageAccount `
    --resource-group $resourceGroup `
    --query "connectionString" `
    --output tsv

# Set environment variables for the Web App
Write-Host "Configuring Web App settings..."
az webapp config appsettings set `
    --name $webAppName `
    --resource-group $resourceGroup `
    --settings "AZURE_STORAGE_CONNECTION_STRING=$connectionString" `
    "STORAGE_CONTAINER_NAME=$containerName" `
    "AZURE_OPENAI_API_KEY=your-azure-openai-key" `
    "AZURE_OPENAI_ENDPOINT=your-azure-openai-endpoint" `
    "STORAGE_ACCOUNT_NAME=$storageAccount"

# Output important information
Write-Host "`n========================================"
Write-Host "Deployment Summary:"
Write-Host "========================================"
Write-Host "Resource Group: $resourceGroup"
Write-Host "Storage Account: $storageAccount"
Write-Host "Container Name: $containerName"
Write-Host "Web App Name: $webAppName"
Write-Host "`nNext Steps:"
Write-Host "1. Update your backend code to use Azure Blob Storage"
Write-Host "   - Update file operations to use Azure Storage SDK"
Write-Host "   - Update CORS settings to allow your frontend domain"
Write-Host "2. Deploy your backend code to the Web App using:"
Write-Host "   az webapp up --name $webAppName --resource-group $resourceGroup"
Write-Host "3. Update frontend API endpoints to point to: https://$webAppName.azurewebsites.net"
Write-Host "4. Deploy your frontend code to Static Web Apps or another hosting service"
Write-Host "`nNote: Make sure to replace the Azure OpenAI credentials with your actual values"
Write-Host "========================================"
