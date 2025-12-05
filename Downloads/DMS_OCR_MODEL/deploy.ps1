# Login to Azure
az login

# Set your subscription
$subscriptionId = "b6b02c6a-3981-420c-9018-270723d41506"
az account set --subscription $subscriptionId

# Set variables
$resourceGroup = "dms-ocr-dev-rg"
$location = "eastus"
$acrName = "dmsocr$(Get-Random -Minimum 1000 -Maximum 9999)"
$backendAppName = "dmsocr-backend-$(Get-Random -Minimum 1000 -Maximum 9999)"
$imageName = "$($acrName).azurecr.io/dms-ocr-backend:latest"

# Build the Docker image
docker build -t $imageName .

# Create Azure Container Registry (ACR)
Write-Host "Creating Azure Container Registry..."
az acr create --resource-group $resourceGroup --name $acrName --sku Basic --admin-enabled true

# Login to ACR
$acrUsername = az acr credential show --name $acrName --query "username" -o tsv
$acrPassword = az acr credential show --name $acrName --query "passwords[0].value" -o tsv
docker login "$acrName.azurecr.io" --username $acrUsername --password $acrPassword

# Push the image to ACR
docker push $imageName

# Create Container Apps environment if it doesn't exist
$envExists = az containerapp env show --name "dmsocr-env" --resource-group $resourceGroup --query "name" -o tsv 2>$null
if (-not $envExists) {
    Write-Host "Creating Container Apps Environment..."
    az containerapp env create `
        --name "dmsocr-env" `
        --resource-group $resourceGroup `
        --location $location
}

# Get storage connection string
$storageAccount = az storage account list --resource-group $resourceGroup --query "[0].name" -o tsv
$connectionString = az storage account show-connection-string `
    --name $storageAccount `
    --resource-group $resourceGroup `
    --query "connectionString" `
    --output tsv

# Create or update the container app
Write-Host "Deploying Container App..."
az containerapp create `
    --name $backendAppName `
    --resource-group $resourceGroup `
    --environment "dmsocr-env" `
    --image $imageName `
    --target-port 8000 `
    --ingress external `
    --min-replicas 1 `
    --max-replicas 1 `
    --cpu 0.5 `
    --memory 1Gi `
    --env-vars `
        "AZURE_STORAGE_CONNECTION_STRING=$connectionString" `
        "STORAGE_CONTAINER_NAME=documents" `
        "AZURE_OPENAI_API_KEY=your-azure-openai-key" `
        "AZURE_OPENAI_ENDPOINT=your-azure-openai-endpoint"

# Get the public URL
$url = az containerapp show `
    --name $backendAppName `
    --resource-group $resourceGroup `
    --query "properties.configuration.ingress.fqdn" `
    --output tsv

Write-Host "`n========================================"
Write-Host "Deployment Complete!"
Write-Host "Backend URL: https://$url"
Write-Host "`nNext Steps:"
Write-Host "1. Update your frontend to use: https://$url"
Write-Host "2. Test the API endpoints"
Write-Host "3. Check logs with: az containerapp logs show --name $backendAppName --resource-group $resourceGroup"
Write-Host "========================================"
