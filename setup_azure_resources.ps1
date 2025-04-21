# PowerShell script to create Azure resources for Customer Master Matching

# 1. Login to Azure
az login

# 2. Create Resource Group
# az group create --name hirao-test_group --location japaneast

# 3. Create Storage Account
az storage account create `
  --name custmatchstorageacct1 `
  --resource-group hirao-test_group `
  --location japaneast `
  --sku Standard_LRS

# 4. Create Blob Container (AAD auth)
az storage container create `
  --account-name custmatchstorageacct1 `
  --name exports `
  --public-access off `
  --auth-mode login

# 5. Create Key Vault
az keyvault create `
  --name CustomerMatcherKeyVault1 `
  --resource-group hirao-test_group `
  --location japaneast

# 6. Store Storage connection string in Key Vault
$ST_CONN = az storage account show-connection-string `
  --name custmatchstorageacct1 `
  --resource-group hirao-test_group `
  --query connectionString -o tsv
az keyvault secret set `
  --vault-name CustomerMatcherKeyVault1 `
  --name blob-connection-string `
  --value "$ST_CONN"

# 7. Create SQL Server
az sql server create --name customer-matcher-sql-server-001 `
  --resource-group hirao-test_group `
  --location japaneast `
  --admin-user mhirao  `
  --admin-password Welcome#1

# 8. Create SQL Database
az sql db create `
  --resource-group hirao-test_group `
  --server customer-matcher-sql-server-001 `
  --name CustomerMatchingDB `
  --service-objective S0

# 9. Configure Firewall for your IP
$MY_IP = (curl.exe ifconfig.me)
az sql server firewall-rule create `
  --resource-group hirao-test_group `
  --server customer-matcher-sql-server-001 `
  --name AllowMyIP `
  --start-ip-address $MY_IP `
  --end-ip-address $MY_IP

# 10. Assign Key Vault Secrets Officer role to your account
$VAULT_ID = az keyvault show --name CustomerMatcherKeyVault1 --query id -o tsv
$APPID = az account show --query user.name -o tsv
az role assignment create --role "Key Vault Secrets Officer" --assignee muneyuki_hirao_global.komatsu#EXT#@hirotakaoohashikomatsuco.onmicrosoft.com --scope $VAULT_ID

# 11. Store SQL connection string in Key Vault
$SQL_CONN = "Server=tcp:customer-matcher-sql-server-001.database.windows.net,1433;Initial Catalog=CustomerMatchingDB;User ID=mhirao;Password=Welcome#1"
az keyvault secret set `
  --vault-name CustomerMatcherKeyVault1 `
  --name sql-connection-string `
  --value "$SQL_CONN"

Write-Host "Azure resource setup complete"

# Key Vault のリソースID を取得
$VAULT_ID = az keyvault show --name CustomerMatcherKeyVault1 --query id -o tsv

# SP のクライアントID（.env に設定した AZURE_CLIENT_ID）を使って割り当て
az role assignment create `
  --role "Key Vault Secrets Officer" `
  --assignee 539d080b-a9fb-4390-8b43-82c7ebaf773f `
  --scope $VAULT_ID

# SQL Server に自分の IP を追加
az sql server firewall-rule create `
  --resource-group hirao-test_group `
  --server customer-matcher-sql-server-001 `
  --name AllowMyIP `
  --start-ip-address 223.135.207.25 `
  --end-ip-address 223.135.207.25

# Ensure OPENAI_API_KEY env var is set
if (-not $env:OPENAI_API_KEY) {
  Write-Error "Please set OPENAI_API_KEY environment variable before running script"
  exit 1
}
# Upload OpenAI API key to Key Vault
az keyvault secret set --vault-name CustomerMatcherKeyVault1 --name openai-api-key --value $env:OPENAI_API_KEY

# Create App Service Plan
az appservice plan create --name custmatch-plan --resource-group hirao-test_group --is-linux --sku B1

# Create Web Apps (use `--%` immediately after az to stop PowerShell parsing for `|`)
az webapp create --resource-group hirao-test_group --plan custmatch-plan --name custmatch-back --runtime "PYTHON|3.10"
az webapp create --resource-group hirao-test_group --plan custmatch-plan --name custmatch --runtime "NODE|16-lts"

# Wait for Web Apps provisioning
Start-Sleep -Seconds 30

# Configure backend App Settings
az webapp config appsettings set --resource-group hirao-test_group --name custmatch-back --settings SQL_CONN="$SQL_CONN" KEYVAULT_NAME="CustomerMatcherKeyVault1" OPENAI_API_KEY="$(az keyvault secret show --vault-name CustomerMatcherKeyVault1 --name openai-api-key --query value -o tsv)"
Write-Host "App Service setup complete"