# Databricks notebook source
# MAGIC %md
# MAGIC #Mount Storage Container
# MAGIC

# COMMAND ----------

# Secrets using key vault
client_id = dbutils.secrets.get(scope='de-bd-project-scope', key='de-bd-app-client-id')
client_secret = dbutils.secrets.get(scope='de-bd-project-scope', key='de-bd-app-client-secret')
tenant_id = dbutils.secrets.get(scope='de-bd-project-scope', key='de-bd-app-tenant-id')

# COMMAND ----------

# Lists secret scopes
dbutils.secrets.listScopes()

# COMMAND ----------

def mount_adls(storage_account_name, container_name):
    # Get secrets from key vault
    client_id = dbutils.secrets.get(scope='de-bd-project-scope', key='de-bd-app-client-id')
    client_secret = dbutils.secrets.get(scope='de-bd-project-scope', key='de-bd-app-client-secret')
    tenant_id = dbutils.secrets.get(scope='de-bd-project-scope', key='de-bd-app-tenant-id')

    # Set spark configurations
    configs = {
        "fs.azure.account.auth.type": "OAuth",
        "fs.azure.account.oauth.provider.type": "org.apache.hadoop.fs.azurebfs.oauth2.ClientCredsTokenProvider",
        "fs.azure.account.oauth2.client.id": client_id,
        "fs.azure.account.oauth2.client.secret": client_secret,
        "fs.azure.account.oauth2.client.endpoint": f"https://login.microsoftonline.com/{tenant_id}/oauth2/token",
        }
    
    # Checks if the container has already been mounted. If True, it will unmount and re-mount, and continue with the next one
    if any(mount.mountPoint == f"/mnt/{storage_account_name}/{container_name}" for mount in dbutils.fs.mounts()):
        dbutils.fs.unmount(f"/mnt/{storage_account_name}/{container_name}")
        

    # Mount the storage account container
    dbutils.fs.mount(
        source= f"abfss://{container_name}@{storage_account_name}.dfs.core.windows.net/",
        mount_point= f"/mnt/{storage_account_name}/{container_name}",
        extra_configs=configs,
        )
    
    # Display Mounts
    display(dbutils.fs.mounts())

# COMMAND ----------

mount_adls("debdingestiondl", "de-bd-project")

# COMMAND ----------


