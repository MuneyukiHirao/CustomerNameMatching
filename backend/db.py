import os
import re
import logging
import pyodbc
try:
    from azure.identity import DefaultAzureCredential
    from azure.keyvault.secrets import SecretClient
except ImportError:
    DefaultAzureCredential = None
    SecretClient = None
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from urllib.parse import quote_plus

# Retrieve SQL connection string from Azure Key Vault
def get_connection_string() -> str:
    # Local override: direct DATABASE_URL env (for testing or local dev)
    direct = os.getenv("DATABASE_URL")
    if direct:
        return direct
    # If not using Key Vault creds, or retrieval fails, fallback to local SQLite
    vault_name = os.getenv("KEYVAULT_NAME")
    secret_name = os.getenv("SQL_CONNECTION_STRING_SECRET_NAME")
    if not vault_name or not secret_name:
        return "sqlite:///./dev.db"
    vault_url = f"https://{vault_name}.vault.azure.net"
    try:
        credential = DefaultAzureCredential()
        client = SecretClient(vault_url=vault_url, credential=credential)
        secret = client.get_secret(secret_name)
        conn_str = secret.value
    except Exception as e:
        logging.warning(f"KeyVault SQL retrieval failed: {e}; falling back to SQLite")
        return "sqlite:///./dev.db"
    # Convert Azure connection string to SQLAlchemy URL
    match = re.match(
        r"Server=tcp:(?P<host>[^,]+),(?P<port>\d+);Initial Catalog=(?P<catalog>[^;]+);User ID=(?P<user>[^;]+);Password=(?P<pwd>.+)", conn_str
    )
    if match:
        gd = match.groupdict()
        user = quote_plus(gd['user'])
        pwd = quote_plus(gd['pwd'])
        host = gd['host']
        port = gd['port']
        catalog = gd['catalog']
        # Dynamically select installed ODBC Driver
        drivers = pyodbc.drivers()
        logging.info(f"Available ODBC drivers: {drivers}")
        driver = next((d for d in drivers if d.startswith("ODBC Driver")), None)
        if not driver:
            raise RuntimeError(f"No suitable ODBC Driver found. Installed drivers: {drivers}")
        drv = quote_plus(driver)
        return f"mssql+pyodbc://{user}:{pwd}@{host}:{port}/{catalog}?driver={drv}"
    # Fallback to raw value if not matching
    return conn_str

# Retrieve Blob connection string from Azure Key Vault or env
def get_blob_secret() -> str:
    # Local override: direct env var
    direct = os.getenv("BLOB_CONNECTION_STRING")
    if direct:
        return direct
    vault_name = os.getenv("KEYVAULT_NAME")
    secret_name = os.getenv("BLOB_CONNECTION_STRING_SECRET_NAME")
    if not vault_name or not secret_name:
        raise RuntimeError("Key Vault info for BLOB connection string not configured")
    vault_url = f"https://{vault_name}.vault.azure.net"
    credential = DefaultAzureCredential()
    client = SecretClient(vault_url=vault_url, credential=credential)
    return client.get_secret(secret_name).value

# SQLAlchemy setup
import logging
try:
    DATABASE_URL = get_connection_string()
except Exception as e:
    logging.warning(f"get_connection_string failed: {e}; falling back to SQLite")
    DATABASE_URL = "sqlite:///./dev.db"
engine = create_engine(DATABASE_URL, echo=False)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()
