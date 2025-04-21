import os
from urllib.parse import urlparse
from azure.storage.blob import BlobServiceClient, BlobSasPermissions, generate_blob_sas
from datetime import datetime, timedelta


def _get_container_client(blob_secret: str):
    """Retrieve container client, validating env."""
    container_name = os.getenv('BLOB_CONTAINER_NAME')
    if not container_name:
        raise RuntimeError('BLOB_CONTAINER_NAME is not configured')
    service = BlobServiceClient.from_connection_string(blob_secret)
    return service.get_container_client(container_name)


def _generate_sas(blob_client):
    """Generate SAS URL, with env override, client-side, or parsed URL fallback."""
    account_key = os.getenv('BLOB_ACCOUNT_KEY')
    if not account_key:
        raise RuntimeError("BLOB_ACCOUNT_KEY is not configured")

    # Env override SAS token
    env_sas = os.getenv('BLOB_SAS_TOKEN')
    if env_sas:
        return f"{blob_client.url}?{env_sas}"

    # Client-side generation if supported
    if hasattr(blob_client, 'generate_shared_access_signature'):
        try:
            sas = blob_client.generate_shared_access_signature(
                permission=BlobSasPermissions(read=True),
                expiry=datetime.utcnow() + timedelta(hours=1)
            )
        except Exception as e:
            raise RuntimeError(f"SAS generation via BlobClient failed: {e}")
        return f"{blob_client.url}?{sas}"

    # Fallback: parse URL to determine account, container, and blob
    parsed = urlparse(blob_client.url)
    account_name = os.getenv('BLOB_ACCOUNT') or parsed.netloc.split('.')[0]
    path_parts = parsed.path.lstrip('/').split('/', 1)
    if len(path_parts) == 2:
        container_name, blob_name = path_parts
    else:
        container_name = os.getenv('BLOB_CONTAINER_NAME')
        blob_name = path_parts[0]

    try:
        sas = generate_blob_sas(
            account_name=account_name,
            container_name=container_name,
            blob_name=blob_name,
            account_key=account_key,
            permission=BlobSasPermissions(read=True),
            expiry=datetime.utcnow() + timedelta(hours=1)
        )
    except Exception as e:
        raise RuntimeError(f"SAS generation via BlobClient failed: {e}")

    return f"{blob_client.url}?{sas}"
