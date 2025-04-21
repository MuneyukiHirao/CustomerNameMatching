import os
os.environ['NO_DOTENV'] = '1'
os.environ['DATABASE_URL'] = 'sqlite:///./dev.db'
os.environ.pop('KEYVAULT_NAME', None)
os.environ.pop('SQL_CONNECTION_STRING_SECRET_NAME', None)
os.environ.pop('BLOB_CONNECTION_STRING_SECRET_NAME', None)
os.environ['BLOB_CONNECTION_STRING'] = 'UseDevelopmentStorage=true'
os.environ['BLOB_ACCOUNT_KEY'] = 'dummykey'
os.environ['BLOB_CONTAINER_NAME'] = 'dummycontainer'

import pytest
from backend.pipeline import _get_container_client, _generate_sas

# Dummy service/client for testing
class DummyService:
    def __init__(self, container_name=None):
        self.container_name = container_name
    def get_container_client(self, name):
        return f"client_for:{name}"

class DummyBlob:
    url = "http://dummy/blob"
    def __init__(self, should_fail=False):
        self._fail = should_fail
    def generate_shared_access_signature(self, permission=None, expiry=None):
        if self._fail:
            raise Exception("failure")
        return "gen_sas"


def test_get_container_client_missing_env():
    os.environ.pop("BLOB_CONTAINER_NAME", None)
    with pytest.raises(RuntimeError) as exc:
        _get_container_client("connstr")
    assert "BLOB_CONTAINER_NAME is not configured" in str(exc.value)


def test_get_container_client_success(monkeypatch):
    os.environ['BLOB_CONTAINER_NAME'] = 'mycontainer'
    # Stub SAS helper's BlobServiceClient
    import backend.sas_helpers as sh
    dummy = DummyService()
    # Stub from_connection_string to return dummy
    monkeypatch.setattr(sh.BlobServiceClient, 'from_connection_string', staticmethod(lambda conn: dummy))
    client = _get_container_client('connstr')
    assert client == 'client_for:mycontainer'


def test_generate_sas_missing_key():
    os.environ.pop('BLOB_ACCOUNT_KEY', None)
    blob = DummyBlob()
    with pytest.raises(RuntimeError) as exc:
        _generate_sas(blob)
    assert 'BLOB_ACCOUNT_KEY' in str(exc.value)


def test_generate_sas_env_override():
    os.environ['BLOB_ACCOUNT_KEY'] = 'key'
    os.environ['BLOB_SAS_TOKEN'] = 'tok'
    blob = DummyBlob()
    url = _generate_sas(blob)
    assert url.endswith('?tok')


def test_generate_sas_blob_success(monkeypatch):
    os.environ['BLOB_ACCOUNT_KEY'] = 'key'
    os.environ.pop('BLOB_SAS_TOKEN', None)
    blob = DummyBlob()
    url = _generate_sas(blob)
    assert url.endswith('?gen_sas')


def test_generate_sas_blob_failure(monkeypatch):
    os.environ['BLOB_ACCOUNT_KEY'] = 'key'
    os.environ.pop('BLOB_SAS_TOKEN', None)
    blob = DummyBlob(should_fail=True)
    with pytest.raises(RuntimeError) as exc:
        _generate_sas(blob)
    assert 'SAS generation via BlobClient failed' in str(exc.value)
