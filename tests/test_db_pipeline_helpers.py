import os
os.environ['NO_DOTENV'] = '1'
os.environ.pop('OPENAI_API_KEY', None)
import backend.pipeline as pipeline
pipeline.client = None

import pytest
import pandas as pd
import numpy as np
from backend.db import get_connection_string, get_blob_secret
from backend.pipeline import normalize_and_embed, validate_mappings
from backend.models import CustomerStdA

# DB helper tests
def test_get_connection_string_direct():
    os.environ['DATABASE_URL'] = 'sqlite:///tmp.db'
    assert get_connection_string() == 'sqlite:///tmp.db'

def test_get_connection_string_fallback_local(monkeypatch):
    os.environ.pop('DATABASE_URL', None)
    os.environ.pop('KEYVAULT_NAME', None)
    os.environ.pop('SQL_CONNECTION_STRING_SECRET_NAME', None)
    url = get_connection_string()
    assert url.startswith('sqlite:///')

def test_get_connection_string_keyvault_error(monkeypatch):
    os.environ.pop('DATABASE_URL', None)
    os.environ['KEYVAULT_NAME'] = 'vault'
    os.environ['SQL_CONNECTION_STRING_SECRET_NAME'] = 'secret'
    # simulate KeyVault access failure
    import backend.db as dbmod
    monkeypatch.setattr(dbmod, 'DefaultAzureCredential', lambda: (_ for _ in ()).throw(Exception('fail')))
    monkeypatch.setattr(dbmod, 'SecretClient', lambda *args, **kwargs: (_ for _ in ()).throw(Exception('fail')))
    url = get_connection_string()
    assert url.startswith('sqlite:///')

def test_get_blob_secret_direct():
    os.environ['BLOB_CONNECTION_STRING'] = 'UseDevStorage'
    assert get_blob_secret() == 'UseDevStorage'

def test_get_blob_secret_keyvault_missing():
    os.environ.pop('BLOB_CONNECTION_STRING', None)
    os.environ.pop('KEYVAULT_NAME', None)
    os.environ.pop('BLOB_CONNECTION_STRING_SECRET_NAME', None)
    with pytest.raises(RuntimeError):
        get_blob_secret()

# Pipeline helper tests
def test_normalize_and_embed_default():
    df = pd.DataFrame({'raw_name': ['a '], 'raw_code': [' x'], 'cust_id': ['1'], 'lang': ['en']})
    result = normalize_and_embed(df)
    assert result.loc[0, 'norm_name'] == 'A'
    assert result.loc[0, 'norm_code'] == 'X'
    vec = result.loc[0, 'vector']
    assert isinstance(vec, (bytes, bytearray))
    arr = np.frombuffer(vec, dtype=np.float32)
    assert arr.shape == (384,)

def test_validate_mappings_success():
    mappings = [{'cust_id': '1', 'lang': 'en'}]
    # should not raise
    validate_mappings(CustomerStdA, mappings)

def test_validate_mappings_failure():
    with pytest.raises(ValueError):
        validate_mappings(CustomerStdA, [{'invalid': 1}])
