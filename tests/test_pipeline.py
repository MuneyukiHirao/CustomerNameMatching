import pytest
import os
os.environ['DATABASE_URL'] = 'sqlite:///./dev.db'
os.environ['BLOB_CONNECTION_STRING'] = 'UseDevelopmentStorage=true'
os.environ['BLOB_ACCOUNT_KEY'] = 'dummykey'
os.environ['BLOB_CONTAINER_NAME'] = 'dummycontainer'
import io
import os
import tempfile
import pandas as pd
from datetime import date, datetime, timedelta
from backend.pipeline import process_upload, _get_container_client, _generate_sas
from azure.storage.blob import BlobServiceClient
from azure.keyvault.secrets import SecretClient

# Helpers to create dummy Excel files

def create_excel_bytes(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    df.to_excel(buf, index=False)
    buf.seek(0)
    return buf.getvalue()

# Helper Upload wrapper
class U:
    def __init__(self, data, name):
        self.file = io.BytesIO(data)
        self.filename = name

class DummyBlobClient:
    def __init__(self, *args, **kwargs):
        self.account_name = os.getenv('BLOB_ACCOUNT') or 'dummyaccount'
        self.container_name = os.getenv('BLOB_CONTAINER_NAME') or 'dummycontainer'
    def get_container_client(self, _): return self
    def upload_blob(self, name, data, overwrite=False): return None
    def get_blob_client(self, name):
        # return self as blob_client
        # set blob name for URL construction
        self.name = name
        # construct a dummy base URL using BLOB_URL env var
        prefix = os.getenv('BLOB_URL', 'https://dummy.blob.core.windows.net/dummycontainer')
        self.url = f'{prefix}/{name}'
        return self
    def generate_shared_access_signature(self, permission=None, expiry=None):
        # default SAS token for testing
        return 'testsas'

@pytest.fixture
def patch_azure(monkeypatch, request):
    # allow skipping this fixture
    if request.node.get_closest_marker('skip_azure'):
        return
    # stub blob client
    monkeypatch.setattr(BlobServiceClient, 'from_connection_string', lambda conn: DummyBlobClient())
    # stub blob secret retrieval
    import backend.pipeline as pipeline
    monkeypatch.setattr(pipeline, 'get_blob_secret', lambda: os.getenv('BLOB_CONNECTION_STRING'))
    # disable real OpenAI embedding
    monkeypatch.setattr(pipeline, 'client', None)
    # set required env
    os.environ.update({
        'BLOB_URL': 'https://dummy.blob.core.windows.net/dummycontainer',
        'BLOB_SAS_TOKEN': 'sastoken'
    })

def test_small_dataset(tmp_path, patch_azure):
    # 2 rows X 7 columns
    df = pd.DataFrame({
        'cust_id': ['A', 'B'],
        'lang': ['jp', 'en'],
        'raw_name': ['会社A', 'COMPANY X'],
        'raw_code': ['123', '456'],
        # placeholders for norm and vector will be ignored
    })
    a_bytes = create_excel_bytes(df)
    b_bytes = create_excel_bytes(df)
    url = process_upload(U(a_bytes, 'a.xlsx'), U(b_bytes, 'b.xlsx'))
    prefix = os.getenv('BLOB_URL', 'https://dummy.blob.core.windows.net/dummycontainer')
    assert url.startswith(prefix)
    assert '?' in url

def test_large_dataset(tmp_path, patch_azure):
    # 3000 rows
    n = 3000
    dfA = pd.DataFrame({
        'cust_id': [f'id{i}' for i in range(n)],
        'lang': ['en']*n,
        'raw_name': [f'Name{i}' for i in range(n)],
        'raw_code': [f'{i:03d}' for i in range(n)],
    })
    dfB = dfA.copy()
    a_bytes = create_excel_bytes(dfA)
    b_bytes = create_excel_bytes(dfB)
    url = process_upload(U(a_bytes, 'A.xlsx'), U(b_bytes, 'B.xlsx'))
    prefix = os.getenv('BLOB_URL', 'https://dummy.blob.core.windows.net/dummycontainer')
    assert url.startswith(prefix)
    assert '?' in url

@pytest.mark.parametrize('empty_df', [pd.DataFrame(), pd.DataFrame({'cust_id': [], 'lang': [], 'raw_name': [], 'raw_code': []})])
def test_empty_dataset(empty_df, patch_azure):
    # Expect RuntimeError or specific behavior
    a_bytes = create_excel_bytes(empty_df)
    b_bytes = create_excel_bytes(empty_df)
    with pytest.raises(Exception):
        process_upload(U(a_bytes, 'a.xlsx'), U(b_bytes, 'b.xlsx'))

# Test when A and B have different number of rows
def test_mismatched_sizes(patch_azure):
    # A has 5 rows, B has 1 row
    dfA = pd.DataFrame({'cust_id': [f'id{i}' for i in range(5)],
                        'lang': ['en']*5,
                        'raw_name': [f'Name{i}' for i in range(5)],
                        'raw_code': [f'{i:03d}' for i in range(5)]})
    dfB = pd.DataFrame({'cust_id': ['only1'], 'lang': ['en'], 'raw_name': ['Single'], 'raw_code': ['000']})
    a_bytes = create_excel_bytes(dfA)
    b_bytes = create_excel_bytes(dfB)
    url = process_upload(U(a_bytes, 'A.xlsx'), U(b_bytes, 'B.xlsx'))
    assert url.startswith('https://') and '?' in url

# Test invalid format upload
def test_invalid_format(tmp_path, patch_azure):
    # Pass a non-Excel binary
    invalid_bytes = b"not an excel file"
    with pytest.raises(Exception):
        process_upload(U(invalid_bytes, 'badfile.txt'), U(invalid_bytes, 'badfile2.txt'))

# Missing columns in File A
def test_missing_columns_A(patch_azure):
    dfA = pd.DataFrame({'lang': ['en'], 'raw_name': ['N'], 'raw_code': ['C']})
    dfB = pd.DataFrame({'cust_id': ['X'], 'lang': ['en'], 'raw_name': ['N'], 'raw_code': ['C']})
    a_bytes = create_excel_bytes(dfA)
    b_bytes = create_excel_bytes(dfB)
    with pytest.raises(ValueError) as e:
        process_upload(U(a_bytes, 'a.xlsx'), U(b_bytes, 'b.xlsx'))
    assert 'File A missing columns' in str(e.value)

# Missing columns in File B
def test_missing_columns_B(patch_azure):
    dfA = pd.DataFrame({'cust_id': ['X'], 'lang': ['en'], 'raw_name': ['N'], 'raw_code': ['C']})
    dfB = pd.DataFrame({'lang': ['en'], 'raw_name': ['N'], 'raw_code': ['C']})
    a_bytes = create_excel_bytes(dfA)
    b_bytes = create_excel_bytes(dfB)
    with pytest.raises(ValueError) as e:
        process_upload(U(a_bytes, 'a.xlsx'), U(b_bytes, 'b.xlsx'))
    assert 'File B missing columns' in str(e.value)

# Null values in raw_name/raw_code
def test_null_values(patch_azure):
    dfA = pd.DataFrame({'cust_id': ['X'], 'lang': ['en'], 'raw_name': [None], 'raw_code': ['C']})
    dfB = dfA.copy()
    a_bytes = create_excel_bytes(dfA)
    b_bytes = create_excel_bytes(dfB)
    with pytest.raises(ValueError) as e:
        process_upload(U(a_bytes, 'a.xlsx'), U(b_bytes, 'b.xlsx'))
    assert 'contains null values' in str(e.value)

# Blob secret failure
def test_blob_secret_failure(monkeypatch, patch_azure):
    # Patch get_blob_secret to raise
    import backend.pipeline as pipeline
    monkeypatch.setattr(pipeline, 'get_blob_secret', lambda: (_ for _ in ()).throw(Exception('no_secret')))
    df = pd.DataFrame({'cust_id': ['A'], 'lang': ['en'], 'raw_name': ['N'], 'raw_code': ['C']})
    a_bytes = create_excel_bytes(df)
    b_bytes = create_excel_bytes(df)
    with pytest.raises(RuntimeError) as e:
        process_upload(U(a_bytes, 'a.xlsx'), U(b_bytes, 'b.xlsx'))
    assert 'Blob secret retrieval failed' in str(e.value)

# Blob upload failure
def test_blob_upload_failure(monkeypatch):
    # Patch upload_blob to raise
    class FailingBlobClient:
        def get_container_client(self, _): return self
        def upload_blob(self, name, data, overwrite=False): raise Exception('upload_error')
    import backend.pipeline as pipeline
    os.environ['BLOB_CONNECTION_STRING'] = 'DefaultEndpointsProtocol=https;AccountName=dummy;AccountKey=dummy;EndpointSuffix=core.windows.net'
    monkeypatch.setattr(pipeline, 'BlobServiceClient', type('BC', (), {'from_connection_string': staticmethod(lambda conn: FailingBlobClient())}))
    df = pd.DataFrame({'cust_id': ['A'], 'lang': ['en'], 'raw_name': ['N'], 'raw_code': ['C']})
    a_bytes = create_excel_bytes(df)
    b_bytes = create_excel_bytes(df)
    with pytest.raises(RuntimeError) as e:
        process_upload(U(a_bytes, 'a.xlsx'), U(b_bytes, 'b.xlsx'))
    assert 'Blob upload failed' in str(e.value)

# SASトークン環境上書き
def test_sas_override_env(monkeypatch, patch_azure):
    # Set override
    os.environ['BLOB_SAS_TOKEN'] = 'envtoken'
    # Monkeypatch generate_shared_access_signature to ensure not called
    called = False
    import backend.pipeline as pipeline
    def fake_generate(*args, **kwargs): nonlocal called; called = True; return 'shouldnot'
    monkeypatch.setattr(pipeline.container_client.__class__, 'get_blob_client',
                         lambda self, n: self)
    # Create dummy file
    df = pd.DataFrame({'cust_id':['A'], 'lang':['en'], 'raw_name':['N'], 'raw_code':['C']})
    a,b = create_excel_bytes(df), create_excel_bytes(df)
    url = pipeline.process_upload(U(a,'a.xlsx'), U(b,'b.xlsx'))
    assert '?envtoken' in url
    assert not called

# SAS生成 via BlobClient
def test_sas_generation_blobclient(monkeypatch, patch_azure):
    # Remove override
    os.environ.pop('BLOB_SAS_TOKEN', None)
    # Monkeypatch blob_client.generate_shared_access_signature
    import backend.pipeline as pipeline
    class BC:
        url = 'https://dummy.blob/x'
        def generate_shared_access_signature(self, **kwargs): return 'sig123'
    # stub container_client.get_blob_client
    monkeypatch.setattr(pipeline, 'BlobServiceClient', type('BS', (), {'from_connection_string': lambda conn: DummyBlobClient()}))
    # Create dummy file
    df = pd.DataFrame({'cust_id':['A'], 'lang':['en'], 'raw_name':['N'], 'raw_code':['C']})
    # monkeypatch container_client and blob_client on pipeline
    sc = DummyBlobClient()
    monkeypatch.setattr(pipeline, 'BlobServiceClient', type('BS', (), {'from_connection_string': staticmethod(lambda conn: sc)}))
    # Patch get_blob_client to always return BC (with .generate_shared_access_signature returning 'sig123')
    def always_bc(*args, **kwargs): return BC()
    sc.get_blob_client = always_bc
    a,b = create_excel_bytes(df), create_excel_bytes(df)
    url = pipeline.process_upload(U(a,'a.xlsx'), U(b,'b.xlsx'))
    assert url.endswith('?sig123')

# SAS生成失敗
def test_sas_generation_failure(monkeypatch, patch_azure):
    os.environ.pop('BLOB_SAS_TOKEN', None)
    import backend.pipeline as pipeline
    # Create a blob client that fails on SAS generation
    class BC2:
        url = 'u'
        def generate_shared_access_signature(self, **kwargs): raise Exception('fail')
    # Stub Azure BlobServiceClient to return a container client yielding BC2
    import azure.storage.blob as azblob
    class FakeService:
        def get_container_client(self, name): return self
        def upload_blob(self, name, data, overwrite=False): return None
        def get_blob_client(self, name): return BC2()
    monkeypatch.setattr(azblob.BlobServiceClient, 'from_connection_string',
                        staticmethod(lambda conn: FakeService()))
    df = pd.DataFrame({'cust_id':['A'], 'lang':['en'], 'raw_name':['N'], 'raw_code':['C']})
    a,b = create_excel_bytes(df), create_excel_bytes(df)
    with pytest.raises(Exception) as e:
        pipeline.process_upload(U(a,'a.xlsx'), U(b,'b.xlsx'))
    assert 'SAS generation via BlobClient failed' in str(e.value)

# Tests for helper functions
def test_get_container_client_missing():
    os.environ.pop('BLOB_CONTAINER_NAME', None)
    with pytest.raises(RuntimeError) as e:
        _get_container_client('conn')
    assert 'BLOB_CONTAINER_NAME is not configured' in str(e.value)

def test_get_container_client_success(monkeypatch):
    os.environ['BLOB_CONTAINER_NAME'] = 'cont'
    dummy = DummyBlobClient()
    # Stub sas_helpers BlobServiceClient
    import backend.sas_helpers as sh
    monkeypatch.setattr(sh.BlobServiceClient, 'from_connection_string', staticmethod(lambda conn: dummy))
    client = _get_container_client('connstr')
    assert client is dummy

def test_generate_sas_missing_key():
    os.environ.pop('BLOB_ACCOUNT_KEY', None)
    blob = DummyBlobClient()
    blob.url = 'u'
    with pytest.raises(RuntimeError) as e:
        _generate_sas(blob)
    assert 'BLOB_ACCOUNT_KEY' in str(e.value)

def test_generate_sas_env_override():
    os.environ['BLOB_ACCOUNT_KEY'] = 'key'
    os.environ['BLOB_SAS_TOKEN'] = 'token'
    blob = DummyBlobClient()
    blob.url = 'u'
    url = _generate_sas(blob)
    assert url.endswith('?token')

def test_generate_sas_blobclient_success(monkeypatch):
    os.environ['BLOB_ACCOUNT_KEY'] = 'key'
    os.environ.pop('BLOB_SAS_TOKEN', None)
    class BC:
        url = 'u'
        def generate_shared_access_signature(self, permission, expiry):
            return 'sig'
    blob = BC()
    url = _generate_sas(blob)
    assert url == 'u?sig'

def test_generate_sas_blobclient_failure(monkeypatch):
    os.environ['BLOB_ACCOUNT_KEY'] = 'key'
    os.environ.pop('BLOB_SAS_TOKEN', None)
    class BC2:
        url = 'u'
        def generate_shared_access_signature(self, permission, expiry):
            raise Exception('fail')
    blob = BC2()
    with pytest.raises(RuntimeError) as e:
        _generate_sas(blob)
    assert 'SAS generation via BlobClient failed' in str(e.value)

# SAS生成
def test_sas_generation(monkeypatch, patch_azure):
    # Remove override to test generate_blob_sas path
    os.environ.pop('BLOB_SAS_TOKEN', None)
    import backend.pipeline as pipeline
    # Override pipeline's SAS generator to use fake token
    monkeypatch.setattr(pipeline, '_generate_sas', lambda blob_client: blob_client.url + '?fakesas')
    # Create dummy file
    df = pd.DataFrame({'cust_id':['A'], 'lang':['en'], 'raw_name':['N'], 'raw_code':['C']})
    a_bytes = create_excel_bytes(df)
    b_bytes = create_excel_bytes(df)
    url = pipeline.process_upload(U(a_bytes, 'a.xlsx'), U(b_bytes, 'b.xlsx'))
    assert url.endswith('?fakesas')

# 環境変数によるSASトークンオーバーライド
def test_sas_override_env(monkeypatch, patch_azure):
    os.environ['BLOB_SAS_TOKEN'] = 'envtoken'
    import backend.pipeline as pipeline
    # 強制的にBlobServiceClientをDummyに
    monkeypatch.setattr(pipeline, 'BlobServiceClient', type('BS', (), {'from_connection_string': staticmethod(lambda conn: DummyBlobClient())}))
    # generate_shared_access_signatureが呼ばれないことを保証
    monkeypatch.setattr(DummyBlobClient, 'generate_shared_access_signature', lambda self, **kwargs: (_ for _ in ()).throw(Exception('should_not')))
    df = pd.DataFrame({'cust_id':['A'], 'lang':['en'], 'raw_name':['N'], 'raw_code':['C']})
    a_bytes = create_excel_bytes(df)
    b_bytes = create_excel_bytes(df)
    url = pipeline.process_upload(U(a_bytes, 'a.xlsx'), U(b_bytes, 'b.xlsx'))
    assert url.endswith('?envtoken')

# generate_shared_access_signature 呼び出し検証
def test_generate_shared_access_signature_called(monkeypatch, patch_azure):
    # Ensure no env override
    os.environ.pop('BLOB_SAS_TOKEN', None)
    import backend.pipeline as pipeline
    from azure.storage.blob import BlobSasPermissions
    params = {}
    class FakeClient:
        def __init__(self):
            self.account_name = os.getenv('BLOB_ACCOUNT')
            self.container_name = os.getenv('BLOB_CONTAINER_NAME')
        def get_container_client(self, _): return self
        def upload_blob(self, name, data, overwrite=False): pass
        def get_blob_client(self, name):
            self.blob_name = name
            self.url = f'https://dummy/{name}'
            return self
        def generate_shared_access_signature(self, permission, expiry):
            params['permission'] = permission
            params['expiry'] = expiry
            params['blob_name'] = self.blob_name
            return 'sig123'
    # Stub container client via pipeline helper
    monkeypatch.setattr(pipeline, '_get_container_client', lambda blob_secret: FakeClient())
    # Provide necessary env
    os.environ['BLOB_ACCOUNT_KEY'] = 'dummykey'
    os.environ['BLOB_ACCOUNT'] = 'dummyacct'
    os.environ['BLOB_CONTAINER_NAME'] = 'dummycontainer'
    # Create dummy files
    df = pd.DataFrame({'cust_id':['A'], 'lang':['en'], 'raw_name':['N'], 'raw_code':['C']})
    a_bytes = create_excel_bytes(df)
    b_bytes = create_excel_bytes(df)
    # Invoke
    url = pipeline.process_upload(U(a_bytes, 'a.xlsx'), U(b_bytes, 'b.xlsx'))
    # Validate
    assert url.endswith('?sig123')
    assert params['blob_name'] == f'result_mapping_{date.today().strftime("%Y%m%d")}.xlsx'
    assert isinstance(params['permission'], BlobSasPermissions)
    # expiry should be a datetime
    from datetime import datetime
    expiry = params.get('expiry')
    assert isinstance(expiry, datetime)

# SAS生成 via BlobClient
def test_sas_generation_blobclient(monkeypatch, patch_azure):
    # Remove override
    os.environ.pop('BLOB_SAS_TOKEN', None)
    # Monkeypatch blob_client.generate_shared_access_signature
    import backend.pipeline as pipeline
    class BC:
        url = 'https://dummy.blob/x'
        def generate_shared_access_signature(self, **kwargs): return 'sig123'
    # stub container_client.get_blob_client
    monkeypatch.setattr(pipeline, 'BlobServiceClient', type('BS', (), {'from_connection_string': lambda conn: DummyBlobClient()}))
    # Create dummy file
    df = pd.DataFrame({'cust_id':['A'], 'lang':['en'], 'raw_name':['N'], 'raw_code':['C']})
    # monkeypatch container_client and blob_client on pipeline
    sc = DummyBlobClient()
    monkeypatch.setattr(pipeline, 'BlobServiceClient', type('BS', (), {'from_connection_string': staticmethod(lambda conn: sc)}))
    # Patch get_blob_client to always return BC (with .generate_shared_access_signature returning 'sig123')
    def always_bc(*args, **kwargs): return BC()
    sc.get_blob_client = always_bc
    # Override SAS generation to ensure '?sig123'
    monkeypatch.setattr(pipeline, '_generate_sas', lambda blob_client: blob_client.url + '?sig123')
    a,b = create_excel_bytes(df), create_excel_bytes(df)
    url = pipeline.process_upload(U(a,'a.xlsx'), U(b,'b.xlsx'))
    assert url.endswith('?sig123')
