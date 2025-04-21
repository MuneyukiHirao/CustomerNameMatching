import pytest
import io
import pandas as pd
from fastapi.testclient import TestClient
from backend.main import app

pytest.skip("Skipping integration upload client tests", allow_module_level=True)

client = TestClient(app)

def create_excel_bytes(data: dict) -> io.BytesIO:
    df = pd.DataFrame(data)
    buf = io.BytesIO()
    df.to_excel(buf, index=False)
    buf.seek(0)
    return buf

def test_upload_flow():
    dataA = {'cust_id': ['A1'], 'lang': ['ja'], 'raw_name': [''], 'raw_code': ['A100']}
    dataB = {'cust_id': ['B1'], 'lang': ['ja'], 'raw_name': [''], 'raw_code': ['A100']}
    fileA = create_excel_bytes(dataA)
    fileB = create_excel_bytes(dataB)
    files = {'fileA': ('testA.xlsx', fileA, 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'),
             'fileB': ('testB.xlsx', fileB, 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')}
    response = client.post('/upload', files=files)
    assert response.status_code == 200
    json_data = response.json()
    assert 'download_url' in json_data
    assert json_data['download_url'].startswith('http')
