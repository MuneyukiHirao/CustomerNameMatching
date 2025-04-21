import io
import os
import pandas as pd
import pytest
from fastapi.testclient import TestClient
import backend.main as main_module

@pytest.fixture(scope='function')
def client(monkeypatch):
    # Enable synchronous test mode
    os.environ['TESTING'] = '1'
    # set environment before app import/startup
    os.environ.update({
        'KEYVAULT_NAME': 'dummy',
        'BLOB_CONTAINER_NAME': 'dummy',
        'BLOB_ACCOUNT_KEY': 'dummykey',
        'BLOB_ACCOUNT': 'dummyacct',
        'BLOB_CONNECTION_STRING': 'UseDevelopmentStorage=true'
    })
    # stub pipeline.process_upload and init_db
    monkeypatch.setattr(main_module, 'process_upload', lambda a, b, progress_callback=None: 'https://dummy.blob/container/f.xlsx?sig=testsig')
    monkeypatch.setattr(main_module, 'init_db', lambda: None)
    from fastapi.testclient import TestClient
    client = TestClient(main_module.app)
    # Patch jobs dict for async download test
    import backend.main
    job_id = 'testjobid'
    job_data = {
        'status': 'completed',
        'messages': [],
        'download_url': 'https://dummy.blob/container/f.xlsx?sig=testsig',
        'error': None
    }
    backend.main.jobs.clear()
    backend.main.jobs[job_id] = job_data
    main_module.jobs[job_id] = job_data
    if hasattr(main_module.app, 'state'):
        main_module.app.state.jobs = getattr(main_module.app.state, 'jobs', {})
        main_module.app.state.jobs[job_id] = job_data
    client._test_job_id = job_id
    return client

# Utility to create excel bytes
def create_excel_bytes(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    df.to_excel(buf, index=False)
    buf.seek(0)
    return buf.getvalue()

def test_upload_and_download_flow(client):
    # Prepare minimal valid DataFrames A and B
    df = pd.DataFrame({
        'cust_id': ['X'],
        'lang': ['en'],
        'raw_name': ['N'],
        'raw_code': ['C']
    })
    bytesA = create_excel_bytes(df)
    bytesB = create_excel_bytes(df)
    files = {
        'fileA': ('A.xlsx', bytesA, 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'),
        'fileB': ('B.xlsx', bytesB, 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
    }
    # Call upload endpoint in debug mode for synchronous return
    resp = client.post('/upload?debug=true', files=files)
    assert resp.status_code == 200
    data = resp.json()
    # Should return completed status and download_url
    assert data['status'] == 'completed'
    assert data['download_url'] == 'https://dummy.blob/container/f.xlsx?sig=testsig'

# Failure cases for download
def test_download_errors(monkeypatch, client):
    # Non-existent job
    resp = client.get('/download?job_id=not_exist')
    assert resp.status_code == 404
    # Job not completed
    # Setup a job
    j = 'j1'
    main_module.jobs[j] = {'status': 'pending', 'messages': [], 'download_url': None}
    resp2 = client.get(f'/download?job_id={j}')
    assert resp2.status_code == 400
    # Job completed but no URL
    j2 = 'j2'
    main_module.jobs[j2] = {'status': 'completed', 'messages': [], 'download_url': None}
    resp3 = client.get(f'/download?job_id={j2}')
    assert resp3.status_code == 404
