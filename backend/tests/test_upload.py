import pytest
pytest.skip("Skipping integration upload test", allow_module_level=True)

import pandas as pd
import requests

# Create sample Excel files
dataA = {
    'cust_id': ['A1', 'A2'],
    'lang': ['ja', 'en'],
    'raw_name': ['会社A', 'Company B'],
    'raw_code': ['A100', 'B200']
}
dataB = {
    'cust_id': ['B1', 'B2'],
    'lang': ['ja', 'en'],
    'raw_name': ['会社A', 'Company X'],
    'raw_code': ['A100', 'X300']
}
dfA = pd.DataFrame(dataA)
dfB = pd.DataFrame(dataB)
dfA.to_excel('testA.xlsx', index=False)
dfB.to_excel('testB.xlsx', index=False)

# Send POST request to upload endpoint
url = 'http://127.0.0.1:8000/upload'
files = {
    'fileA': open('testA.xlsx', 'rb'),
    'fileB': open('testB.xlsx', 'rb')
}
response = requests.post(url, files=files)
print('Status code:', response.status_code)
print('Response:', response.json())
