# 顧客マスターマッチング自動化システム (Azure Web App版)

ユーザーがSystem A/Bの顧客マスターExcelをアップロードし、マッチング結果をExcelで返却するWebアプリケーション。

## 構成

- backend: FastAPI + Azure SQL + OpenAI呼び出し
- frontend: React(TypeScript) + MUI

## 環境構築

### Backend

```
cd backend
python -m venv venv
.\\venv\\Scripts\\Activate.ps1
pip install -r requirements.txt
```

.env.templateをコピーして.envを作成し、Azure Key Vault参照設定を行う。

### Frontend

```
cd frontend
npm install
npm start
```

## ディレクトリ構成

```
CustomerNameMatching/
├── .gitignore
├── README.md
├── backend/
│   ├── requirements.txt
│   ├── .env.template
│   └── main.py
└── frontend/
    ├── package.json
    ├── tsconfig.json
    ├── public/index.html
    └── src/index.tsx
```
