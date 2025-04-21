# 既存の DB/Blob 処理をスキップし、インメモリ部分のみ計測
import pandas as pd
import numpy as np
import faiss
import jellyfish
import time
from dotenv import load_dotenv
import os
from fastapi.datastructures import UploadFile
from backend.pipeline import process_upload

# 1. テストデータ生成
def generate_excel(path, prefix, n):
    df = pd.DataFrame({
        'cust_id': [f"{prefix}{i+1}" for i in range(n)],
        'lang': ['ja' if i%2==0 else 'en' for i in range(n)],
        'raw_name': [f"Company{i+1}" for i in range(n)],
        'raw_code': [f"C{i+1000}" for i in range(n)],
    })
    df.to_excel(path, index=False)

n = 3000
pathA, pathB = 'backend/testA_large.xlsx', 'backend/testB_large.xlsx'
print(f"Generating {n} records for A and B...")
generate_excel(pathA, 'A', n)
generate_excel(pathB, 'B', n)

# 2. インメモリ処理ベンチ
start = time.time()
# 読み込み & 正規化
A = pd.read_excel(pathA)
B = pd.read_excel(pathB)
for df in (A, B):
    df['norm_name'] = df['raw_name'].astype(str).str.strip().str.upper()
    df['norm_code'] = df['raw_code'].astype(str).str.strip().str.upper()
    # ランダムベクトル埋め込み (384次元)
    df['vector'] = [np.random.rand(384).astype(np.float32).tobytes() for _ in range(len(df))]
print(f"[BENCH] Loaded & embedded: {time.time()-start:.2f}s")

# FAISS インデックス構築
vecs_B = np.vstack([np.frombuffer(v, dtype=np.float32) for v in B['vector']])
faiss.normalize_L2(vecs_B)
idx = faiss.IndexHNSWFlat(vecs_B.shape[1], 32, faiss.METRIC_INNER_PRODUCT)
idx.hnsw.efConstruction=200; idx.hnsw.efSearch=64; idx.add(vecs_B)
print(f"[BENCH] FAISS index build: {time.time()-start:.2f}s")

# 検索+スコア計算
vecs_A = np.vstack([np.frombuffer(v, dtype=np.float32) for v in A['vector']])
faiss.normalize_L2(vecs_A)
D, I = idx.search(vecs_A, 5)
# DataFrameでベクトル化スコア計算
rows = []
for ai, aid in enumerate(A['cust_id']):
    for bi, dist in zip(I[ai], D[ai]):
        rows.append((ai, bi, dist))
dr = pd.DataFrame(rows, columns=['i','j','vec_cos'])
dr = dr.merge(A[['cust_id','norm_code','norm_name','lang']], left_on='i', right_index=True)
dr = dr.merge(B[['cust_id','norm_code','norm_name','lang']], left_on='j', right_index=True, suffixes=('_A','_B'))
dr['code_sim'] = np.where(dr.norm_code_A==dr.norm_code_B,1.0,np.where(dr.norm_code_A.str[:3]==dr.norm_code_B.str[:3],0.7,0.0))
dr['name_sim'] = dr.apply(lambda r: jellyfish.jaro_winkler_similarity(r.norm_name_A, r.norm_name_B), axis=1)
dr['lang_pen'] = np.where(dr.lang_A==dr.lang_B,0,-0.05)
dr['final_scr'] = (0.3*dr.vec_cos+0.3*dr.code_sim+0.3*dr.name_sim+dr.lang_pen)*100
print(f"[BENCH] Search & scoring: {time.time()-start:.2f}s")

# Aごと最良ペア抽出
best = dr.sort_values('final_scr',ascending=False).drop_duplicates('cust_id_A')
print(f"[BENCH] Selected best {len(best)} rows: {time.time()-start:.2f}s")

# .env の Azure 設定をロード
load_dotenv(dotenv_path=os.path.join('backend', '.env'), override=True)

# 3. E2E ベンチマーク: DB 書き込み & Blob アップロードを含む
t_start = time.time()
fileA = UploadFile(filename=pathA, file=open(pathA, 'rb'))
fileB = UploadFile(filename=pathB, file=open(pathB, 'rb'))
url = process_upload(fileA, fileB)
print(f"[BENCH] E2E elapsed: {time.time() - t_start:.2f}s, SAS URL: {url}")
