import os
import pandas as pd
import faiss
from dotenv import load_dotenv
from pathlib import Path
import numpy as np
import jellyfish
from sqlalchemy.orm import Session
from .db import SessionLocal, engine, Base, get_blob_secret
from .models import CustomerStdA, CustomerStdB, MatchCandidate, MatchScore
import json
import os
from datetime import date, datetime, timedelta
from azure.storage.blob import BlobServiceClient, BlobSasPermissions
try:
    from azure.identity import DefaultAzureCredential
    from azure.keyvault.secrets import SecretClient
except ImportError:
    DefaultAzureCredential = None
    SecretClient = None
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None
import logging
from sqlalchemy.inspection import inspect
import uuid
import time
import pyodbc
from typing import Callable, Optional
from .sas_helpers import _get_container_client, _generate_sas

# Load environment variables from .env (non-overriding)
env_path = Path(__file__).parent / ".env"
if not os.getenv('NO_DOTENV'):
    # Load .env without overriding existing env vars
    load_dotenv(dotenv_path=env_path, override=False)

# Read OpenAI API key now that .env is loaded
api_key = os.getenv("OPENAI_API_KEY")
if not api_key or OpenAI is None:
    logging.warning("OPENAI API disabled: no key or client missing")
    client = None
else:
    try:
        client = OpenAI(api_key=api_key)
    except Exception as e:
        logging.warning(f"OpenAI client init failed: {e}")
        client = None

# Initialize DB tables
def init_db():
    logging.info("init_db: ensuring tables exist (no drop)")
    try:
        Base.metadata.create_all(bind=engine, checkfirst=True)
        logging.info("init_db: tables ensured")
    except Exception as e:
        logging.warning(f"init_db failed: {e}")

# Preprocessing
def normalize_and_embed(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize raw fields and generate OpenAI embeddings."""
    import numpy as _np
    # normalize text fields
    df['norm_name'] = df['raw_name'].astype(str).str.strip().str.upper()
    df['norm_code'] = df['raw_code'].astype(str).str.strip().str.upper()
    # chunked batch embeddings via OpenAI Embedding API
    texts = df['norm_name'].tolist()
    chunk_size = 1000
    embs = []
    start_embed = time.time()
    for i in range(0, len(texts), chunk_size):
        batch = texts[i:i+chunk_size]
        if client:
            try:
                resp = client.embeddings.create(input=batch, model="text-embedding-3-small")
                embs.extend([ _np.array(item.embedding, dtype=_np.float32) for item in resp.data ])
            except Exception as e:
                print(f"[ERROR] Batch embedding chunk error: {e}")
                embs.extend([ _np.zeros(384, dtype=_np.float32) for _ in batch ])
        else:
            embs.extend([ _np.zeros(384, dtype=_np.float32) for _ in batch ])
    print(f"[BENCH] Total embed time (chunked): {time.time() - start_embed:.2f}s")
    df['vector'] = [ emb.tobytes() for emb in embs ]
    return df

# Store into SQL
def store_customers(session: Session, df: pd.DataFrame, model_cls):
    for _, row in df.iterrows():
        session.merge(model_cls(
            cust_id=row['cust_id'],
            lang=row['lang'],
            norm_name=row['norm_name'],
            norm_code=row['norm_code'],
            vector=row['vector'],
            raw_name=row['raw_name'],
            raw_code=row['raw_code']
        ))
    session.commit()

def validate_mappings(model, mappings):
    """モデル定義とマッピングのキーが一致しているか検証"""
    valid_cols = set(c.name for c in inspect(model).columns)
    for i, record in enumerate(mappings):
        bad = set(record.keys()) - valid_cols
        if bad:
            raise ValueError(f"{model.__name__} mapping # {i} に不正なカラムがあります: {bad}")

# Main pipeline
def process_upload(fileA, fileB, progress_callback: Optional[Callable[[str], None]] = None) -> str:
    """Process uploaded Excel files and return SAS URL."""
    # helper for logging and front-end progress
    def log(msg: str):
        print(msg)
        if progress_callback:
            try:
                progress_callback(msg)
            except Exception:
                pass

    """高速版: 一括FAISS検索、pandas vectorized、bulk insert"""
    # 認証＆接続文字列取得
    vault_name = os.getenv("KEYVAULT_NAME")
    try:
        blob_secret = get_blob_secret()
    except Exception as e:
        raise RuntimeError(f"Blob secret retrieval failed: {e}")
    print("[E2E] Retrieved blob secret")
    # --- データ読み込み & 埋め込み ---
    # read and validate input File A
    rawA = pd.read_excel(fileA.file)
    required = ['cust_id','lang','raw_name','raw_code']
    missing = set(required) - set(rawA.columns)
    if missing:
        raise ValueError(f"File A missing columns: {missing}")
    if rawA.empty:
        raise ValueError("File A contains no records")
    if rawA[['raw_name','raw_code']].isnull().any().any():
        # Drop rows with null raw_name or raw_code
        drop_count = rawA[['raw_name','raw_code']].isnull().any(axis=1).sum()
        log(f"[DEBUG] Dropping {drop_count} rows from File A due to null raw_name/raw_code")
        rawA = rawA.dropna(subset=['raw_name','raw_code'])
    start_embed_A = time.time()
    dfA = normalize_and_embed(rawA)
    log(f"[BENCH] OpenAI embed A total time: {time.time() - start_embed_A:.2f}s")
    # read and validate input File B
    rawB = pd.read_excel(fileB.file)
    missing = set(required) - set(rawB.columns)
    if missing:
        raise ValueError(f"File B missing columns: {missing}")
    if rawB.empty:
        raise ValueError("File B contains no records")
    if rawB[['raw_name','raw_code']].isnull().any().any():
        # Drop rows with null raw_name or raw_code
        drop_count = rawB[['raw_name','raw_code']].isnull().any(axis=1).sum()
        log(f"[DEBUG] Dropping {drop_count} rows from File B due to null raw_name/raw_code")
        rawB = rawB.dropna(subset=['raw_name','raw_code'])
    start_embed_B = time.time()
    dfB = normalize_and_embed(rawB)
    log(f"[BENCH] OpenAI embed B total time: {time.time() - start_embed_B:.2f}s")
    log(f"[BENCH] Data loaded: A={dfA.shape}, B={dfB.shape}")
    # DBに格納 via pyodbc fast_executemany (binary vectors handled with pyodbc.Binary)
    conn = engine.raw_connection()
    cursor = conn.cursor()
    if hasattr(cursor, 'fast_executemany'):
        cursor.fast_executemany = True
    # テーブルを初期化 (古いデータをクリア)
    try:
        cursor.execute(f"DELETE FROM {MatchScore.__tablename__}")
        cursor.execute(f"DELETE FROM {MatchCandidate.__tablename__}")
        cursor.execute(f"DELETE FROM {CustomerStdB.__tablename__}")
        cursor.execute(f"DELETE FROM {CustomerStdA.__tablename__}")
        conn.commit()
        log("[BENCH] Cleared existing data in target tables")
        # Trim string fields and IDs to avoid DB truncation errors
        MAX_ID_LEN = 50  # Column cust_id nvarchar(50)
        MAX_NAME_LEN = 100
        MAX_CODE_LEN = 50
        # Truncate IDs
        dfA['cust_id'] = dfA['cust_id'].astype(str).str.slice(0, MAX_ID_LEN)
        dfB['cust_id'] = dfB['cust_id'].astype(str).str.slice(0, MAX_ID_LEN)
        # Remove duplicate keys to avoid PK violations
        dup_count_A = dfA.duplicated(subset=['cust_id'], keep=False).sum()
        if dup_count_A:
            log(f"[DEBUG] Dropping {dup_count_A} duplicate cust_id entries in DataFrame A")
            dfA = dfA.drop_duplicates(subset=['cust_id'], keep='first')
        dup_count_B = dfB.duplicated(subset=['cust_id'], keep=False).sum()
        if dup_count_B:
            log(f"[DEBUG] Dropping {dup_count_B} duplicate cust_id entries in DataFrame B")
            dfB = dfB.drop_duplicates(subset=['cust_id'], keep='first')
        for col in ['norm_name', 'raw_name']:
            dfA[col] = dfA[col].astype(str).str.slice(0, MAX_NAME_LEN)
            dfB[col] = dfB[col].astype(str).str.slice(0, MAX_NAME_LEN)
        for col in ['norm_code', 'raw_code']:
            dfA[col] = dfA[col].astype(str).str.slice(0, MAX_CODE_LEN)
            dfB[col] = dfB[col].astype(str).str.slice(0, MAX_CODE_LEN)
    except Exception as e:
        log(f"[ERROR] Failed to clear tables: {e}")
        conn.rollback()
        raise
    # 各テーブル別にエラーをキャッチして特定できるように
    # CustomerStdA
    sqlA = f"INSERT INTO {CustomerStdA.__tablename__} (cust_id, lang, norm_name, norm_code, vector, raw_name, raw_code) VALUES (?,?,?,?,?,?,?)"
    paramsA = [(
        row['cust_id'], row['lang'], row['norm_name'], row['norm_code'],
        pyodbc.Binary(row['vector']), row['raw_name'], row['raw_code']
    ) for _, row in dfA.iterrows()]
    try:
        tA = time.time()
        cursor.executemany(sqlA, paramsA)
        conn.commit()
        log(f"[BENCH] CustomerStdA insert time: {time.time() - tA:.2f}s")
    except Exception as e:
        log(f"[ERROR] CustomerStdA insert failed: {e}")
        # DEBUG: Log engine URL and column definitions
        try:
            log(f"[DEBUG] Engine URL: {engine.url}")
        except:
            pass
        try:
            cursor.execute(
                "SELECT COLUMN_NAME, CHARACTER_MAXIMUM_LENGTH FROM INFORMATION_SCHEMA.COLUMNS \
                 WHERE TABLE_NAME='customers_std_A' AND COLUMN_NAME IN ('norm_name','raw_name','norm_code','raw_code')"
            )
            for col, maxlen in cursor.fetchall():
                log(f"[DEBUG] Column {col} max length: {maxlen}")
        except Exception as meta_err:
            log(f"[DEBUG] Failed to fetch column metadata: {meta_err}")
        # DEBUG: Max input lengths for cust_id
        try:
            lens_id = [len(str(r['cust_id'])) for _, r in dfA.iterrows()]
            max_id = max(lens_id) if lens_id else 0
            log(f"[DEBUG] Input cust_id max length: {max_id}")
        except:
            pass
        # DEBUG: Input lengths for name and code columns
        try:
            lens_n = [len(str(r['norm_name'])) for _, r in dfA.iterrows()]
            log(f"[DEBUG] Input norm_name max length: {max(lens_n) if lens_n else 0}")
        except:
            pass
        try:
            lens_rn = [len(str(r['raw_name'])) for _, r in dfA.iterrows()]
            log(f"[DEBUG] Input raw_name max length: {max(lens_rn) if lens_rn else 0}")
        except:
            pass
        try:
            lens_code = [len(str(r['norm_code'])) for _, r in dfA.iterrows()]
            log(f"[DEBUG] Input norm_code max length: {max(lens_code) if lens_code else 0}")
        except:
            pass
        try:
            lens_rc = [len(str(r['raw_code'])) for _, r in dfA.iterrows()]
            log(f"[DEBUG] Input raw_code max length: {max(lens_rc) if lens_rc else 0}")
        except:
            pass
        conn.rollback()
        raise
    # CustomerStdB
    sqlB = f"INSERT INTO {CustomerStdB.__tablename__} (cust_id, lang, norm_name, norm_code, vector, raw_name, raw_code) VALUES (?,?,?,?,?,?,?)"
    paramsB = [(
        row['cust_id'], row['lang'], row['norm_name'], row['norm_code'],
        pyodbc.Binary(row['vector']), row['raw_name'], row['raw_code']
    ) for _, row in dfB.iterrows()]
    try:
        tB = time.time()
        cursor.executemany(sqlB, paramsB)
        conn.commit()
        log(f"[BENCH] CustomerStdB insert time: {time.time() - tB:.2f}s")
    except Exception as e:
        log(f"[ERROR] CustomerStdB insert failed: {e}")
        conn.rollback()
        raise
    # --- FAISS一括検索 ---
    vecs_B = np.vstack([np.frombuffer(v, dtype=np.float32) for v in dfB['vector']])
    faiss.normalize_L2(vecs_B)
    idx = faiss.IndexHNSWFlat(vecs_B.shape[1], 32, faiss.METRIC_INNER_PRODUCT)
    idx.hnsw.efConstruction = 200; idx.hnsw.efSearch = 64; idx.add(vecs_B)
    vecs_A = np.vstack([np.frombuffer(v, dtype=np.float32) for v in dfA['vector']])
    faiss.normalize_L2(vecs_A)
    D, I = idx.search(vecs_A, 5)
    # Convert neighbor arrays to native Python types
    D = D.tolist()
    I = I.tolist()
    log(f"[BENCH] FAISS search completed: D shape={np.array(D).shape}, I shape={np.array(I).shape}")
    # 候補DataFrame作成
    cand_map = []
    for ai, a_id in enumerate(dfA['cust_id']):
        for bi, dist in zip(I[ai], D[ai]):
            # skip invalid neighbor index returned by FAISS
            if bi < 0:
                continue
            # use positional indexing to get cust_id
            cust_b = dfB.iloc[bi]['cust_id']
            cand_map.append({'cust_idA': a_id, 'cust_idB': cust_b, 'vec_cos': float(dist)})
    df_cand = pd.DataFrame(cand_map)
    # ルールスコア計算
    df_merge = df_cand.merge(
        dfA[['cust_id','norm_code','norm_name','lang','raw_name']], left_on='cust_idA', right_on='cust_id'
    ).merge(
        dfB[['cust_id','norm_code','norm_name','lang','raw_name']], left_on='cust_idB', right_on='cust_id', suffixes=('_A','_B')
    )
    df_merge['code_sim'] = np.where(
        df_merge.norm_code_A==df_merge.norm_code_B, 1.0,
        np.where(df_merge.norm_code_A.str[:3]==df_merge.norm_code_B.str[:3],0.7,0.0)
    )
    # Calculate name similarity via normalized Levenshtein ratio
    def lev_ratio(a, b):
        dist = jellyfish.levenshtein_distance(a, b)
        max_len = max(len(a), len(b)) or 1
        return 1 - dist / max_len
    df_merge['name_sim'] = df_merge.apply(
        lambda r: lev_ratio(r.norm_name_A, r.norm_name_B), axis=1
    )
    df_merge['lang_pen'] = np.where(df_merge.lang_A==df_merge.lang_B,0,-0.05)
    # Compute final score as weighted embedding + name similarity
    # Embedding similarity gets 70%, name similarity 30%
    raw_score = 0.7 * df_merge.vec_cos + 0.3 * df_merge.name_sim
    df_merge['final_scr'] = raw_score * 100
    df_merge['status'] = np.where(df_merge.final_scr>=85,'auto',
        np.where(df_merge.final_scr>=60,'pending','rejected'))
    log(f"[BENCH] Score calculation completed")
    # MatchCandidate のマッピングを準備（UUID付き）
    df_merge['candidate_id'] = [str(uuid.uuid4()) for _ in range(len(df_merge))]
    cand_mappings = df_merge[['candidate_id','cust_idA','cust_idB','vec_cos','lang_pen']]
    cand_mappings = cand_mappings.rename(columns={
        'candidate_id':'id',
        'cust_idA':'cust_idA','cust_idB':'cust_idB',
        'vec_cos':'vec_cos','lang_pen':'lang_pen'
    }).to_dict('records')
    validate_mappings(MatchCandidate, cand_mappings)
    # Match Scores のマッピングを準備（candidate_idを参照）
    score_df = df_merge[['candidate_id','code_sim','name_sim']].copy()
    score_df['llm_prob'] = 0.0
    score_df['final_scr'] = df_merge.final_scr
    score_df['status'] = df_merge.status
    # action_userとaction_timeはNULL許可とする
    score_df['action_user'] = None
    score_df['action_time'] = None
    score_mappings = score_df.rename(columns={
        'candidate_id':'cand_id',
        'code_sim':'code_sim','name_sim':'name_sim',
        'llm_prob':'llm_prob','final_scr':'final_scr','status':'status',
        'action_user':'action_user','action_time':'action_time'
    }).to_dict('records')
    validate_mappings(MatchScore, score_mappings)
    # MatchCandidate
    sqlC = f"INSERT INTO {MatchCandidate.__tablename__} (id, cust_idA, cust_idB, vec_cos, lang_pen) VALUES (?,?,?,?,?)"
    paramsC = [(rec['id'], rec['cust_idA'], rec['cust_idB'], rec['vec_cos'], rec['lang_pen']) for rec in cand_mappings]
    try:
        tC = time.time()
        cursor.executemany(sqlC, paramsC)
        conn.commit()
        log(f"[BENCH] MatchCandidate insert time: {time.time() - tC:.2f}s")
    except Exception as e:
        log(f"[ERROR] MatchCandidate insert failed: {e}")
        conn.rollback()
        raise
    # MatchScore
    sqlS = f"INSERT INTO {MatchScore.__tablename__} (cand_id, code_sim, name_sim, llm_prob, final_scr, status, action_user, action_time) VALUES (?,?,?,?,?,?,?,?)"
    paramsS = [(rec['cand_id'], rec['code_sim'], rec['name_sim'], rec['llm_prob'], rec['final_scr'], rec['status'], rec['action_user'], rec['action_time']) for rec in score_mappings]
    try:
        tS = time.time()
        cursor.executemany(sqlS, paramsS)
        conn.commit()
        log(f"[BENCH] MatchScore insert time: {time.time() - tS:.2f}s")
    except Exception as e:
        log(f"[ERROR] MatchScore insert failed: {e}")
        conn.rollback()
        raise
    # リソース開放
    cursor.close()
    conn.close()
    # Excel & Blob
    # For each A record, select its best B match (one-to-one)
    df_best = df_merge.sort_values('final_scr', ascending=False).drop_duplicates('cust_idA')
    log(f"[BENCH] Selected best matches count: {len(df_best)} (should equal rows in A)")
    fn = f"result_mapping_{date.today().strftime('%Y%m%d')}_{uuid.uuid4().hex}.xlsx"
    # include original and normalized names for A/B and export
    df_best[['cust_idA','raw_name_A','norm_name_A','cust_idB','raw_name_B','norm_name_B','final_scr','status']].to_excel(fn, index=False)
    log(f"[BENCH] Excel written to {fn}")
    log("[E2E] Excel written")
    # Blob upload and SAS URL generation
    container_client = _get_container_client(blob_secret)
    start_blob_upload = time.time()
    with open(fn, 'rb') as data:
        try:
            container_client.upload_blob(fn, data, overwrite=True)
        except Exception as e:
            raise RuntimeError(f"Blob upload failed: {e}")
    log(f"[BENCH] Blob upload time: {time.time() - start_blob_upload:.2f}s")
    log(f"[BENCH] Blob uploaded: {fn}")
    log("[E2E] Blob uploaded")
    blob_client = container_client.get_blob_client(fn)
    sas_url = _generate_sas(blob_client)
    log(f"[E2E] SAS URL generated: {sas_url}")
    return sas_url

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)s %(message)s")
