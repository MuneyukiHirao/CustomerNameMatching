import sys
from pathlib import Path
# Ensure project root is in PYTHONPATH
sys.path.append(str(Path(__file__).resolve().parents[2]))
from backend.db import engine

# This script alters customer standard tables to expand cust_id columns to NVARCHAR(100)
conn = engine.raw_connection()
cursor = conn.cursor()
try:
    cursor.execute("ALTER TABLE customers_std_A ALTER COLUMN cust_id NVARCHAR(100);")
    cursor.execute("ALTER TABLE customers_std_B ALTER COLUMN cust_id NVARCHAR(100);")
    conn.commit()
    print("[INFO] Altered tables: cust_id columns expanded to NVARCHAR(100)")
except Exception as e:
    print(f"[ERROR] Failed to alter cust_id columns: {e}")
finally:
    cursor.close()
    conn.close()
