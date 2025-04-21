import sys
from pathlib import Path
# Ensure project root is in PYTHONPATH
sys.path.append(str(Path(__file__).resolve().parents[2]))  # add project root to PYTHONPATH
from backend.db import engine

# This script alters customer standard tables to expand name columns to NVARCHAR(200)
conn = engine.raw_connection()
cursor = conn.cursor()
try:
    cursor.execute("ALTER TABLE customers_std_A ALTER COLUMN norm_name NVARCHAR(200);")
    cursor.execute("ALTER TABLE customers_std_A ALTER COLUMN raw_name NVARCHAR(200);")
    cursor.execute("ALTER TABLE customers_std_B ALTER COLUMN norm_name NVARCHAR(200);")
    cursor.execute("ALTER TABLE customers_std_B ALTER COLUMN raw_name NVARCHAR(200);")
    conn.commit()
    print("[INFO] Altered tables: name columns expanded to NVARCHAR(200)")
except Exception as e:
    print(f"[ERROR] Failed to alter tables: {e}")
finally:
    cursor.close()
    conn.close()
