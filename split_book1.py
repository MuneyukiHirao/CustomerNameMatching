import pandas as pd

# 1. Read Book1.xlsx
df = pd.read_excel("backend/Book1.xlsx")

# 2. Split first column by '-' into two columns '前半' and '後半'
df[["前半", "後半"]] = df.iloc[:, 0].astype(str).str.split("-", n=1, expand=True)

# 3. Save to new Excel file
output = "backend/Book1_split.xlsx"
df.to_excel(output, index=False)
print(f"Output written to {output}")
