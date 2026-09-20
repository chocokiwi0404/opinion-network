import pandas as pd
import numpy as np

# ---------- 1. Load ----------
df = pd.read_csv('Survey_Results_UC.csv', encoding='utf-8-sig')

# ---------- 2. Fix column names ----------
df.columns = (
    df.columns
      .str.replace('ï»¿', '', regex=False)
      .str.replace('"', '', regex=False)
      .str.strip()
)
df = df.rename(columns={df.columns[0]: 'respondent_id'})

# ---------- 3. Identify Likert item columns ----------
item_cols = [c for c in df.columns if c != 'respondent_id']

# ---------- 4. Likert mapping ----------
likert = {
    'Strongly Disagree': 1,
    'Disagree': 2,
    'Neutral': 3,
    'Agree': 4,
    'Strongly Agree': 5
}

# Apply mapping; anything not in the map (No Comments, blanks) -> NaN
for c in item_cols:
    df[c] = (
        df[c]
          .astype(str)
          .str.strip()
          .map(likert)
    )

# ---------- 5. Drop fully empty rows ----------
df = df.dropna(how='all', subset=item_cols).reset_index(drop=True)

# ---------- 6. Drop rows with <50% completion ----------
completion = df[item_cols].notna().mean(axis=1)
df['completion_rate'] = completion.round(3)
df = df[completion >= 0.50].reset_index(drop=True)

# ---------- 7. Save ----------
df.to_csv('Survey_Results_UC_cleaned.csv', index=False)

# ---------- 8. Summary report ----------
print("Shape after cleaning:", df.shape)
print("\nCompletion rate distribution:")
print(df['completion_rate'].describe().round(3))
print("\nMissing values per item (top 10):")
print(df[item_cols].isna().sum().sort_values(ascending=False).head(10))
print("\nMean agreement by domain:")
domain_means = df[item_cols].mean().groupby(lambda x: x[0]).mean().round(3)
print(domain_means)