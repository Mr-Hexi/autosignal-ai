import pandas as pd

df = pd.read_csv("data/bronze/transcripts/_all_transcripts.csv")

print(f"Total rows: {len(df)}")
print(f"Columns: {list(df.columns)}")
print(f"\nraw_text null count: {df['raw_text'].isnull().sum()}")
print(f"raw_text empty string: {(df['raw_text'] == '').sum()}")
print(f"raw_text has content: {df['raw_text'].notna().sum()}")
print(f"\nSample raw_text values:")
for i, row in df.head(10).iterrows():
    text = str(row.get('raw_text', ''))
    print(f"  [{row['company']}] len={len(text)} | preview: {text[:80]}")
    
print(f"\nFull first row:")
print(df.iloc[0].to_dict())