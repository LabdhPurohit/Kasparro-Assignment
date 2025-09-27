# clean_google_play.py
import pandas as pd
import numpy as np
import re
from datetime import datetime

INPUT = "googleplaystore.csv"   
OUTPUT = "clean_google_play_data.csv"


# In this part we are removing + and commas and quotes/spaces
def parse_installs(s):
    if pd.isna(s): return np.nan
    s = str(s).strip().replace('+','').replace(',','').replace('"','')
    try:
        return int(s)
    except:
        return np.nan

# In this We are cleaning the reviews column by removing commas 
# and safely converting it into an integer, else returning NaN if invalid
def parse_reviews(s):
    if pd.isna(s): return np.nan
    s = str(s).strip().replace(',','')
    try:
        return int(float(s))
    except:
        return np.nan

def parse_price(s):
    if pd.isna(s): return 0.0
    s = str(s).strip()
    # Remove $ and whitespace
    s = s.replace('$','').replace('Free','0').replace('"','')
    try:
        return float(s)
    except:
        # if something weird like '0.00' or text
        try:
            return float(re.findall(r"[-+]?\d*\.\d+|\d+", s)[0])
        except:
            return 0.0

def size_to_kb(x):
    """Convert Size field to KB (float). Return np.nan if unknown."""
    if pd.isna(x): 
        return np.nan
    s = str(x).strip()
    if s == "Varies with device" or s == "Varies with device.": 
        return np.nan
    # sometimes size could be like '1.0M' or '2,000k' or '100k'
    s = s.replace(',','').replace('"','')
    m = re.match(r"^([\d\.]+)\s*([kKmM])$", s)
    if m:
        val, unit = m.groups()
        try:
            val = float(val)
        except:
            return np.nan
        if unit.lower() == 'm':
            return val * 1024.0
        else:
            return val
    # sometimes value 'Varies with device' or '1,000'
    # try to extract any number (assume bytes if large?) — fallback: parse number
    m2 = re.findall(r"[\d\.]+", s)
    if m2:
        try:
            return float(m2[0])
        except:
            return np.nan
    return np.nan

def parse_last_updated(s):
    if pd.isna(s): return pd.NaT
    s = str(s).strip().replace('"','')
    # pandas can parse many formats
    try:
        return pd.to_datetime(s, errors='coerce')
    except:
        return pd.NaT

def primary_genre(genres):
    if pd.isna(genres): return np.nan
    # Genres may be 'Art & Design;Pretend Play' — keep first
    return str(genres).split(';')[0].strip()

def clean_dataframe(df):
    # drop completely empty columns/rows
    df = df.copy()
    # Standardize column names
    df.columns = [c.strip() for c in df.columns]

    # Strip whitespace in string columns (small cost)
    str_cols = df.select_dtypes(include=['object']).columns
    for c in str_cols:
        df[c] = df[c].apply(lambda x: x.strip() if isinstance(x, str) else x)

    # Remove obvious duplicates by App + Category + Current Ver
    df = df.drop_duplicates(subset=['App','Category','Current Ver'], keep='first')

    # Numeric conversions
    df['Installs_clean'] = df['Installs'].apply(parse_installs)
    df['Reviews_clean'] = df['Reviews'].apply(parse_reviews)
    df['Rating'] = pd.to_numeric(df['Rating'], errors='coerce')   # keep NaN if invalid
    df['Price_USD'] = df['Price'].apply(parse_price)

    # Size -> KB
    df['Size_KB'] = df['Size'].apply(size_to_kb)

    # Genres primary
    df['Primary_Genre'] = df['Genres'].apply(primary_genre)

    # Last Updated -> datetime
    df['Last_Updated_dt'] = df['Last Updated'].apply(parse_last_updated)

    # Extract Android min version if useful (take first number)
    def parse_android_ver(v):
        if pd.isna(v): return np.nan
        v = str(v)
        m = re.findall(r"(\d+\.\d+|\d+)", v)
        if m:
            try:
                return float(m[0])
            except:
                return np.nan
        return np.nan
    df['Android_Version_min'] = df['Android Ver'].apply(parse_android_ver)

    # Reviews and Installs: if Reviews or Installs are NaN, try to coerce from other fields
    df['Reviews_clean'] = df['Reviews_clean'].fillna(0).astype(int)
    # Installs could be NaN for some rows, keep as float
    # Keep Rating NaN as-is so later imputation can be decided

    # Drop rows missing critical fields: App OR Category
    df = df.dropna(subset=['App','Category'])

    # Optionally drop rows with no installs & no reviews & no rating (garbage)
    df = df[~((df['Installs_clean'].isna()) & (df['Reviews_clean']==0) & (df['Rating'].isna()))]

    # Reorder and keep essential cleaned columns
    out_cols = [
        'App','Category','Primary_Genre','Rating','Reviews_clean','Installs_clean',
        'Size_KB','Type','Price_USD','Content Rating','Last_Updated_dt','Current Ver','Android_Version_min'
    ]
    # Ensure all exist
    out_cols = [c for c in out_cols if c in df.columns]
    clean = df[out_cols].rename(columns={
        'Reviews_clean': 'Reviews',
        'Installs_clean': 'Installs',
        'Price_USD': 'Price',
        'Last_Updated_dt': 'Last Updated',
        'Primary_Genre': 'Genre',
        'Android_Version_min': 'Android_Min_Version'
    })

    # Final small fixes: Ratings out of bounds -> set NaN
    clean.loc[(clean['Rating'] < 0) | (clean['Rating'] > 5), 'Rating'] = np.nan

    # Reset index
    clean = clean.reset_index(drop=True)
    return clean

def main():
    print("Loading dataset:", INPUT)
    df = pd.read_csv(INPUT)
    print("Original shape:", df.shape)

    clean = clean_dataframe(df)
    print("Clean shape:", clean.shape)

    # Save CSV
    clean.to_csv(OUTPUT, index=False)
    print("Saved cleaned data to:", OUTPUT)
    # Quick summary
    print("\nSample rows:")
    print(clean.head(5).to_string(index=False))
    print("\nColumns in cleaned file:", list(clean.columns))

if __name__ == "__main__":
    main()
