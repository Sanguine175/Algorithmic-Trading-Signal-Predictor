import pandas as pd
import os

input_path = '/Users/ishanarora18/Downloads/financial_news.csv/raw_analyst_ratings.csv'
output_path = '/Users/ishanarora18/.gemini/antigravity/scratch/financial_sentiment_trading/financial_news.csv'

print(f"Reading from {input_path}...")
chunks = pd.read_csv(input_path, usecols=['date', 'headline', 'stock'], chunksize=100000)

aapl_dfs = []
for chunk in chunks:
    aapl_chunk = chunk[chunk['stock'] == 'AAPL']
    if not aapl_chunk.empty:
        aapl_dfs.append(aapl_chunk)

if aapl_dfs:
    df_aapl = pd.concat(aapl_dfs)
    print(f"Found {len(df_aapl)} AAPL headlines.")
    
    # Drop rows with missing values in key columns
    df_aapl = df_aapl.dropna(subset=['date', 'headline'])
    
    # Parse dates using mixed format and convert to UTC
    df_aapl['Date'] = pd.to_datetime(df_aapl['date'], format='mixed', utc=True, errors='coerce')
    df_aapl = df_aapl.dropna(subset=['Date'])
    
    # Convert to date-only string format (YYYY-MM-DD)
    df_aapl['Date'] = df_aapl['Date'].dt.strftime('%Y-%m-%d')
    
    # Rename and select columns
    df_aapl = df_aapl.rename(columns={'headline': 'Headline'})
    df_final = df_aapl[['Date', 'Headline']].sort_values('Date')
    
    # Save to CSV
    df_final.to_csv(output_path, index=False)
    print(f"Saved {len(df_final)} headlines to {output_path}")
else:
    print("No AAPL headlines found!")
