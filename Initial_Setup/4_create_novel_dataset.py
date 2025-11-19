import pandas as pd
import numpy as np
import os
from tqdm import tqdm

# --- CONFIGURATION ---

# 1. The list of 50 stocks from the paper
STOCKS_50 = [
    'AAL', 'AAPL', 'ABBV', 'AMGN', 'BABA', 'BHP', 'BIIB', 'BIDU', 'BRK-B', 'C', 
    'CAT', 'CMCSA', 'CMG', 'COP', 'COST', 'CRM', 'CVX', 'DIS', 'EBAY', 'GE', 
    'GILD', 'GLD', 'GSK', 'INTC', 'MRK', 'MSFT', 'MU', 'NKE', 'NVDA', 'ORCL', 
    'PEP', 'PYPL', 'QCOM', 'QQQ', 'SBUX', 'T', 'TGT', 'TM', 'TSLA', 'USO', 'V', 
    'WFC', 'XLF', 'KO', 'AMD', 'TSM', 'GOOG', 'WMT', 'AMZN', 'DAL'
]

# 2. File paths
PRICE_FOLDER = "full_history"
NEWS_FILE = "sentiment_analyzed_news.csv" # Your 2.11 GB file
OUTPUT_FOLDER = "processed_data_novel" # Final data for your "novel" model

# 3. Paper's parameters
DECAY_RATE = 0.03 # This is the lambda from Equation 1 in the paper

# --- HELPER FUNCTIONS ---

def fill_missing_with_decay_novel(df, col_name, decay_rate, neutral_val):
    """ Fills missing sentiment using simple exponential decay towards neutral. """
    df[col_name] = pd.to_numeric(df[col_name], errors='coerce')
    
    last_valid_sentiment = neutral_val
    last_valid_date = pd.NaT
    
    filled_values = []
    
    for i, row in df.iterrows():
        if pd.isna(row[col_name]):
            if pd.notna(last_valid_date):
                days_diff = (row['Date'] - last_valid_date).days
                # Simple decay: S_t = S_0 * e^(-lambda*t)
                # We decay towards 0, not 3, since these are 0-1 scores
                decayed_val = last_valid_sentiment * np.exp(-decay_rate * days_diff)
                filled_values.append(decayed_val)
            else:
                filled_values.append(neutral_val)
        else:
            last_valid_sentiment = row[col_name]
            last_valid_date = row['Date']
            filled_values.append(row[col_name])
            
    df[col_name] = filled_values
    return df

# --- MAIN SCRIPT ---

def process_novel_data():
    print(f"Loading all news from {NEWS_FILE}. This might take a moment...")
    try:
        all_news_df = pd.read_csv(NEWS_FILE, low_memory=False)
    except Exception as e:
        print(f"Error reading {NEWS_FILE}: {e}")
        all_news_df = pd.read_csv(NEWS_FILE, encoding='latin1', low_memory=False)

    all_news_df.columns = all_news_df.columns.str.capitalize()
    
    all_news_df['Date'] = pd.to_datetime(all_news_df['Date']).dt.normalize()
    all_news_df['Stock_symbol'] = all_news_df['Stock_symbol'].str.upper()

    print("News file loaded. Averaging daily FinBERT scores...")
    
    # 1. AVERAGE SENTIMENT BY DAY for all 3 columns
    sentiment_cols = ['Finbert_positive', 'Finbert_negative', 'Finbert_neutral']
    group_cols = ['Stock_symbol', 'Date']
    daily_sentiment = all_news_df.groupby(group_cols)[sentiment_cols].mean().reset_index()

    # Make the output directory
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    print(f"Starting to merge data for {len(STOCKS_50)} stocks...")

    for stock in tqdm(STOCKS_50, desc="Processing Stocks"):
        stock_csv = f"{stock}.csv"
        price_file_path = os.path.join(PRICE_FOLDER, stock_csv)
        
        if not os.path.exists(price_file_path):
            continue
            
        # 1. Load price data
        price_df = pd.read_csv(price_file_path)
        price_df.columns = price_df.columns.str.capitalize()
        price_df['Date'] = pd.to_datetime(price_df['Date']).dt.normalize().dt.tz_localize('UTC')
        
        # 2. Get the news for *this* stock
        stock_news_df = daily_sentiment[daily_sentiment['Stock_symbol'] == stock].copy()
        
        if stock_news_df.empty:
            continue
            
        # 3. Create full date range
        min_date = price_df['Date'].min()
        max_date = price_df['Date'].max()
        all_dates = pd.DataFrame(pd.date_range(start=min_date, end=max_date), columns=['Date'])
        
        # 4. Merge prices and news
        merged = all_dates.merge(price_df, on='Date', how='left')
        merged = merged.merge(stock_news_df, on='Date', how='left')
        
        # 5. Fill missing prices (e.g., weekends/holidays)
        merged = merged.ffill() 
        merged = merged.bfill() 
        
        # 6. Fill missing sentiment using exponential decay for all 3 columns
        # We fill with 0, as that is the neutral value for these scores
        merged = fill_missing_with_decay_novel(merged, 'Finbert_positive', DECAY_RATE, neutral_val=0.0)
        merged = fill_missing_with_decay_novel(merged, 'Finbert_negative', DECAY_RATE, neutral_val=0.0)
        merged = fill_missing_with_decay_novel(merged, 'Finbert_neutral', DECAY_RATE, neutral_val=0.0)
        
        # 7. Select final columns for the model
        # Note: These are already scaled 0-1, so no extra scaling needed.
        final_cols = ['Date', 'Volume', 'Open', 'High', 'Low', 'Close', 
                      'Finbert_positive', 'Finbert_negative', 'Finbert_neutral']
        final_df = merged[final_cols].copy()
        
        # 8. Drop any remaining NaNs and save
        final_df = final_df.dropna()
        if not final_df.empty:
            output_path = os.path.join(OUTPUT_FOLDER, stock_csv)
            final_df.to_csv(output_path, index=False)

    print(f"\nNovel dataset complete! Files are saved in '{OUTPUT_FOLDER}'.")

if __name__ == "__main__":
    process_novel_data()