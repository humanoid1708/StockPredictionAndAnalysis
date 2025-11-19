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
NEWS_FILE = "sentiment_analyzed_news.csv" # Your 2.13 GB file
OUTPUT_FOLDER = "processed_data_baseline" # Final data for the "baseline" model

# 3. Paper's parameters
DECAY_RATE = 0.03 # This is the lambda from Equation 1 in the paper

# --- HELPER FUNCTIONS (from the original repo, but adapted) ---

def fill_missing_with_decay(df, col_name, decay_rate, neutral_val):
    """ Fills missing sentiment using the paper's exponential decay formula. """
    df[col_name] = pd.to_numeric(df[col_name], errors='coerce')
    
    last_valid_sentiment = neutral_val
    last_valid_date = pd.NaT
    
    filled_values = []
    
    for i, row in df.iterrows():
        if pd.isna(row[col_name]):
            if pd.notna(last_valid_date):
                days_diff = (row['Date'] - last_valid_date).days
                # This is Equation (1) from the paper: S_t = 3 + (S_0 - 3) * e^(-lambda*t)
                decayed_val = neutral_val + (last_valid_sentiment - neutral_val) * np.exp(-decay_rate * days_diff)
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

def process_baseline_data():
    print(f"Loading all news from {NEWS_FILE}. This might take a moment...")
    try:
        # Load your 2.11 GB file. This requires enough RAM.
        all_news_df = pd.read_csv(NEWS_FILE, low_memory=False)
    except Exception as e:
        print(f"Error reading {NEWS_FILE}: {e}")
        print("Trying with 'latin1' encoding...")
        all_news_df = pd.read_csv(NEWS_FILE, encoding='latin1', low_memory=False)

    all_news_df.columns = all_news_df.columns.str.capitalize()
    
    # Ensure 'Date' is in datetime format
    all_news_df['Date'] = pd.to_datetime(all_news_df['Date']).dt.normalize()
    all_news_df['Stock_symbol'] = all_news_df['Stock_symbol'].str.upper()

    print("News file loaded. Creating new 'baseline_sentiment' score...")
    
    # 1. CREATE THE BASELINE SENTIMENT SCORE
    # We will combine FinBERT scores: (positive - negative)
    # This gives a score from -1.0 to +1.0
    all_news_df['baseline_sentiment'] = all_news_df['Finbert_positive'] - all_news_df['Finbert_negative']

    # 2. SCALE THE SCORE
    # Scale from [-1, 1] to [1, 5] to match the paper's 'Sentiment_gpt' scale
    # This makes our baseline as "apples-to-apples" as possible
    # NewValue = (((OldValue - OldMin) * (NewMax - NewMin)) / (OldMax - OldMin)) + NewMin
    all_news_df['baseline_sentiment_scaled'] = (((all_news_df['baseline_sentiment'] - (-1)) * (5 - 1)) / (1 - (-1))) + 1

    # 3. AVERAGE SENTIMENT BY DAY
    # Group by both stock and date, then calculate the mean sentiment for that day
    daily_sentiment = all_news_df.groupby(['Stock_symbol', 'Date'])['baseline_sentiment_scaled'].mean().reset_index()

    # Make the output directory
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    print(f"Starting to merge data for {len(STOCKS_50)} stocks...")

    for stock in tqdm(STOCKS_50, desc="Processing Stocks"):
        stock_csv = f"{stock}.csv"
        price_file_path = os.path.join(PRICE_FOLDER, stock_csv)
        
        if not os.path.exists(price_file_path):
            # print(f"Warning: Price file not found for {stock}. Skipping.")
            continue
            
        # 1. Load price data
        price_df = pd.read_csv(price_file_path)
        price_df.columns = price_df.columns.str.capitalize()
        price_df['Date'] = pd.to_datetime(price_df['Date']).dt.normalize().dt.tz_localize('UTC')
        
        # 2. Get the news for *this* stock
        stock_news_df = daily_sentiment[daily_sentiment['Stock_symbol'] == stock].copy()
        
        if stock_news_df.empty:
            # print(f"Warning: No news found for {stock}. Skipping.")
            continue
            
        # 3. Create full date range
        min_date = price_df['Date'].min()
        max_date = price_df['Date'].max()
        all_dates = pd.DataFrame(pd.date_range(start=min_date, end=max_date), columns=['Date'])
        
        # 4. Merge prices and news
        merged = all_dates.merge(price_df, on='Date', how='left')
        merged = merged.merge(stock_news_df, on='Date', how='left')
        
        # 5. Fill missing prices (e.g., weekends/holidays)
        merged = merged.ffill() # Forward-fill missing price/volume data
        merged = merged.bfill() # Back-fill any remaining NaNs at the start
        
        # 6. Fill missing sentiment using exponential decay
        # We fill with 3 (neutral) as the paper did
        merged = fill_missing_with_decay(merged, 'baseline_sentiment_scaled', DECAY_RATE, neutral_val=3.0)
        
        # 7. Final "Scaled_sentiment" column (paper's [1, 5] -> [0, 1])
        # This is the final feature for the model
        merged['Scaled_sentiment'] = (merged['baseline_sentiment_scaled'] - 1) / 4
        
        # 8. Select final columns for the model
        final_cols = ['Date', 'Volume', 'Open', 'High', 'Low', 'Close', 'Scaled_sentiment']
        final_df = merged[final_cols].copy()
        
        # 9. Drop any remaining NaNs and save
        final_df = final_df.dropna()
        if not final_df.empty:
            output_path = os.path.join(OUTPUT_FOLDER, stock_csv)
            final_df.to_csv(output_path, index=False)

    print(f"\nBaseline dataset complete! Files are saved in '{OUTPUT_FOLDER}'.")

if __name__ == "__main__":
    process_baseline_data()