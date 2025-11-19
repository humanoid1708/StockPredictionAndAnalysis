import pandas as pd
import os

print("Starting to filter the giant news file. This may take a while...")

# These are the 50 stocks from the 'names_50' list in the paper's script
# We make them uppercase for easy matching.
STOCKS_TO_KEEP = [
    'AAL', 'AAPL', 'ABBV', 'AMGN', 'BABA', 'BHP', 'BIIB', 'BIDU', 'BRK-B', 'C', 
    'CAT', 'CMCSA', 'CMG', 'COP', 'COST', 'CRM', 'CVX', 'DIS', 'EBAY', 'GE', 
    'GILD', 'GLD', 'GSK', 'INTC', 'MRK', 'MSFT', 'MU', 'NKE', 'NVDA', 'ORCL', 
    'PEP', 'PYPL', 'QCOM', 'QQQ', 'SBUX', 'T', 'TGT', 'TM', 'TSLA', 'USO', 'V', 
    'WFC', 'XLF', 'KO', 'AMD', 'TSM', 'GOOG', 'WMT', 'AMZN', 'DAL' # Added AMZN/DAL just in case
]

# --- CHANGE 1: Set robust file paths ---
# Get the absolute path of the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Define file paths by joining the script's directory with the filenames
raw_news_file = os.path.join(script_dir, 'nasdaq_external_data.csv') 
filtered_news_file = os.path.join(script_dir, 'filtered_50_stock_news.csv')
# --- END CHANGE 1 ---


# Define the chunk size (how many rows to read into memory at once)
chunk_size = 1_000_000 
is_first_chunk = True

# Create a file iterator
try:
    # --- CHANGE 2: Added low_memory=False ---
    reader = pd.read_csv(raw_news_file, chunksize=chunk_size, on_bad_lines='skip', low_memory=False)
except Exception as e:
    print(f"Error reading {raw_news_file}: {e}")
    print("Trying again with 'latin1' encoding...")
    # --- CHANGE 2: Added low_memory=False ---
    reader = pd.read_csv(raw_news_file, chunksize=chunk_size, on_bad_lines='skip', encoding='latin1', low_memory=False)

print(f"Reading {raw_news_file} in {chunk_size}-line chunks...")

# Loop through the file chunk by chunk
for chunk in reader:
    # Make column names consistent
    chunk.columns = chunk.columns.str.capitalize()
    
    if 'Stock_symbol' not in chunk.columns:
        print("Skipping a chunk because 'Stock_symbol' column was not found.")
        continue

    # --- CHANGE 3: Force column to string *before* using .str accessor ---
    # This prevents the "AttributeError: Can only use .str accessor"
    chunk['Stock_symbol'] = chunk['Stock_symbol'].astype(str)
    # --- END CHANGE 3 ---

    # Convert the 'Stock_symbol' column to uppercase for reliable matching
    chunk_upper = chunk['Stock_symbol'].str.upper()
    
    # Filter the chunk to keep only rows where the stock is in our list
    filtered_chunk = chunk[chunk_upper.isin(STOCKS_TO_KEEP)]
    
    if not filtered_chunk.empty:
        print(f"   > Found {len(filtered_chunk)} matching rows in this chunk.")
        
        # If it's the first time, write the header
        if is_first_chunk:
            filtered_chunk.to_csv(filtered_news_file, mode='w', index=False, header=True)
            is_first_chunk = False
        # For all other chunks, append without the header
        else:
            filtered_chunk.to_csv(filtered_news_file, mode='a', index=False, header=False)

print("--------------------------------------------------")
print(f"Filtering complete! Your new, smaller file is saved as: {filtered_news_file}")
print("You can now use this file for your sentiment analysis.")