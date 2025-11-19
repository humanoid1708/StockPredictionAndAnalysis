import pandas as pd
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch

# --- 1. Load Model ---

print("Loading FinBERT sentiment model...")
# Check if a GPU is available, otherwise use CPU
device = 0 if torch.cuda.is_available() else -1
print(f"Using device: {'GPU' if device == 0 else 'CPU'}")

# We will use FinBERT, a model specifically trained on financial text
model_name = "ProsusAI/finbert"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
sentiment_pipeline = pipeline(
    "sentiment-analysis", 
    model=model, 
    tokenizer=tokenizer, 
    device=device
)
print("Model loaded successfully.")

# --- 2. Define File Paths and Chunking ---

input_file = 'filtered_50_news.csv'
output_file = 'sentiment_analyzed_news.csv'
chunk_size = 50_000  # Process 50,000 rows at a time. Lower this if you run out of memory.
is_first_chunk = True

# Create a file iterator
reader = pd.read_csv(
    input_file, 
    chunksize=chunk_size, 
    on_bad_lines='skip'
)

print(f"Starting to process {input_file} in {chunk_size}-line chunks...")

# --- 3. Process File Chunk by Chunk ---

for i, chunk in enumerate(reader):
    print(f"\n--- Processing Chunk {i+1} ---")
    
    # Make sure 'Headline' column is clean and in string format
    texts = chunk['Article_title'].astype(str).tolist()

    print(f"Analyzing sentiment for {len(texts)} headlines...")
    # Run the pipeline on all texts in this chunk.
    # This will return a list of dictionaries, e.g., [{'label': 'positive', 'score': 0.9}]
    try:
        results = sentiment_pipeline(texts, batch_size=32, truncation=True)
    except Exception as e:
        print(f"Error during sentiment analysis on this chunk: {e}. Skipping chunk.")
        continue

    print("Analysis for chunk complete. Processing results...")

    # This FinBERT model outputs 'positive', 'negative', or 'neutral'
    # We will create your "multi-dimensional" sentiment vector
    pos_scores = []
    neg_scores = []
    neu_scores = []

    for res in results:
        label = res['label']
        score = res['score']
        
        if label == 'positive':
            pos_scores.append(score)
            neg_scores.append(0.0)
            neu_scores.append(0.0)
        elif label == 'negative':
            pos_scores.append(0.0)
            neg_scores.append(score)
            neu_scores.append(0.0)
        elif label == 'neutral':
            pos_scores.append(0.0)
            neg_scores.append(0.0)
            neu_scores.append(score)

    # Add your new sentiment columns to the chunk DataFrame
    chunk['finbert_positive'] = pos_scores
    chunk['finbert_negative'] = neg_scores
    chunk['finbert_neutral'] = neu_scores

    # Save the processed chunk
    if is_first_chunk:
        # For the first chunk, create the file and write the header
        chunk.to_csv(output_file, mode='w', index=False, header=True)
        is_first_chunk = False
        print(f"Saved first chunk to {output_file}")
    else:
        # For all other chunks, append to the file without the header
        chunk.to_csv(output_file, mode='a', index=False, header=False)
        print(f"Appended chunk {i+1} to {output_file}")

print("\n--------------------------------------------------")
print("All chunks processed.")
print(f"Success! Your file with novel sentiment scores is saved as: {output_file}")
print("You can now modify 'price_news_integrate.py' to use THIS file.")