# 💰 FinBERT-XGBoost Comparative Stock Prediction

This project investigates and benchmarks the effectiveness of multi-dimensional sentiment features in financial time-series forecasting. It re-evaluates the methodology of a published study by creating a stable, high-performance prediction pipeline using FinBERT and XGBoost, focusing specifically on predicting stock **Closing Price** based on historical data and news sentiment.

## 🚀 Project Goal & Novelty

The primary goal of this project was to test the hypothesis that **a richer, multi-dimensional sentiment vector provides superior predictive power** compared to a single, compressed sentiment score.

### Key Novelty: Feature Engineering

The project compared two advanced sentiment feature representations, both derived from the specialized **FinBERT** classifier:

| Model | Sentiment Feature | Dimensionality | Novelty Level |
| --- | --- | --- | --- |
| **Baseline (1-Feature)** | **Compressed Score** (Positive Score - Negative Score) | 1 | Tests FinBERT as a baseline AI sentiment source (vs. LLM). |
| **Novel (3-Feature)** | **Multi-Dimensional Vector** (Positive, Negative, Neutral scores) | 3 | Tests the core hypothesis: Do separated signals add value? |

### Methodological Improvement

The project switched the predictive model from the unstable **Transformer** used in the original methodology to a highly stable **XGBoost Regressor** and redefined the task to predict **only the Close Price**, thereby eliminating metric-skewing errors caused by the high-magnitude 'Volume' feature.

## 🛠️ Setup and Execution

### 1. Prerequisites

You must have **Python 3.8+** installed. This project requires external data files that are too large for GitHub.

1. **Clone the repository:**
    
    ```
    git clone https://github.com/humanoid1708/StockPredictionAndAnalysis.git
    
    ```
    
2. **Install Libraries:** Install all necessary dependencies, including the PyTorch backend required by FinBERT and the XGBoost model.
    
    ```
    pip install -r requirements.txt
    
    ```
    
3. **Acquire Raw Data:** Download the two raw data files from the original repository's hosting location (links are usually found in the original repo's documentation). Place these files in the root directory:
    - `full_history` (folder containing individual price CSVs)
    - `nasdaq_external_data.csv` (the large news file)

### 2. Data Processing Pipeline (MUST BE RUN ONCE)

The raw data must be processed to create the sentiment features and merge them with price data. These steps handle the large 21.6 GB file size and implement the exponential decay logic.

| Step | Script | Purpose |
| --- | --- | --- |
| **1. Filter News** | `1_filter_news.py` | Filters the 21.6 GB news file for the 50 stocks. |
| **2. Analyze Sentiment** | `2_analyze_sentiment_chunked.py` | Runs FinBERT on the filtered file to generate the 3-feature sentiment vector. (Requires substantial RAM/VRAM). |
| **3. Create Baseline Data** | `3_create_baseline_dataset.py` | Merges price data with the 1-feature compressed sentiment and saves to `new_model/processed_data_baseline/`. |
| **4. Create Novel Data** | `4_create_novel_dataset.py` | Merges price data with the 3-feature sentiment and saves to `new_model/processed_data_novel/`. |

### 3. Running the Experiments

All training scripts use the powerful XGBoost Regressor (`n_estimators=1000`, `learning_rate=0.1`) and are executed from the `new_model/` directory.

### **Execution Commands:**

Navigate to the experiment folder: `cd new_model/`

Run the following commands to generate the final comparison results (results for 5, 25, and 50 stocks):

| Experiment Type | Command | Purpose |
| --- | --- | --- |
| **Baseline (1-Feature)** | `python runbaseline_xgb_summary.py 50` | Generates the performance benchmark. |
| **Novel (3-Feature)** | `python runnovel_xgb_summary.py 50` | Generates the result for the new feature approach. |

## 📊 Final Results & Conclusion

### Summary Metrics (50-Stock Run)

(Values are for the stocks T, C, BHP, GSK, GILD)

| Model | Task | Avg MAE (Eval 5) | Avg R² (Eval 5) | **Avg MSE (Eval 5)** |
| --- | --- | --- | --- | --- |
| **BASELINE (1-Feature)** | Close Only | 0.007023 | 0.973085 | 0.000125 |
| **NOVEL (3-Feature)** | Close Only | 0.007031 | 0.973073 | 0.000125 |

### Output Files

Results are saved to the root directory where the script is executed:

- `summary_results_BASELINE_XGB_CLOSE_50stocks.csv` (Metrics for all 50 stocks).
- `summary_results_NOVEL_XGB_CLOSE_50stocks.csv` (Metrics for all 50 stocks).
- Individual results and plots are saved in `plot_saved...` folders.
