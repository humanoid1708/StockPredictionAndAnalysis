import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# --- CHANGE 1: Corrected import based on 'tst' folder structure ---
from tst.transformer import Transformer
from tqdm import tqdm
import glob
import pdb
import sys # Added for command-line arguments

# --- Global device setting ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- MODIFIED: Target Feature Index ---
# Find the index of 'Close' in the selected columns for the NOVEL setup
NOVEL_FEATURES = ['Volume', 'Open', 'High', 'Low', 'Close', 'Finbert_positive', 'Finbert_negative', 'Finbert_neutral']
TARGET_FEATURE_INDEX = NOVEL_FEATURES.index('Close') # Should be 4

# --- Function Definitions (Identical to baseline_close_only.py) ---
# NOTE: In a real project, put these in a separate file and import.

# --- MODIFIED: create_sequences to only target 'Close' price ---
def create_sequences(data, input_length, output_length, target_feature_index):
    X, y = [], []
    # Data shape is [num_samples, num_features]
    for i in range(0, (len(data) - input_length - output_length + 1), 1):
        X.append(data[i:(i + input_length), :]) # Input keeps all features
        # Target y only takes the 'Close' price column for the next 'output_length' steps
        y.append(data[(i + input_length):(i + input_length + output_length), target_feature_index])
    # Output y shape will be [N, output_length]
    return np.array(X), np.array(y)

def read_csv_case_insensitive(file_path):
    # (Same robust function as before)
    try: return pd.read_csv(file_path)
    except FileNotFoundError:
        directory, filename = os.path.split(file_path)
        if not directory: directory = "."
        pattern = os.path.join(directory, ''.join(['[{}{}]'.format(c.lower(), c.upper()) if c.isalpha() else c for c in filename]))
        matching_files = glob.glob(pattern)
        if matching_files:
            print(f"Reading case-insensitive match: {matching_files[0]}")
            try: return pd.read_csv(matching_files[0])
            except pd.errors.EmptyDataError: print(f"Warning: File {matching_files[0]} is empty. Skipping."); return None
            except Exception as e_inner: print(f"An error occurred reading {matching_files[0]}: {e_inner}"); return None
        else: print(f"No file matches the pattern: {file_path}"); return None
    except pd.errors.EmptyDataError: print(f"Warning: File {file_path} is empty. Skipping."); return None
    except Exception as e:
        print(f"An error occurred reading {file_path}: {e}")
        try:
            print("Attempting read with encoding='latin1'")
            return pd.read_csv(file_path, encoding='latin1')
        except Exception as e_enc: print(f"Error reading {file_path} even with latin1 encoding: {e_enc}"); return None

# --- MODIFIED: data_processor uses new create_sequences ---
def data_processor(data, target_feature_index):
    input_length = 50
    output_length = 3
    split_ratio = 0.85
    split_idx = int(split_ratio * len(data))
    raw_train = data[:split_idx]
    raw_test  = data[split_idx:]
    if raw_train.shape[0] == 0 or raw_test.shape[0] == 0:
        print(f"Warning: Not enough data for train/test split."); return None, None, None
    print(f'Train data shape: {raw_train.shape}')
    print(f'Test data shape: {raw_test.shape}')
    # --- IMPORTANT: Scaler now needs to handle ONLY the target variable separately ---
    scaler_X = MinMaxScaler().fit(raw_train)
    # Create a separate scaler JUST for the target column ('Close')
    scaler_y = MinMaxScaler().fit(raw_train[:, target_feature_index].reshape(-1, 1))
    # Build sequences using the raw data and target index
    X_train, y_train_unscaled = create_sequences(raw_train, input_length, output_length, target_feature_index)
    X_test,  y_test_unscaled  = create_sequences(raw_test,  input_length, output_length, target_feature_index)
    # Scale the inputs X using the multi-feature scaler
    X_train_scaled = np.array([scaler_X.transform(seq) for seq in X_train])
    X_test_scaled = np.array([scaler_X.transform(seq) for seq in X_test])
    # Scale the targets y using the single-feature scaler
    y_train_scaled = np.array([scaler_y.transform(seq.reshape(-1, 1)).flatten() for seq in y_train_unscaled])
    y_test_scaled = np.array([scaler_y.transform(seq.reshape(-1, 1)).flatten() for seq in y_test_unscaled])
    if X_train_scaled.shape[0] == 0 or X_test_scaled.shape[0] == 0:
        print(f"Warning: Not enough data after sequencing."); return None, None, None
    print(f'X_train: {X_train_scaled.shape}, X_test: {X_test_scaled.shape}, y_train: {y_train_scaled.shape}, y_test: {y_test_scaled.shape}')
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32).to(device)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test_scaled, dtype=torch.float32).to(device) # Target tensor
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    batch_size = 64
    dataloader_train = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    dataloader_test = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    return dataloader_train, dataloader_test, scaler_X, scaler_y # Return both scalers

# --- MODIFIED: train_model now sets d_output=output_length ---
def train_model(dataloader_train, symbol, num_csvs, d_input, run_type):
    # d_input is now 8 for novel
    chunk_mode = None
    output_length = 3
    # --- CHANGE: Model output dimension is now just the prediction length ---
    d_output = output_length # Predict 3 values (Close price for day 1, 2, 3)
    d_model = 32; q = 8; v = 8; h = 8; N = 4
    attention_size = None; dropout = 0.1; pe = 'regular'
    model = Transformer(d_input=d_input, d_model=d_model, d_output=d_output, q=q, v=v, h=h, N=N, attention_size=attention_size, dropout=dropout, chunk_mode=chunk_mode, pe=pe).to(device)
    model_dir = "model_saved_close_only" # Separate save dir
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f'{run_type}_{num_csvs}stocks_{N}layers_close.pt')
    epochs = 50
    loss_function = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.00001)
    print(f"Starting CLOSE PRICE training for {symbol} ({run_type}, {num_csvs} stocks)...")
    model.train()
    hist_loss = np.zeros(epochs)
    if not dataloader_train or len(dataloader_train.dataset) == 0:
        print("Error: Training dataloader is empty. Skipping training."); return None
    for idx_epoch in range(epochs):
        running_loss = 0
        with tqdm(total=len(dataloader_train.dataset), desc=f"[Epoch {idx_epoch+1:3d}/{epochs}] {symbol}") as pbar:
            for idx_batch, (x, y) in enumerate(dataloader_train):
                if x.size(0) != dataloader_train.batch_size or y.size(0) != dataloader_train.batch_size: continue
                optimizer.zero_grad()
                y_pred = model(x) # Output: [Batch, output_length]
                loss = loss_function(y_pred, y)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * x.size(0)
                processed_samples = (idx_batch + 1) * dataloader_train.batch_size
                pbar.set_postfix({'loss': running_loss / processed_samples})
                pbar.update(x.shape[0])
            if len(dataloader_train.dataset) > 0:
                 epoch_loss = running_loss / len(dataloader_train.dataset)
                 hist_loss[idx_epoch] = epoch_loss
                 pbar.set_postfix({'loss': f'{epoch_loss:.6f}'})
            else: hist_loss[idx_epoch] = 0
    print("Training complete.")
    plot_dir = "plot_saved_close_only" # Separate plot dir
    os.makedirs(plot_dir, exist_ok=True)
    plt.figure()
    plt.plot(hist_loss, 'o-', label='train_loss')
    plt.title(f"Training Loss ({symbol}, {run_type}, {num_csvs} stocks - Close Only)")
    plt.xlabel("Epoch"); plt.ylabel("MSE Loss"); plt.legend()
    plt.savefig(os.path.join(plot_dir, f"{symbol}_{num_csvs}stocks_{run_type}_training_curve_close.pdf"))
    plt.close()
    torch.save(model.state_dict(), model_path)
    print(f"Saved model to {model_path}")
    return model

# --- MODIFIED: eval_model uses scaler_y and focuses on Close price ---
def eval_model(model, dataloader_test, symbol, num_csvs, scaler_y, output_length, run_type): # Removed scaler_X, feature_names
    if not model: print("Evaluation skipped: Model not trained."); return
    if not dataloader_test or len(dataloader_test.dataset) == 0: print("Evaluation skipped: Test dataloader empty."); return
    model.eval()
    predictions, actuals = [], []
    with torch.no_grad():
        for x, y in dataloader_test:
            if x.size(0) != dataloader_test.batch_size or y.size(0) != dataloader_test.batch_size: continue
            try:
                y_pred = model(x) # Output: [Batch, output_length]
                if y_pred.shape != y.shape:
                     print(f"Warning: Eval output shape mismatch. Expected {y.shape}, Got {y_pred.shape}. Skipping batch.")
                     continue
                predictions.append(y_pred.cpu().numpy())
                actuals.append(y.cpu().numpy())
            except Exception as e: print(f"Error during eval inference: {e}"); continue
    if not predictions: print("Error: No predictions generated."); return
    preds_scaled = np.concatenate(predictions, axis=0)
    trues_scaled = np.concatenate(actuals,     axis=0)
    if preds_scaled.size == 0 or trues_scaled.size == 0: print("Error: Eval arrays empty."); return
    P_flat_scaled = preds_scaled.flatten()
    T_flat_scaled = trues_scaled.flatten()
    mse_scaled = mean_squared_error(T_flat_scaled, P_flat_scaled)
    mae_scaled = mean_absolute_error(T_flat_scaled, P_flat_scaled)
    print(f"[SCALED Close Price] MSE={mse_scaled:.6f}, MAE={mae_scaled:.6f}")
    P_clipped_scaled = np.clip(preds_scaled, 0.0, 1.0)
    P_for_inverse = P_clipped_scaled.reshape(-1, 1)
    T_for_inverse = trues_scaled.reshape(-1, 1)
    try:
        raw_pred  = scaler_y.inverse_transform(P_for_inverse).flatten()
        raw_true  = scaler_y.inverse_transform(T_for_inverse).flatten()
    except ValueError as e: print(f"Error during inverse transform for {symbol}: {e}"); return
    print(f"Raw True shape: {raw_true.shape}, Raw Pred shape: {raw_pred.shape}")
    mask = ~np.isnan(raw_true) & ~np.isinf(raw_true) & ~np.isnan(raw_pred) & ~np.isinf(raw_pred)
    raw_true_clean = raw_true[mask]
    raw_pred_clean = raw_pred[mask]
    if raw_true_clean.shape[0] > 0:
        mse_raw = mean_squared_error(raw_true_clean, raw_pred_clean)
        mae_raw = mean_absolute_error(raw_true_clean, raw_pred_clean)
        try: r2_raw  = r2_score(raw_true_clean, raw_pred_clean)
        except ValueError: r2_raw = np.nan; print("Warning: R2 failed.")
        print(f"\nOverall raw metrics for CLOSE PRICE - {symbol} ({run_type}, {num_csvs} stocks):")
        print(f"   MSE={mse_raw:,.2f}, MAE={mae_raw:,.2f}, R2={r2_raw:.4f}")
    else:
        print(f"\nNot enough valid data points for overall CLOSE metrics for {symbol}.")
        mse_raw, mae_raw, r2_raw = np.nan, np.nan, np.nan
    plot_dir = "plot_saved_close_only"
    os.makedirs(plot_dir, exist_ok=True)
    plt.figure(figsize=(12, 6))
    plt.plot(raw_true, label="Ground Truth", alpha=0.7)
    plt.plot(raw_pred, label="Predicted", alpha=0.7, linestyle='--')
    plt.title(f"{symbol}: Close Price - Ground Truth vs Predicted ({run_type}, {num_csvs} stocks)")
    plt.xlabel("Time Steps (Flattened)"); plt.ylabel("Close Price"); plt.legend()
    plt.savefig(os.path.join(plot_dir, f"{symbol}_{num_csvs}stocks_{run_type}_prediction_close.pdf"))
    plt.close()
    if not np.isnan(mae_raw):
        eval_df = pd.DataFrame({'MAE': [mae_raw], 'MSE': [mse_raw], 'R2': [r2_raw]})
        date_str = datetime.now().strftime("%Y%m%d%H%M")
        results_dir = f"test_result_{num_csvs}_close_only"
        os.makedirs(results_dir, exist_ok=True)
        folder = os.path.join(results_dir, f"{symbol}_{date_str}_{run_type}")
        os.makedirs(folder, exist_ok=True)
        eval_df.to_csv(os.path.join(folder, f"{symbol}_{date_str}_eval_close.csv"), index=False)
        print(f"Saved evaluation to {folder}")
    else: print(f"Skipping saving metrics for {symbol} due to calculation errors.")


# --- MODIFIED: Renamed function for clarity, added run_type ---
def run_novel_predict(csv_data, symbol, num_csvs, pred_flag, pred_names):
    run_type = "NOVEL_CLOSE" # Modified run type name
    # --- CHANGE 4: Select relevant columns for NOVEL experiment (8 features) ---
    selected_columns = NOVEL_FEATURES

    if csv_data is None: print(f"Error: No data for {symbol}."); return
    available_cols_lower = [col.lower() for col in csv_data.columns]
    required_cols_lower = [col.lower() for col in selected_columns]
    missing_cols = [req for req in required_cols_lower if req not in available_cols_lower]
    if missing_cols:
        print(f"Error: Missing NOVEL columns for {symbol}. Missing: {missing_cols}"); return
    else:
         selected_columns_actual_case = []
         available_cols_map = {col.lower(): col for col in csv_data.columns}
         for req_lower in required_cols_lower: selected_columns_actual_case.append(available_cols_map[req_lower])
         selected_columns = selected_columns_actual_case
         print(f"Using columns for NOVEL: {selected_columns}")
    try:
        for col in selected_columns:
            if col in csv_data.columns: csv_data[col] = pd.to_numeric(csv_data[col], errors='coerce')
        initial_len = len(csv_data)
        csv_data = csv_data.dropna(subset=selected_columns)
        if len(csv_data) < initial_len: print(f"Warning: Dropped rows with NaN for {symbol}.")
        if len(csv_data) == 0: print(f"Error: No valid data after NaN drop for {symbol}."); return
    except Exception as e: print(f"Error during cleaning for {symbol}: {e}"); return

    data = csv_data[selected_columns].values
    d_input = len(selected_columns)  # Should be 8

    # --- Pass target index to data_processor ---
    processed_data = data_processor(data, TARGET_FEATURE_INDEX)
    if processed_data[0] is None: print(f"Data processing failed for {symbol}."); return
    dataloader_train, dataloader_test, _, scaler_y = processed_data # Need only scaler_y

    model = train_model(dataloader_train, symbol, num_csvs, d_input, run_type)

    if model is not None and dataloader_test is not None and len(dataloader_test.dataset) > 0:
         print(f"\n--- Evaluating NOVEL CLOSE for {symbol} ---")
         eval_model(model, dataloader_test, symbol, num_csvs, scaler_y, output_length=3, run_type=run_type)
    else: print(f"Skipping evaluation for {symbol}.")


# --- Main Execution Block ---
if __name__ == "__main__":
    # --- CHANGE 3: Corrected stock lists to use UPPERCASE filenames ---
    names_5 = ['KO.CSV', 'AMD.CSV', 'TSM.CSV', 'GOOG.CSV','WMT.CSV']
    names_25 = ['BHP.CSV', 'C.CSV', 'COST.CSV', 'CVX.CSV','DIS.CSV', 'GE.CSV', 'INTC.CSV', 'MSFT.CSV', 'NVDA.CSV', 'PYPL.CSV','QQQ.CSV', 'SBUX.CSV', 'T.CSV', 'TSLA.CSV', 'WFC.CSV', 'GSK.CSV', 'KO.CSV', 'AMD.CSV', 'TSM.CSV', 'GOOG.CSV', 'WMT.CSV']
    names_50 = ['AAL.CSV', 'AAPL.CSV', 'ABBV.CSV', 'AMGN.CSV','BABA.CSV', 'BHP.CSV', 'BIIB.CSV', 'BIDU.CSV', 'BRK-B.CSV','C.CSV', 'CAT.CSV', 'CMCSA.CSV', 'CMG.CSV', 'COP.CSV', 'COST.CSV', 'CRM.CSV', 'CVX.CSV', 'DIS.CSV', 'EBAY.CSV','GE.CSV','GILD.CSV', 'GLD.CSV', 'GSK.CSV', 'INTC.CSV', 'MRK.CSV', 'MSFT.CSV', 'MU.CSV', 'NKE.CSV', 'NVDA.CSV', 'ORCL.CSV', 'PEP.CSV', 'PYPL.CSV', 'QCOM.CSV', 'QQQ.CSV', 'SBUX.CSV', 'T.CSV', 'TGT.CSV', 'TM.CSV', 'TSLA.CSV', 'USO.CSV', 'V.CSV', 'WFC.CSV', 'XLF.CSV','KO.CSV', 'AMD.CSV', 'TSM.CSV', 'GOOG.CSV', 'WMT.CSV', 'AMZN.CSV', 'DAL.CSV']
    names_5 = sorted(list(set(names_5))); names_25 = sorted(list(set(names_25))); names_50 = sorted(list(set(names_50)))
    pred_names = ['KO','AMD',"TSM","GOOG",'WMT'] # Symbols

    # --- Read command-line argument ---
    if len(sys.argv) != 2 or sys.argv[1] not in ['5', '25', '50']:
        print("\nUsage: python runnovel_close_only.py [5|25|50]"); sys.exit(1)
    stock_count_arg = sys.argv[1]
    if stock_count_arg == '5': names_to_process = names_5
    elif stock_count_arg == '25': names_to_process = names_25
    else: names_to_process = names_50
    num_stocks = len(names_to_process)
    run_type = "NOVEL_CLOSE"
    print(f"\n--- 🚀 STARTING {run_type} RUN FOR {num_stocks} STOCKS ---")
    data_folder = "processed_data_novel" # CHANGE 2

    # --- Loop through stocks ---
    for name in names_to_process:
        data_path = os.path.join(data_folder, name)
        csv_data = read_csv_case_insensitive(data_path)
        if csv_data is None or csv_data.empty: print(f"Warning: Skip {data_path}."); continue
        essential_cols = ['Date', 'Volume', 'Open', 'High', 'Low', 'Close', 'Finbert_positive', 'Finbert_negative', 'Finbert_neutral']
        available_cols_lower = [col.lower() for col in csv_data.columns]
        required_cols_lower = [col.lower() for col in essential_cols]
        if not all(req in available_cols_lower for req in required_cols_lower):
            print(f"Warning: Missing essential NOVEL columns in {data_path} (case-insensitive). Skipping."); continue
        try:
             if isinstance(csv_data['Date'], pd.DataFrame): csv_data['Date'] = csv_data['Date'].iloc[:, 0]
             csv_data['Date'] = pd.to_datetime(csv_data['Date'], errors='coerce')
             csv_data = csv_data.dropna(subset=['Date'])
             csv_data = csv_data.sort_values(by='Date').reset_index(drop=True)
        except Exception as e: print(f"Error processing Date for {name}: {e}. Skipping."); continue
        symbol_name = name.split('.')[0].upper()
        print(f"\n--- Processing {run_type} for {symbol_name} ({num_stocks} stock set) ---")
        run_novel_predict(csv_data, symbol_name, num_stocks, True, pred_names) # Passing pred_flag=True
    print(f"\n--- ✅ {run_type} Training Complete for {num_stocks} stocks ---")

