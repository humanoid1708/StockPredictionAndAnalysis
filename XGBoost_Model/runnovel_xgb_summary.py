import os
import pandas as pd
import numpy as np
import glob
import sys
import matplotlib.pyplot as plt
from datetime import datetime
from xgboost import XGBRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tqdm import tqdm

RUN_TYPE_NAME = "NOVEL_XGB_CLOSE"
SELECTED_COLUMNS = ['Date', 'Volume', 'Open', 'High', 'Low', 'Close', 
                    'Finbert_positive', 'Finbert_negative', 'Finbert_neutral']
TARGET_FEATURE_NAME = 'Close'
N_ESTIMATORS = 1000

def create_sequences(data, input_length, output_length, target_feature_index):
    X, y = [], []
    for i in range(len(data) - input_length - output_length + 1):
        X.append(data[i:(i + input_length), :])
        y.append(data[(i + input_length):(i + input_length + output_length), target_feature_index])
    return np.array(X), np.array(y)

def read_csv_case_insensitive(file_path):
    try:
        return pd.read_csv(file_path)
    except FileNotFoundError:
        directory, filename = os.path.split(file_path)
        if not directory: directory = "."
        pattern = os.path.join(directory, ''.join(['[{}{}]'.format(c.lower(), c.upper()) if c.isalpha() else c for c in filename]))
        matching_files = glob.glob(pattern)
        if matching_files:
            print(f"Reading case-insensitive match: {matching_files[0]}")
            return pd.read_csv(matching_files[0])
        else:
            print(f"No file matches the pattern: {file_path}"); return None
    except Exception as e:
        print(f"An error occurred reading {file_path}: {e}"); return None

def data_processor_xgb(data, target_feature_index):
    input_length = 50
    output_length = 3
    split_ratio = 0.85
    split_idx = int(split_ratio * len(data))
    
    raw_train = data[:split_idx]
    raw_test  = data[split_idx:]
    
    if raw_train.shape[0] < (input_length + output_length) or raw_test.shape[0] < (input_length + output_length):
        print(f"Warning: Not enough data for train/test split. Train: {raw_train.shape[0]}, Test: {raw_test.shape[0]}")
        return None, None, None, None, None 

    scaler_X = MinMaxScaler().fit(raw_train)
    scaler_y = MinMaxScaler().fit(raw_train[:, target_feature_index].reshape(-1, 1))

    train_scaled = scaler_X.transform(raw_train)
    test_scaled  = scaler_X.transform(raw_test)

    X_train_scaled, y_train_unscaled = create_sequences(train_scaled, input_length, output_length, target_feature_index)
    X_test_scaled,  y_test_unscaled  = create_sequences(test_scaled,  input_length, output_length, target_feature_index)

    y_train_scaled = scaler_y.transform(y_train_unscaled.reshape(-1, 1)).reshape(y_train_unscaled.shape)
    y_test_scaled  = scaler_y.transform(y_test_unscaled.reshape(-1, 1)).reshape(y_test_unscaled.shape)

    if X_train_scaled.shape[0] == 0 or X_test_scaled.shape[0] == 0:
        print(f"Warning: Not enough data after sequencing. X_train: {X_train_scaled.shape[0]}, X_test: {X_test_scaled.shape[0]}")
        return None, None, None, None, None 

    print(f'X_train: {X_train_scaled.shape}, X_test: {X_test_scaled.shape}, y_train: {y_train_scaled.shape}, y_test: {y_test_scaled.shape}')
    
    return X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled, scaler_y

def run_xgb_experiment(csv_data, symbol, num_csvs, pred_flag, pred_names):
    
    print(f"Using features: {SELECTED_COLUMNS}")
    
    try:
        target_col_actual = [col for col in csv_data.columns if col.lower() == TARGET_FEATURE_NAME.lower()][0]
        target_feature_index = [col.lower() for col in SELECTED_COLUMNS].index(TARGET_FEATURE_NAME.lower())
    except IndexError:
        print(f"Error: Target column '{TARGET_FEATURE_NAME}' not found. Skipping.")
        return None

    csv_data['Date'] = csv_data['Date'].apply(lambda x: x.timestamp())

    data = csv_data[SELECTED_COLUMNS].values
    
    processed_data = data_processor_xgb(data, target_feature_index)
    if processed_data[0] is None:
        print(f"Data processing failed for {symbol}. Skipping.")
        return None
        
    X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled, scaler_y = processed_data

    n_samples, seq_len, n_features = X_train_scaled.shape
    X_train_flat = X_train_scaled.reshape(n_samples, seq_len * n_features)
    X_test_flat = X_test_scaled.reshape(X_test_scaled.shape[0], seq_len * n_features)

    print("Training XGBoost...")
    model = XGBRegressor(n_estimators=N_ESTIMATORS,
                         learning_rate=0.1,
                         objective='reg:squarederror',
                         n_jobs=-1, 
                         random_state=42)
                            
    model.fit(X_train_flat, y_train_scaled)
    print("Training complete.")
    
    y_pred_scaled = model.predict(X_test_flat) 
    
    mse_scaled = mean_squared_error(y_test_scaled, y_pred_scaled)
    mae_scaled = mean_absolute_error(y_test_scaled, y_pred_scaled)
    print(f"[SCALED Close Price] MSE={mse_scaled:.6f}, MAE={mae_scaled:.6f}")

    raw_true_day1 = scaler_y.inverse_transform(y_test_scaled[:, 0].reshape(-1, 1)).flatten()
    raw_pred_day1 = scaler_y.inverse_transform(y_pred_scaled[:, 0].reshape(-1, 1)).flatten()
    raw_true_day2 = scaler_y.inverse_transform(y_test_scaled[:, 1].reshape(-1, 1)).flatten()
    raw_pred_day2 = scaler_y.inverse_transform(y_pred_scaled[:, 1].reshape(-1, 1)).flatten()
    raw_true_day3 = scaler_y.inverse_transform(y_test_scaled[:, 2].reshape(-1, 1)).flatten()
    raw_pred_day3 = scaler_y.inverse_transform(y_pred_scaled[:, 2].reshape(-1, 1)).flatten()

    raw_true = np.concatenate([raw_true_day1, raw_true_day2, raw_true_day3])
    raw_pred = np.concatenate([raw_pred_day1, raw_pred_day2, raw_pred_day3])

    mse_raw = mean_squared_error(raw_true, raw_pred)
    mae_raw = mean_absolute_error(raw_true, raw_pred)
    r2_raw  = r2_score(raw_true, raw_pred)
    
    print(f"\nOverall raw metrics for CLOSE PRICE - {symbol} ({RUN_TYPE_NAME}, {num_csvs} stocks):")
    print(f"   MSE={mse_raw:,.2f}, MAE={mae_raw:,.2f}, R2={r2_raw:.4f}")
    
    
    print(f"--- Saving plot for: {symbol} ---")
    plot_dir = f"plot_saved_{RUN_TYPE_NAME}"
    os.makedirs(plot_dir, exist_ok=True)
    plt.figure(figsize=(12, 6))
    plt.plot(raw_true_day1, label="Ground Truth (Day 1)", alpha=0.7)
    plt.plot(raw_pred_day1, label="Predicted (Day 1)", alpha=0.7, linestyle='--')
    plt.title(f"{symbol}: Close Price Prediction (Day 1) - {RUN_TYPE_NAME}")
    plt.xlabel("Test Sample Index"); plt.ylabel("Close Price"); plt.legend()
    plt.savefig(os.path.join(plot_dir, f"{symbol}_{num_csvs}stocks_prediction.pdf"))
    plt.close()

    
    if pred_flag and symbol.upper() in pred_names:
        print(f"--- Saving individual *CSV report* for evaluation stock: {symbol} ---")

        eval_df = pd.DataFrame({'Stock': [symbol], 'MAE': [mae_raw], 'MSE': [mse_raw], 'R2': [r2_raw]})
        date_str = datetime.now().strftime("%Y%m%d%H%M")
        results_dir = f"test_result_{num_csvs}_{RUN_TYPE_NAME}"
        os.makedirs(results_dir, exist_ok=True)
        folder = os.path.join(results_dir, f"{symbol}_{date_str}")
        os.makedirs(folder, exist_ok=True)
        eval_df.to_csv(os.path.join(folder, f"{symbol}_{date_str}_eval.csv"), index=False)
        print(f"Saved evaluation CSV to {folder}")

    return {'Stock': symbol, 'MAE': mae_raw, 'MSE': mse_raw, 'R2': r2_raw}


if __name__ == "__main__":
    names_5 = ['KO.CSV', 'AMD.CSV', 'TSM.CSV', 'GOOG.CSV','WMT.CSV']
    names_25 = [
        'BHP.CSV', 'C.CSV', 'COST.CSV', 'CVX.CSV','DIS.CSV', 'GE.CSV',
        'INTC.CSV', 'MSFT.CSV', 'NVDA.CSV', 'PYPL.CSV','QQQ.CSV', 'SBUX.CSV', 'T.CSV', 'TSLA.CSV', 'WFC.CSV', 'GSK.CSV',
        'KO.CSV', 'AMD.CSV', 'TSM.CSV', 'GOOG.CSV', 'WMT.CSV'
    ]
    names_50 = [
       'AAL.CSV', 'AAPL.CSV', 'ABBV.CSV', 'AMGN.CSV','BABA.CSV', 'BHP.CSV',
        'BIIB.CSV', 'BIDU.CSV', 'BRK-B.CSV','C.CSV', 'CAT.CSV', 'CMCSA.CSV', 'CMG.CSV', 'COP.CSV', 'COST.CSV', 'CRM.CSV', 'CVX.CSV',
        'DIS.CSV', 'EBAY.CSV','GE.CSV','GILD.CSV', 'GLD.CSV', 'GSK.CSV', 'INTC.CSV',
        'MRK.CSV', 'MSFT.CSV', 'MU.CSV', 'NKE.CSV', 'NVDA.CSV', 'ORCL.CSV',
        'PEP.CSV', 'PYPL.CSV', 'QCOM.CSV', 'QQQ.CSV', 'SBUX.CSV', 'T.CSV', 'TGT.CSV', 'TM.CSV', 'TSLA.CSV', 'USO.CSV', 'V.CSV',
        'WFC.CSV', 'XLF.CSV','KO.CSV', 'AMD.CSV', 'TSM.CSV', 'GOOG.CSV', 'WMT.CSV',
        'AMZN.CSV', 'DAL.CSV'
    ]
    names_5 = sorted(list(set(names_5)))
    names_25 = sorted(list(set(names_25)))
    names_50 = sorted(list(set(names_50)))

    pred_names = ['KO','AMD',"TSM","GOOG",'WMT']

    if len(sys.argv) != 2 or sys.argv[1] not in ['5', '25', '50']:
        print(f"\nUsage: python {sys.argv[0]} [5|25|50]")
        sys.exit(1)

    stock_count_arg = sys.argv[1]
    if stock_count_arg == '5': names_to_process = names_5
    elif stock_count_arg == '25': names_to_process = names_25
    else: names_to_process = names_50

    num_stocks = len(names_to_process)
    print(f"\n--- 🚀 STARTING {RUN_TYPE_NAME} RUN FOR {num_stocks} STOCKS (Estimators={N_ESTIMATORS}) ---")
    
    data_folder = "processed_data_novel"

    all_results_list = []

    for name in tqdm(names_to_process, desc=f"Processing {num_stocks} Stocks"):
        data_path = os.path.join(data_folder, name)
        csv_data = read_csv_case_insensitive(data_path)
        
        if csv_data is None or csv_data.empty:
            print(f"Warning: Could not read or empty file: {data_path}. Skipping.")
            continue

        try:
            date_col_actual = [col for col in csv_data.columns if col.lower() == 'date']
            if not date_col_actual:
                    print(f"Error: 'Date' column not found in {data_path}. Skipping.")
                    continue
            date_col_actual = date_col_actual[0]
            
            csv_data[date_col_actual] = pd.to_datetime(csv_data[date_col_actual], errors='coerce')
            csv_data = csv_data.dropna(subset=[date_col_actual])
            csv_data = csv_data.sort_values(by=date_col_actual).reset_index(drop=True)

            required_cols_lower = [col.lower() for col in SELECTED_COLUMNS]
            available_cols_lower = [col.lower() for col in csv_data.columns]
            if not all(req in available_cols_lower for req in required_cols_lower):
                print(f"Warning: Missing one or more required columns in {data_path}. Skipping.")
                missing = [req for req in required_cols_lower if req not in available_cols_lower]
                print(f"Missing: {missing}")
                continue
            
            col_map = {col: req for req in SELECTED_COLUMNS for col in csv_data.columns if col.lower() == req.lower()}
            csv_data = csv_data[list(col_map.keys())].rename(columns=col_map)
            
            numeric_cols = [col for col in SELECTED_COLUMNS if col.lower() != 'date']
            csv_data[numeric_cols] = csv_data[numeric_cols].apply(pd.to_numeric, errors='coerce')
            csv_data = csv_data.dropna(subset=SELECTED_COLUMNS)
            
        except Exception as e:
            print(f"Error processing or sorting {name}: {e}. Skipping.")
            continue
            
        if csv_data.empty:
             print(f"No valid data left for {name} after cleaning. Skipping.")
             continue

        symbol_name = name.split('.')[0].upper()
        print(f"\n--- Processing {RUN_TYPE_NAME} for {symbol_name} ({num_stocks} stock set) ---")
        
        metrics = run_xgb_experiment(csv_data, symbol_name, num_stocks, True, pred_names)
        if metrics:
            all_results_list.append(metrics)

    print(f"\n--- ✅ {RUN_TYPE_NAME} Training Complete for {num_stocks} stocks ---")
    
    if all_results_list:
        results_df = pd.DataFrame(all_results_list)
        results_df = results_df.sort_values(by='R2', ascending=False)
        summary_filename = f"summary_results_{RUN_TYPE_NAME}_{num_stocks}stocks.csv"
        results_df.to_csv(summary_filename, index=False)
        print(f"\nSuccessfully saved summary of ALL {len(all_results_list)} stocks to: {summary_filename}")
    else:
        print("\nNo results were generated to save in a summary file.")