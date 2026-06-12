import pandas as pd
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
# Ensure you're adding the project root ("Trading/") to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import BACKTEST
from BACKTEST.simulation import Simulation
from BACKTEST.methods import Methods
import config
from config import *

import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import time

# =========================================================
# PyTorch helper classes
# =========================================================

class SequenceDataset(Dataset):
    """
    Dataset for LSTM sequence classification.
    X: numpy array of shape (n_samples, seq_len, n_features)
    y: numpy array of shape (n_samples,)
    """
    def __init__(self, X, y=None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = None if y is None else torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is None:
            return self.X[idx]
        return self.X[idx], self.y[idx]


class LSTMSignalModel(nn.Module):
    """
    3-class classifier:
        class 0 -> signal -1
        class 1 -> signal  0
        class 2 -> signal +1
    """
    def __init__(self, input_size, hidden_size=64, num_layers=3, dropout=0.2, num_classes=3):
        super().__init__()

        effective_dropout = dropout if num_layers > 1 else 0.0

        self.head = nn.Sequential(
                                nn.Linear(hidden_size, 64),
                                nn.ReLU(),
                                nn.Dropout(dropout),
                                nn.Linear(64, num_classes)
                            )

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=effective_dropout
        )
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        out, _ = self.lstm(x)
        # take last timestep output
        out = out[:, -1, :]
        out = self.fc(out)
        # out = self.head(out)
        return out


class StandardScalerNP:
    """
    Simple numpy-based scaler for 3D sequence input.
    Fit on 2D feature matrix, then transform 2D or 3D arrays.
    """
    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, x_2d):
        self.mean_ = np.nanmean(x_2d, axis=0)
        self.std_ = np.nanstd(x_2d, axis=0)
        self.std_[self.std_ == 0] = 1.0
        return self

    def transform(self, x):
        if x.ndim == 2:
            return (x - self.mean_) / self.std_
        elif x.ndim == 3:
            return (x - self.mean_[None, None, :]) / self.std_[None, None, :]
        else:
            raise ValueError("Input must be 2D or 3D.")

    def fit_transform(self, x_2d):
        self.fit(x_2d)
        return self.transform(x_2d)


class LSTMTrainer:
    """
    Small trainer wrapper so generate_signals stays cleaner.
    """
    def __init__(
        self,
        input_size,
        hidden_size=128,
        num_layers=2,
        dropout=0.2,
        lr=1e-3,
        epochs=10,
        batch_size=128,
        device=None
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = LSTMSignalModel(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            num_classes=3
        ).to(self.device)

        self.epochs = epochs
        self.batch_size = batch_size
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def fit(self, X_train, y_train):
        ds = SequenceDataset(X_train, y_train)
        dl = DataLoader(ds, batch_size=self.batch_size, shuffle=True, drop_last=False)

        self.model.train()
        for _ in range(self.epochs):
            for xb, yb in dl:
                xb = xb.to(self.device)
                yb = yb.to(self.device)

                self.optimizer.zero_grad()
                logits = self.model(xb)
                loss = self.criterion(logits, yb)
                loss.backward()
                self.optimizer.step()

    @torch.no_grad()
    def predict(self, X):
        ds = SequenceDataset(X, y=None)
        dl = DataLoader(ds, batch_size=self.batch_size, shuffle=False, drop_last=False)

        self.model.eval()
        preds = []
        for xb in dl:
            xb = xb.to(self.device)
            logits = self.model(xb)
            pred = torch.argmax(logits, dim=1).cpu().numpy()
            preds.append(pred)

        preds = np.concatenate(preds, axis=0)
        return preds


# =========================================================
# Your original class, with generate_signals modified
# =========================================================

class generateSignal(Methods):
    def __init__(self, data, session_start, session_end):
        self.data = data
        self.session_start = session_start
        self.session_end = session_end
        self.mask_sessions(session_start=session_start,
                           session_end=session_end,
                           reference_df=self.data)
        self.convert_margin_cash(self.data.columns.levels[1],
                                 self.data)

    def add_rsi(self, df, window=14):
        delta = df.diff()
        gain = delta.clip(lower=0).ewm(span=window, adjust=False).mean()
        loss = -delta.clip(upper=0).ewm(span=window, adjust=False).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def add_kst(self, df, r1=10, r2=15, r3=20, r4=30, n1=10, n2=10, n3=10, n4=15, signal=9):
        """
        Calculates the Know Sure Thing (KST) oscillator.
        """
        roc1 = df.pct_change(periods=r1)
        roc2 = df.pct_change(periods=r2)
        roc3 = df.pct_change(periods=r3)
        roc4 = df.pct_change(periods=r4)

        rcma1 = roc1.rolling(window=n1).mean()
        rcma2 = roc2.rolling(window=n2).mean()
        rcma3 = roc3.rolling(window=n3).mean()
        rcma4 = roc4.rolling(window=n4).mean()

        kst = 100 * (rcma1 + 2 * rcma2 + 3 * rcma3 + 4 * rcma4)
        kst_signal = kst.rolling(window=signal).mean()

        return kst, kst_signal

    def add_bounds(self, df, window):
        highest_low = df.low.rolling(window).max()
        lowest_high = df.high.rolling(window).min()

        cond_gap = highest_low > lowest_high
        cond_gap = cond_gap > cond_gap.shift()

        return cond_gap

    def add_es(self, df, window):
        return
    
    def _to_series(self, x, name, index):
        """
        Convert Series / 1-col DataFrame / ndarray to a 1D Series with given index.
        """
        if isinstance(x, pd.DataFrame):
            if x.shape[1] != 1:
                raise ValueError(f"{name} has shape {x.shape}; expected 1 column.")
            s = x.iloc[:, 0]
        elif isinstance(x, pd.Series):
            s = x
        else:
            arr = np.asarray(x).reshape(-1)
            if len(arr) != len(index):
                raise ValueError(f"{name} length {len(arr)} does not match index length {len(index)}.")
            s = pd.Series(arr, index=index, name=name)

        s = s.copy()
        s.name = name
        return s.reindex(index)

    def generate_signals(
        self,
        seq_len=120,
        future_horizon=9,
        return_threshold=0.002,
        hidden_size=128,
        num_layers=2,
        dropout=0.1,
        lr=1e-3,
        epochs=15,
        batch_size=128,
        train_ratio=0.7,
        random_seed=42
    ):
        """
        Generate trading signals {-1, 0, 1} using a PyTorch LSTM.

        Label construction:
            future_ret = close[t + future_horizon] / close[t] - 1

            future_ret >  return_threshold  -> +1
            future_ret < -return_threshold  -> -1
            else                            ->  0

        Output:
            pd.DataFrame with one column ['USOIL'] and values in {-1, 0, 1}
        """
        print('Start generate signals')
        # -------------------------------------------------
        # Reproducibility
        # -------------------------------------------------
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(random_seed)

        # -------------------------------------------------
        # Prepare working dataframe
        # -------------------------------------------------
        self.df = self.data.copy()
        self.df = self.df.replace(0, np.nan).copy()

        # # If columns are still multiindex after your preprocessing,
        # # flatten only when needed.
        # if isinstance(self.df.columns, pd.MultiIndex):
        #     # Try to reduce to second level if that is the field name level
        #     # Adjust this if your column structure differs.
        #     if self.df.columns.nlevels >= 2:
        #         self.df.columns = self.df.columns.get_level_values(-1)

        # required_cols = ["close", "high", "low", "open"]
        # symbol = "USOIL"
        # print(self.df.columns)
        # missing_cols = [c for c in required_cols if (c, symbol) not in self.df.columns]
        # if missing_cols:
        #     raise ValueError(f"Missing required columns in self.df: {missing_cols}")

        # # spread and tickvol are optional but useful
        # if "spread" not in self.df.columns:
        #     self.df["spread"] = 0.0
        # if "tickvol" not in self.df.columns:
        #     self.df["tickvol"] = 0.0

        # -------------------------------------------------
        # Feature engineering
        # -------------------------------------------------
        print('Start Feature Engineering')
        bounds_short_signal = self.add_bounds(self.df, 6)
        bounds_short = self.df["close"][bounds_short_signal].ffill()

        bounds_long_signal = self.add_bounds(self.df, 12)
        bounds_long = self.df["close"][bounds_long_signal].ffill()

        diff_bounds = bounds_short - bounds_long
        y_vals = round(bounds_long, 0)
        rolling_arr = np.lib.stride_tricks.sliding_window_view(
            y_vals,
            window_shape=288*7,
            axis=0
        )
        q95 = np.quantile(rolling_arr, 0.95, axis=2, keepdims=True)   # shape (28260, 1, 1)
        mask_95 = rolling_arr >= q95                                      # shape (28260, 1, 2016)

        q05 = np.quantile(rolling_arr, 0.05, axis=2, keepdims=True)   # shape (28260, 1, 1)
        mask_05 = rolling_arr <= q05                                      # shape (28260, 1, 2016)

        array_es_95 = np.sum(np.where(mask_95, rolling_arr, 0), axis=2)/np.sum(mask_95, axis=2)
        es_95 = pd.DataFrame(np.round(array_es_95, 0), columns=['USOIL'], index=y_vals.iloc[-array_es_95.shape[0]:].index)

        array_es_05 = np.sum(np.where(mask_05, rolling_arr, 0), axis=2)/np.sum(mask_05, axis=2)
        es_05 = pd.DataFrame(np.round(array_es_05, 0), columns=['USOIL'], index=y_vals.iloc[-array_es_05.shape[0]:].index)


        rsi = self.add_rsi(self.df["close"], 14)
        kst, kst_signal = self.add_kst(self.df["close"])

        log_ret_1 = np.log(self.df["close"] / self.df["close"].shift(1))

        hl_spread = (self.df["high"] - self.df["low"]) / self.df["close"].replace(0, np.nan)
        oc_spread = (self.df["close"] - self.df["open"]) / self.df["open"].replace(0, np.nan)

        spread_norm = self.df["spread"] / self.df["close"].replace(0, np.nan)
        tickvol_chg = self.df["tickvol"].pct_change()

        vol_12 = log_ret_1.rolling(6).std()
        vol_48 = log_ret_1.rolling(12).std()

        logret = np.log(self.df.close).diff()
        london_ret = logret[self.df_session['London']].reindex(index=logret.index)
        newyork_ret = logret[self.df_session['New York']].reindex(index=logret.index)
        sydney_ret = logret[self.df_session['Sydney']].reindex(index=logret.index)
        tokyo_ret = logret[self.df_session['Tokyo']].reindex(index=logret.index)

        london_cumret = london_ret.groupby(london_ret.isna().cumsum().iloc[:, 0]).cumsum().ffill().fillna(0)
        newyork_cumret = newyork_ret.groupby(newyork_ret.isna().cumsum().iloc[:, 0]).cumsum().ffill().fillna(0)
        sydney_cumret = sydney_ret.groupby(sydney_ret.isna().cumsum().iloc[:, 0]).cumsum().ffill().fillna(0)
        tokyo_cumret = tokyo_ret.groupby(tokyo_ret.isna().cumsum().iloc[:, 0]).cumsum().ffill().fillna(0)

        features = {
            "diff_bounds": diff_bounds,
            "rsi": rsi,
            "kst_diff": kst - kst_signal,
            "hl_spread": hl_spread,
            "oc_spread": oc_spread,
            'es_95' : es_95,
            'es_05' : es_05,
            'london_cumret':london_cumret,
            'newyork_cumret':newyork_cumret,
            'sydney_cumret':sydney_cumret,
            'tokyo_cumret':tokyo_cumret,
        }

        feat_df = pd.concat(
            [self._to_series(v, k, self.df.index) for k, v in features.items()],
            axis=1
        )

        # Optional basic session/microstructure filter as a feature
        feat_df["spread_ok"] = (self.df["spread"] <= 20).astype(float)

        # -------------------------------------------------
        # Build target from future returns
        # -------------------------------------------------
        future_ret = self.df["close"].shift(-future_horizon) / self.df["close"] - 1.0
        
        future_min = self.df["close"].shift(-1).rolling(window=future_horizon, min_periods=future_horizon).min().shift(-(future_horizon-1))
        future_max = self.df["close"].shift(-1).rolling(window=future_horizon, min_periods=future_horizon).max().shift(-(future_horizon-1))

        min_ret = future_min / self.df["close"] - 1.0
        max_ret = future_max / self.df["close"] - 1.0

        cond_long = (future_ret > return_threshold) & (min_ret > -return_threshold * 0.5)
        cond_short = (future_ret < -return_threshold) &  (max_ret < return_threshold * 0.5)

        # class mapping:
        # 0 -> -1
        # 1 ->  0
        # 2 -> +1
        y_raw = np.where(
            cond_long, 1,
            np.where(cond_short, -1, 0)
        )

        class_map = {-1: 0, 0: 1, 1: 2}
        y_class = pd.Series(y_raw.flatten(), index=feat_df.index).map(class_map)

        # -------------------------------------------------
        # Clean rows before sequencing
        # -------------------------------------------------
        model_df = feat_df.copy()
        model_df["target"] = y_class

        model_df = model_df.replace([np.inf, -np.inf], np.nan)
        model_df = model_df.fillna(0).copy()

        if len(model_df) < seq_len + 50:
            # not enough data to train an LSTM
            return pd.DataFrame(
                data=0,
                index=self.df.index,
                columns=["USOIL"]
            )

        feature_cols = [c for c in model_df.columns if c != "target"]
        feature_values = model_df[feature_cols].values.astype(np.float32)
        target_values = model_df["target"].values.astype(np.int64)
        target_index = model_df.index

        # -------------------------------------------------
        # Scale features
        # -------------------------------------------------
        scaler = StandardScalerNP()
        feature_values_scaled = scaler.fit_transform(feature_values)

        # -------------------------------------------------
        # Create sequences
        # Each sequence ending at time t predicts target at time t
        # -------------------------------------------------
        X_seq = []
        y_seq = []
        idx_seq = []

        for i in range(seq_len - 1, len(feature_values_scaled)):
            start = i - seq_len + 1
            end = i + 1
            X_seq.append(feature_values_scaled[start:end])
            y_seq.append(target_values[i])
            idx_seq.append(target_index[i])

        X_seq = np.asarray(X_seq, dtype=np.float32)
        y_seq = np.asarray(y_seq, dtype=np.int64)
        idx_seq = pd.Index(idx_seq)

        if len(X_seq) < 100:
            return pd.DataFrame(
                data=0,
                index=self.df.index,
                columns=["USOIL"]
            )

        # -------------------------------------------------
        # Train/test split by time order
        # -------------------------------------------------
        split_idx = int(len(X_seq) * train_ratio)
        split_idx = max(split_idx, 1)
        split_idx = min(split_idx, len(X_seq) - 1)
        # print(split_idx)
        # print(self.df.index[split_idx]) 2025-10-29 11:15:00
        split_idx = self.df.index.get_loc('2025-10-29 11:15:00') - (seq_len - 1)

        X_train = X_seq[:split_idx]
        y_train = y_seq[:split_idx]

        X_test = X_seq[split_idx:]
        y_test = y_seq[split_idx:]

        # -------------------------------------------------
        # Train / Load LSTM
        # -------------------------------------------------
        print("Prepare LSTM")

        trainer = LSTMTrainer(
            input_size=X_train.shape[-1],
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            lr=lr,
            epochs=epochs,
            batch_size=batch_size
        )

        # Example expected arguments in generate_signals:
        load_existing_model=True
        save_model=True
        model_path="./log/lstm_model4.pth"

        if load_existing_model:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")

            checkpoint = torch.load(model_path, map_location=trainer.device)
            trainer.model.load_state_dict(checkpoint["model_state_dict"])
            trainer.model.to(trainer.device)
            trainer.model.eval()

            print(f"Loaded existing model from: {model_path}")

        else:

            print("Start Train LSTM")
            train_start = time.perf_counter()

            trainer.fit(X_train, y_train)

            train_end = time.perf_counter()
            train_seconds = train_end - train_start
            print(f"LSTM training time: {train_seconds:.2f} seconds ({train_seconds/60:.2f} minutes)")

            if save_model:
                os.makedirs(os.path.dirname(model_path), exist_ok=True) if os.path.dirname(model_path) else None

                torch.save({
                    "model_state_dict": trainer.model.state_dict(),
                    "input_size": X_train.shape[-1],
                    "hidden_size": hidden_size,
                    "num_layers": num_layers,
                    "dropout": dropout,
                    "lr": lr,
                    "epochs": epochs,
                    "batch_size": batch_size,
                }, model_path)

                print(f"Saved trained model to: {model_path}")

        # -------------------------------------------------
        # Predict on all available sequences
        # -------------------------------------------------
        print('Start Predict')
        print(X_test.shape)
        pred_class = trainer.predict(X_test)

        inv_class_map = {0: -1, 1: 0, 2: 1}
        pred_signal = pd.DataFrame(pd.Series(pred_class, index=idx_seq[split_idx:]).map(inv_class_map).astype(int), columns=['USOIL'])

        # -------------------------------------------------
        # Optional post-filtering
        # Keep same spirit as your old logic:
        # do not trade when spread is too high
        # -------------------------------------------------
        trade_sessions = pd.DataFrame((self.df_session['New York'] | self.df_session['London']).reindex(pred_signal.index))
        trade_sessions.columns = ['USOIL']

        spread_filter = (self.df["spread"] <= 20).reindex(pred_signal.index).fillna(False)
        pred_signal = pred_signal[self.process_signal((pred_signal != 0) & (pred_signal.shift() == 0) & spread_filter, pred_signal==0)].fillna(0)
        # pred_signal = pred_signal[self.process_signal(((pred_signal != 0) & (pred_signal.shift() == 0)) & trade_sessions, pred_signal==0)].fillna(0)

        # -------------------------------------------------
        # Reindex back to original dataframe index
        # -------------------------------------------------
        signals = pred_signal.reindex(index=self.df.index)
        signals["USOIL"] = signals["USOIL"].fillna(0).astype(int)
        

        # signals = self.cut_loss(signals = signals, df = self.df, usd=150)

        return signals


if __name__ == "__main__":
    import config
    from config import session_start, session_end
    # always put the trading symbol in the 1st place
    symbols = ['USOIL']  # Replace with actual filenames (without `.csv`)
    sim = Simulation(symbols)
    folder_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'DATA'))
    # folder_path = '/Users/manton/Documents/Code/Trading/RAW_DATA'

    # Adjust folder path and method to your case
    data = sim.add_data(
        folder_path=folder_path,
        freq=5,
        lag=1,
        method={
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'spread': 'max',
            'tickvol': 'sum'
        },
        # from_time='2025-10-28', 
        # to_time='2025-08-18 16:50:00'
    )
    print(data.tail())

    generator = generateSignal(data=data, 
                               session_start=session_start, 
                               session_end=session_end)
    sim.positions = generator.generate_signals()

    sim.get_outputs()