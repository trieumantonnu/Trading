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
import torch.optim as optim

import time
import random
from collections import deque


# =========================================================
# RL helper classes
# =========================================================

class StandardScalerNP:
    def __init__(self, split_idx):
        self.mean_ = None
        self.std_ = None
        self.split_idx = split_idx

    def fit(self, x_2d):
        self.mean_ = np.nanmean(x_2d[ : self.split_idx + 1], axis=0)
        self.std_ = np.nanstd(x_2d[ : self.split_idx + 1], axis=0)
        self.std_[self.std_ == 0] = 1.0
        return self

    def transform(self, x):
        return (x - self.mean_) / self.std_

    def fit_transform(self, x_2d):
        self.fit(x_2d)
        return self.transform(x_2d)

# class StandardScalerNP:
#     """
#     Simple numpy-based scaler for 3D sequence input.
#     Fit on 2D feature matrix, then transform 2D or 3D arrays.
#     """
#     def __init__(self, split_idx):
#         self.mean_ = None
#         self.std_ = None
#         self.split_idx = split_idx

#     def fit(self, x_2d):
#         self.mean_ = np.nanmean(x_2d[ : self.split_idx + 1], axis=0)
#         self.std_ = np.nanstd(x_2d[ : self.split_idx + 1], axis=0)
#         self.std_[self.std_ == 0] = 1.0
#         return self

#     def transform(self, x):
#         if x.ndim == 2:
#             return (x - self.mean_) / self.std_
#         elif x.ndim == 3:
#             return (x - self.mean_[None, None, :]) / self.std_[None, None, :]
#         else:
#             raise ValueError("Input must be 2D or 3D.")

#     def fit_transform(self, x_2d):
#         self.fit(x_2d)
#         return self.transform(x_2d)

class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((
            np.asarray(state, dtype=np.float32),
            int(action),
            float(reward),
            np.asarray(next_state, dtype=np.float32),
            float(done)
        ))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            np.asarray(states, dtype=np.float32),
            np.asarray(actions, dtype=np.int64),
            np.asarray(rewards, dtype=np.float32),
            np.asarray(next_states, dtype=np.float32),
            np.asarray(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


class TradingQNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dims=(128, 64), num_actions=3, dropout=0.1):
        super().__init__()

        h1, h2 = hidden_dims
        self.net = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.ReLU(),
            nn.LayerNorm(h1),
            nn.Dropout(dropout),

            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.LayerNorm(h2),
            nn.Dropout(dropout),

            nn.Linear(h2, num_actions)
        )

    def forward(self, x):
        return self.net(x)

class DuelingTradingQNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dims=(256, 128), num_actions=3, dropout=0.05):
        super().__init__()

        h1, h2 = hidden_dims

        self.feature_net = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.LayerNorm(h1),
            nn.SiLU(),
            nn.Dropout(dropout),

            nn.Linear(h1, h2),
            nn.LayerNorm(h2),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

        self.value_head = nn.Sequential(
            nn.Linear(h2, h2),
            nn.SiLU(),
            nn.Linear(h2, 1)
        )

        self.advantage_head = nn.Sequential(
            nn.Linear(h2, h2),
            nn.SiLU(),
            nn.Linear(h2, num_actions)
        )

    def forward(self, x):
        features = self.feature_net(x)

        value = self.value_head(features)
        advantage = self.advantage_head(features)

        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)

        return q_values
    
# class RecurrentTradingQNetwork(nn.Module):
#     """
#     GRU-based dueling Q-network.

#     Expected input shape:
#         (batch_size, lookback, num_features)

#     This replaces the flat MLP logic. The network now reads a time window
#     and uses the last GRU hidden output to compute Q-values.
#     """
#     def __init__(self, input_dim, hidden_dims=128, num_actions=3, dropout=0.05):
#         super().__init__()

#         if isinstance(hidden_dims, (tuple, list)):
#             hidden_dim = hidden_dims[0]
#         else:
#             hidden_dim = int(hidden_dims)

#         self.input_dim = input_dim
#         self.hidden_dim = hidden_dim
#         self.num_actions = num_actions

#         self.gru = nn.GRU(
#             input_size=input_dim,
#             hidden_size=hidden_dim,
#             batch_first=True
#         )

#         self.value_head = nn.Sequential(
#             nn.LayerNorm(hidden_dim),
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.SiLU(),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_dim, 1)
#         )

#         self.advantage_head = nn.Sequential(
#             nn.LayerNorm(hidden_dim),
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.SiLU(),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_dim, num_actions)
#         )

#     def forward(self, x):
#         if x.dim() != 3:
#             raise ValueError(
#                 f"RecurrentTradingQNetwork expected 3D input "
#                 f"(batch, lookback, features), but got shape {tuple(x.shape)}."
#             )

#         out, _ = self.gru(x)
#         last_hidden = out[:, -1, :]

#         value = self.value_head(last_hidden)
#         advantage = self.advantage_head(last_hidden)

#         q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
#         return q_values

class RecurrentTradingQNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dims=128, num_actions=3, dropout=0.05):
        super().__init__()

        if isinstance(hidden_dims, (tuple, list)):
            hidden_dim = hidden_dims[0]
        else:
            hidden_dim = int(hidden_dims)

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_actions = num_actions

        self.input_norm = nn.LayerNorm(input_dim)

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout
        )

        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

        self.shared = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout)
        )

        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

        self.advantage_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, num_actions)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        if x.dim() != 3:
            raise ValueError(
                f"RecurrentTradingQNetwork expected 3D input "
                f"(batch, lookback, features), but got shape {tuple(x.shape)}."
            )

        x = self.input_norm(x)

        out, _ = self.gru(x)

        # attention over the whole lookback window
        attn_scores = self.attention(out)              # (batch, lookback, 1)
        attn_weights = torch.softmax(attn_scores, dim=1)

        context = torch.sum(attn_weights * out, dim=1) # (batch, hidden_dim)

        features = self.shared(context)

        value = self.value_head(features)
        advantage = self.advantage_head(features)

        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)

        return q_values
    
class DQNTrader:
    def __init__(
        self,
        state_dim,
        num_actions=3,
        hidden_dims=128,
        lr=2e-3,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.995,
        target_update_freq=200,
        buffer_capacity=100000,
        batch_size=128,
        device=None,
        dropout=0.2
    ):
        self.state_dim = state_dim
        self.num_actions = num_actions
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_update_freq = target_update_freq
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.policy_net = RecurrentTradingQNetwork(
            input_dim=state_dim,
            hidden_dims=hidden_dims,
            num_actions=num_actions,
            dropout=dropout
        ).to(self.device)

        self.target_net = RecurrentTradingQNetwork(
            input_dim=state_dim,
            hidden_dims=hidden_dims,
            num_actions=num_actions,
            dropout=dropout
        ).to(self.device)

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr, weight_decay=2e-4)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
                                                                    self.optimizer,
                                                                    gamma=0.95
                                                                )
        self.loss_fn = nn.MSELoss()
        self.replay_buffer = ReplayBuffer(capacity=buffer_capacity)

        self.train_steps = 0

    def select_action(self, state, greedy=False):
        if (not greedy) and (np.random.rand() < self.epsilon):
            return np.random.randint(self.num_actions)

        state_t = torch.tensor(state, dtype=torch.float32, device=self.device)

        # GRU state must be one full window: (lookback, num_features).
        # Add batch dimension: (1, lookback, num_features).
        if state_t.dim() == 2:
            state_t = state_t.unsqueeze(0)
        else:
            raise ValueError(
                f"Invalid GRU state shape {tuple(state_t.shape)}. "
                "Expected (lookback, num_features). "
                "Build state as X[t - lookback + 1 : t + 1]."
            )

        with torch.no_grad():
            q_values = self.policy_net(state_t)

        return int(torch.argmax(q_values, dim=1).item())

    def push_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        states_t = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions_t = torch.tensor(actions, dtype=torch.long, device=self.device).unsqueeze(1)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states_t = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        dones_t = torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)

        if states_t.dim() != 3 or next_states_t.dim() != 3:
            raise ValueError(
                f"Replay buffer contains invalid state shapes. "
                f"states_t={tuple(states_t.shape)}, next_states_t={tuple(next_states_t.shape)}. "
                "For GRU-DQN, every stored state must have shape (lookback, num_features)."
            )

        q_values = self.policy_net(states_t).gather(1, actions_t)

        with torch.no_grad():
            next_q_values = self.target_net(next_states_t).max(dim=1, keepdim=True)[0]
            target_q = rewards_t + (1.0 - dones_t) * self.gamma * next_q_values

        loss = self.loss_fn(q_values, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        self.train_steps += 1
        if self.train_steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return float(loss.item())

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def save(self, model_path, feature_cols=None, lookback=None):
        torch.save({
            "policy_net_state_dict": self.policy_net.state_dict(),
            "target_net_state_dict": self.target_net.state_dict(),
            "state_dim": self.state_dim,
            "num_actions": self.num_actions,
            "epsilon": self.epsilon,
            "feature_cols": feature_cols,
            "lookback": lookback,
            "network_type": "gru_dueling"
        }, model_path)

    def load(self, model_path):
        checkpoint = torch.load(model_path, map_location=self.device)

        saved_state_dim = checkpoint.get("state_dim", None)
        if saved_state_dim is not None and int(saved_state_dim) != int(self.state_dim):
            raise ValueError(
                f"State dimension mismatch. Checkpoint state_dim={saved_state_dim}, "
                f"current state_dim={self.state_dim}. Use the same feature columns/order."
            )

        self.policy_net.load_state_dict(checkpoint["policy_net_state_dict"])
        self.target_net.load_state_dict(checkpoint["target_net_state_dict"])
        self.epsilon = checkpoint.get("epsilon", self.epsilon_end)
        self.policy_net.to(self.device)
        self.target_net.to(self.device)
        self.policy_net.eval()
        self.target_net.eval()


# =========================================================
# Your class
# =========================================================

class generateSignal(Methods):
    def __init__(self, data, session_start, session_end):
        self.data = data
        self.session_start = session_start
        self.session_end = session_end
        self.mask_sessions(
            session_start=session_start,
            session_end=session_end,
            reference_df=self.data
        )
        self.convert_margin_cash(self.data.columns.levels[1], self.data)

    def add_rsi(self, df, window=14):
        delta = df.diff()
        gain = delta.clip(lower=0).ewm(span=window, adjust=False).mean()
        loss = -delta.clip(upper=0).ewm(span=window, adjust=False).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def add_kst(self, df, r1=10, r2=15, r3=20, r4=30, n1=10, n2=10, n3=10, n4=15, signal=9):
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
    
    def compute_sharpe_ratio(self, returns, bars_per_year):
        """
        returns: 1D numpy array of strategy returns per bar
        bars_per_year: annualization factor
        """
        returns = np.asarray(returns, dtype=np.float64)
        returns = returns[np.isfinite(returns)]

        if len(returns) < 2:
            return np.nan

        std = returns.std(ddof=1)
        if std == 0:
            return np.nan

        sharpe = (returns.mean() / std) * np.sqrt(bars_per_year)
        return float(sharpe)

    def generate_signals(
        self,
        train_ratio=0.8,
        hidden_dims=128,
        lr = 3e-4,
        gamma=0.99,
        batch_size=128,
        episodes=20,
        epsilon_start=1,
        epsilon_end=0.05,
        epsilon_decay=0.98,
        target_update_freq=200,
        replay_capacity=100000,
        transaction_cost=0.0005,
        spread_penalty_scale=0.0,
        dropout=0.1,
        random_seed=88,
        train_model=True,
        load_existing_model=False,
        save_model=True,
        model_path="./log/dqn_period_1.pth",
        period_length=10000,
        target_sharpe=1,
        max_episodes_per_period=100,
        reward_horizon=2,
        bars_per_year=None,
        use_period_curriculum=True,
        periods_to_learn = 10,
        lookback = 12
    ):
        """
        Generate trading signals {-1, 0, 1} using a PyTorch DQN-style RL agent.

        Actions:
            0 -> -1
            1 ->  0
            2 ->  1

        Reward:
            reward_t = position_t * next_return_t
                       - transaction_cost * abs(position_t - prev_position)
                       - spread_penalty_scale * spread_norm_t * abs(position_t)
        """

        import os
        import time

        np.random.seed(random_seed)
        random.seed(random_seed)
        torch.manual_seed(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(random_seed)

        self.df = self.data.copy()
        self.df = self.df.replace(0, np.nan).copy()
        

        # ---------------------------------------------
        # Helper to coerce feature objects to Series
        # ---------------------------------------------
        def _to_series(x, name, index):
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

        # ---------------------------------------------
        # Feature engineering
        # ---------------------------------------------
        print('Start Feature Engineering')

        gap = self.df.close.diff().fillna(0)

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
        es_95 = pd.DataFrame(np.round(array_es_95, 0), columns=['USOIL'], index=y_vals.iloc[-array_es_95.shape[0]:].index)[self.df.close.notna()]

        array_es_05 = np.sum(np.where(mask_05, rolling_arr, 0), axis=2)/np.sum(mask_05, axis=2)
        es_05 = pd.DataFrame(np.round(array_es_05, 0), columns=['USOIL'], index=y_vals.iloc[-array_es_05.shape[0]:].index)[self.df.close.notna()]


        # ---------------------------------------------
        # Handle MultiIndex columns like ('close', 'USOIL')
        # ---------------------------------------------
        required_cols = ["close", "high", "low", "open"]
        optional_cols = ["spread", "tickvol"]

        if isinstance(self.df.columns, pd.MultiIndex):
            symbol = self.df.columns.get_level_values(1).unique()[0]

            missing_cols = [c for c in required_cols if (c, symbol) not in self.df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns for {symbol}: {missing_cols}")

            available_cols = [c for c in required_cols + optional_cols if (c, symbol) in self.df.columns]
            self.df = self.df.loc[:, pd.IndexSlice[available_cols, symbol]].copy()
            self.df.columns = self.df.columns.get_level_values(0)
        else:
            missing_cols = [c for c in required_cols if c not in self.df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns in self.df: {missing_cols}")

        for col in optional_cols:
            if col not in self.df.columns:
                self.df[col] = 0.0

        self.df_1 = self.df
        self.df = self.df.dropna(subset='close')

        log_ret_1 = np.log(self.df["close"] / self.df["close"].shift(1))
        log_ret_3 = np.log(self.df["close"] / self.df["close"].shift(3))
        log_ret_6 = np.log(self.df["close"] / self.df["close"].shift(6))

        hl_spread = (self.df["high"] - self.df["low"]) / self.df["close"].replace(0, np.nan)
        oc_spread = (self.df["close"] - self.df["open"]) / self.df["open"].replace(0, np.nan)

        spread_norm = self.df["spread"] / self.df["close"].replace(0, np.nan)
        tickvol_chg = self.df["tickvol"].pct_change()

        vol_12 = log_ret_1.rolling(12).std()
        vol_48 = log_ret_1.rolling(48).std()

        rsi = self.add_rsi(self.df["close"], 14)
        kst, kst_signal = self.add_kst(self.df["close"])

        logret = np.log(self.df.close).diff()
        london_ret = logret[self.df_session['London']].reindex(index=logret.index)
        newyork_ret = logret[self.df_session['New York']].reindex(index=logret.index)
        sydney_ret = logret[self.df_session['Sydney']].reindex(index=logret.index)
        tokyo_ret = logret[self.df_session['Tokyo']].reindex(index=logret.index)

        london_cumret = london_ret.groupby(london_ret.isna().cumsum()).cumsum().ffill().fillna(0)
        newyork_cumret = newyork_ret.groupby(newyork_ret.isna().cumsum()).cumsum().ffill().fillna(0)
        sydney_cumret = sydney_ret.groupby(sydney_ret.isna().cumsum()).cumsum().ffill().fillna(0)
        tokyo_cumret = tokyo_ret.groupby(tokyo_ret.isna().cumsum()).cumsum().ffill().fillna(0)

        trade_sessions_weighted = 2 * self.df_session['New York'].astype(int) + self.df_session['London'].astype(int)
        trade_sessions_weighted_df = pd.DataFrame(trade_sessions_weighted.reindex(index=logret.index), columns=['USOIL'])


        features = {
            "close": self.df["close"],
            "spread": self.df["spread"],
            "tickvol": self.df["tickvol"],
            "gap": gap,
            "diff_bounds": diff_bounds,
            "rsi": rsi,
            "kst": kst,
            "kst_signal": kst_signal,
            "kst_diff": kst - kst_signal,
            "log_ret_1": log_ret_1,
            "log_ret_3": log_ret_3,
            "log_ret_6": log_ret_6,
            "hl_spread": hl_spread,
            "oc_spread": oc_spread,
            "spread_norm": spread_norm,
            "tickvol_chg": tickvol_chg,
            "vol_12": vol_12,
            "vol_48": vol_48,
            'es_95' : es_95,
            'es_05' : es_05,
            'london_cumret':london_cumret,
            'newyork_cumret':newyork_cumret,
            'sydney_cumret':sydney_cumret,
            'tokyo_cumret':tokyo_cumret,
            "trade_sessions_weighted": trade_sessions_weighted_df
        }

        feat_df = pd.concat(
            [_to_series(v, k, self.df.index) for k, v in features.items()],
            axis=1
        )

        feat_df["spread_ok"] = (self.df["spread"] <= 50).astype(float)

        # Next-step return for reward
        next_ret = self.df["close"].shift(-1) / self.df["close"] - 1.0
        next_ret.name = "next_ret"

        model_df = feat_df.copy()
        model_df["next_ret"] = next_ret
        model_df = model_df.replace([np.inf, -np.inf], np.nan)
        model_df[['es_95','es_05']] = model_df[['es_95','es_05']].ffill()
        model_df = model_df.fillna(0)

        if len(model_df) < 200:
            return pd.DataFrame(data=0, index=self.df.index, columns=["USOIL"])

        feature_cols = [c for c in model_df.columns if c != "next_ret"]
        X_all = model_df[feature_cols].values.astype(np.float32)
        ret_all = model_df["next_ret"].values.astype(np.float32)
        idx_all = model_df.index

        # ---------------------------------------------
        # Scale features
        # ---------------------------------------------
        split_idx = self.df.index.get_loc('2026-03-16 00:00:00')
        scaler = StandardScalerNP(split_idx = split_idx)
        X_all = scaler.fit_transform(X_all)
        print(X_all.shape)

        # ---------------------------------------------
        # Train / test split in time order
        # ---------------------------------------------
        n_total = len(X_all)
        split_idx = int(n_total * train_ratio)
        split_idx = max(50, min(split_idx, n_total - 2))
        split_idx = self.df.index.get_loc('2026-03-16 00:00:00')

        X_train = X_all[:split_idx+1]
        ret_train = ret_all[:split_idx+1]
        # print(X_train.shape)

        X_test = X_all[split_idx:]
        ret_test = ret_all[split_idx:]
        # print(X_test.shape)
        # ---------------------------------------------
        # RL agent
        # ---------------------------------------------
        agent = DQNTrader(
            state_dim=X_train.shape[1] + 1,
            num_actions=3,
            hidden_dims=hidden_dims,
            lr=lr,
            gamma=gamma,
            epsilon_start=epsilon_start,
            epsilon_end=epsilon_end,
            epsilon_decay=epsilon_decay,
            target_update_freq=target_update_freq,
            buffer_capacity=replay_capacity,
            batch_size=batch_size,
            dropout=dropout
        )

        action_to_position = {0: -1, 1: 0, 2: 1}

        # ---------------------------------------------
        # Load existing model or train
        # ---------------------------------------------
        if load_existing_model:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")
            agent.load(model_path)
            print(f"Loaded existing RL model from: {model_path}")

        elif train_model:
            print("Start RL training")

            train_start = time.perf_counter()

            if bars_per_year is None:
                bars_per_year = 12 * 24 * 252 / reward_horizon  # rough annualization for 5-minute bars

            if not use_period_curriculum:
                raise ValueError("This block expects use_period_curriculum=True.")

            n_train = len(X_train)
            if lookback < 2:
                raise ValueError("lookback must be at least 2 for the GRU window state.")
            if n_train <= lookback + reward_horizon + 1:
                raise ValueError("Training set is too short for the chosen lookback and reward_horizon.")

            # Build contiguous periods over the training set
            period_ranges = []
            start_idx = 0
            while start_idx < n_train - reward_horizon:
                end_idx = min(start_idx + period_length, n_train)
                if end_idx - start_idx > reward_horizon:
                    period_ranges.append((start_idx, end_idx))
                start_idx = end_idx

            if len(period_ranges) == 0:
                raise ValueError("No valid training periods were created. Check period_length.")

            print(f"Training over {periods_to_learn} periods")

            # for period_num, (p_start, p_end) in enumerate(period_ranges, start=1):
            for i in range(periods_to_learn):
                period_num = i+1
                min_period_rows = max(1000, lookback + 2 * reward_horizon + 1)
                p_start = random.randint(0, max(0, n_train - 5000))
                p_end = p_start + np.minimum(random.randint(min_period_rows, 30000), n_train - p_start)

                if p_end - p_start < lookback + 2 * reward_horizon:
                    print(f"Skip period {period_num}: not enough rows for lookback/reward_horizon.")
                    continue
                print(f"\nPeriod {period_num}/{periods_to_learn} | learn {p_end - p_start} rows {p_start}:{p_end}")

                agent.epsilon = epsilon_start

                period_reached_target = False
                

                for episode_in_period in range(max_episodes_per_period):
                    prev_position = 0
                    episode_reward = 0.0
                    losses = []
                    strategy_returns = []
                    position_history = [prev_position]

                    # Walk through one period using non-overlapping windows.
                    # state shape:      (lookback, num_features)
                    # next_state shape: (lookback, num_features)
                    first_t = p_start + lookback - 1

                    for t in range(first_t, p_end - reward_horizon, reward_horizon):
                        state = X_train[t - lookback + 1 : t + 1]
                        next_state = X_train[
                            t + reward_horizon - lookback + 1 : t + reward_horizon + 1
                        ]

                        pos_col = np.full((lookback, 1), prev_position, dtype=np.float32)
                        state = np.concatenate([state, pos_col], axis=1)

                        action = agent.select_action(state, greedy=False)
                        position = action_to_position[action]

                        next_pos_col = np.full((lookback, 1), position, dtype=np.float32)
                        next_state = np.concatenate([next_state, next_pos_col], axis=1)

                        if state.shape[0] != lookback or next_state.shape[0] != lookback:
                            continue

                        action = agent.select_action(state, greedy=False)
                        position = action_to_position[action]

                        future_slice = ret_train[t:t + reward_horizon]
                        pnl_reward = position * np.sum(future_slice)

                        # if pnl_reward > 0:
                        #     pnl_reward = pnl_reward * 1.5

                        trade_cost = transaction_cost * abs(position - prev_position)

                        reward = (pnl_reward) - trade_cost

                        done = 1.0 if (t + 2 * reward_horizon >= p_end) else 0.0

                        agent.push_transition(state, action, reward, next_state, done)
                        loss = agent.update()
                        if loss is not None:
                            losses.append(loss)

                        # position_history.append(position)
                        prev_position = position
                        episode_reward += reward
                        # done = 1.0 if (episode_reward <= -1) else 0.0
                        strategy_returns.append(reward)
                    agent.scheduler.step()
                    agent.decay_epsilon()

                    avg_loss = np.mean(losses) if len(losses) > 0 else np.nan
                    period_sharpe = self.compute_sharpe_ratio(strategy_returns, bars_per_year=bars_per_year/reward_horizon)


                    target_sharpe_period = target_sharpe * np.sqrt(100 / max((p_end - p_start) / 288, 1e-8))


                    print(
                        f"Period {period_num} | "
                        f"Episode {episode_in_period+1}/{max_episodes_per_period} | "
                        f"reward={episode_reward:.6f} | "
                        f"sharpe={period_sharpe:.4f} | "
                        f"avg_loss={avg_loss:.6f} | "
                        f"epsilon={agent.epsilon:.4f}"
                    )

                    if np.isfinite(period_sharpe) and period_sharpe >= target_sharpe_period:
                        print(
                            f"Period {period_num} reached target Sharpe "
                            f"{period_sharpe:.4f} >= {target_sharpe_period:.4f}. Move to next period."
                        )
                        period_reached_target = True
                        break

                if not period_reached_target:
                    print(
                        f"Period {period_num} did not reach target Sharpe "
                        f"{target_sharpe_period:.4f} after {max_episodes_per_period} episodes. "
                        f"Move to next period anyway."
                    )

            train_end = time.perf_counter()
            train_seconds = train_end - train_start
            print(f"RL training time: {train_seconds:.2f} seconds ({train_seconds/60:.2f} minutes)")

            if save_model:
                model_dir = os.path.dirname(model_path)
                if model_dir:
                    os.makedirs(model_dir, exist_ok=True)
                agent.save(model_path, feature_cols=feature_cols, lookback=lookback)
                print(f"Saved RL model to: {model_path}")
        else:
            raise ValueError("Either train_model=True or load_existing_model=True is required.")

        # ---------------------------------------------
        # Greedy inference on all available states
        # ---------------------------------------------
        signals_arr = []
        signal_index = []

        # Greedy inference also needs a full lookback window.
        for t in range(lookback - 1, len(X_test)):
            state = X_test[t - lookback + 1 : t + 1]
            action = agent.select_action(state, greedy=True)
            signal = action_to_position[action]

            # Optional filter to suppress trades when spread is too high.
            # Use split_idx + t because model_df is indexed on the full dataset.
            spread_ok = model_df.iloc[split_idx + t]["spread_ok"] > 0.5
            if not spread_ok:
                signal = 0

            signals_arr.append(signal)
            signal_index.append(idx_all[split_idx + t])

        pred_signal = pd.Series(signals_arr, index=signal_index, name="USOIL").astype(int)

        signals = pd.DataFrame(index=self.df.index, columns=["USOIL"], data=0)
        signals.loc[pred_signal.index, "USOIL"] = pred_signal.values
        signals["USOIL"] = signals["USOIL"].fillna(0).astype(int)
        # self.df = self.df_1
        signals = signals.reindex(index=self.df_1.index)

        return signals


if __name__ == "__main__":
    import config
    from config import session_start, session_end
    # always put the trading symbol in the 1st place
    symbols = ['USOIL']  # Replace with actual filenames (without `.csv`)
    sim = Simulation(symbols)
    folder_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'DATA'))

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
        # from_time='2025-07-01', 
        # to_time='2025-08-18 16:50:00'
    )
    print(data.tail())

    generator = generateSignal(data=data, 
                               session_start=session_start, 
                               session_end=session_end)
    sim.positions = generator.generate_signals()

    sim.get_outputs()
