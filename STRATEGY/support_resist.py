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
        
        Parameters:
            df (pd.Series): A pandas Series of closing prices.
            r1, r2, r3, r4 (int): ROC periods.
            n1, n2, n3, n4 (int): SMA periods for each ROC.
            signal (int): Signal line smoothing period.
            
        Returns:
            pd.DataFrame: A DataFrame with 'KST' and 'KST_Signal' columns.
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

    def generate_signals(self):
        '''
        A function to map data to signals
        '''

        self.df = self.data.copy()
        self.df = self.df.replace(0, np.nan)
        # self.df.dropna(axis=0, how='all', inplace=True)

        # Features to be added
        bounds_short_signal = self.add_bounds(self.df, 3)
        bounds_short = self.df.close[bounds_short_signal].ffill()
        bounds_long_signal = self.add_bounds(self.df, 6)
        bounds_long = self.df.close[bounds_long_signal].ffill()

        diff_bounds = bounds_short - bounds_long

        y_vals = round(bounds_long, 0)
        x_vals= y_vals.index

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

        range_ = es_95 - es_05
        diff_ = self.df.close - es_05

        ratio_ = diff_/range_

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

        # Assumptions to be tested
        entry_cond = (diff_bounds > 0) & (ratio_ > 0.35) & (ratio_.shift() <= 0.35) & \
                    (self.df.spread <= self.df.spread.rolling(24).quantile(0.8))  
        exit_cond = ((ratio_ > 0.5) | (ratio_ < 0.3))
        pre_long_signals = self.process_signal(entry_cond=entry_cond,
                                               exit_cond=exit_cond)
        
        entry_cond = (diff_bounds < - 0) & (ratio_ < 1 - 0.35) & (ratio_.shift() >= 1 - 0.35) & \
                    (self.df.spread <= self.df.spread.rolling(24).quantile(0.8))  
        exit_cond = (ratio_ < 1 - 0.5) | (ratio_ > 1 - 0.3)
        pre_short_signals = self.process_signal(entry_cond=entry_cond,
                                               exit_cond=exit_cond)
        # london_newyork_signals = (london_cumret.ffill() > 0)[newyork_cumret.notna()].fillna(False)

        signals_long = (pre_long_signals.iloc[:, 0]).fillna(False).astype(int)
        signals_short = (pre_short_signals.iloc[:, 0]).fillna(False).astype(int)
        signals = pd.DataFrame(signals_long-signals_short, columns = ['USOIL'])

        # Take profit and cut loss
        # signals = self.take_profit(signals=signals, df=self.df, usd=300)
        signals = self.cut_loss(signals=signals, df=self.df, usd=20)
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