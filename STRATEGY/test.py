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

    def generate_signals(self):
        '''
        A function to map data to signals
        '''

        self.df = self.data.copy()
        self.df = self.df.replace(0, np.nan)
        # self.df.dropna(axis=0, how='all', inplace=True)

        # Features to be added
        rsi = self.add_rsi(self.df.close, 14)
        kst, kst_signal = self.add_kst(self.df.close)


        # return buy --> return sell
        logret = np.log(self.df.close - self.df.spread/1000) - np.log(self.df.close).shift()
        logret_sell = np.log(self.df.close) - np.log(self.df.close - self.df.spread/1000).shift()
        ret = self.df.close - self.df.close.shift()
        ret_sell = self.df.close - self.df.close.shift()


        london_ret = logret[self.df_session['London']].reindex(index=logret.index)
        newyork_ret = logret[self.df_session['New York']].reindex(index=logret.index)
        sydney_ret = logret[self.df_session['Sydney']].reindex(index=logret.index)
        tokyo_ret = logret[self.df_session['Tokyo']].reindex(index=logret.index)

        london_cumret = london_ret.groupby(london_ret['USOIL'].isna().cumsum()).cumsum()
        newyork_cumret = newyork_ret.groupby(newyork_ret['USOIL'].isna().cumsum()).cumsum()
        sydney_cumret = sydney_ret.groupby(sydney_ret['USOIL'].isna().cumsum()).cumsum()
        tokyo_cumret = tokyo_ret.groupby(tokyo_ret['USOIL'].isna().cumsum()).cumsum()



        # Assumptions to be tested
        entry_cond = (kst > kst_signal * 0.95) & (kst < -4) & \
                    (self.df.spread < 20) 
        # & \
        #             pd.DataFrame(self.df_session['London'] | self.df_session['New York'], columns=['USOIL'])
        exit_cond = ((kst < kst_signal * 1.05) & (kst > 0))
        kst_long_signals = self.process_signal(entry_cond=entry_cond,
                                               exit_cond=exit_cond)
        
        entry_cond = (kst < kst_signal * 1.05) & (kst > 4) & \
                    (self.df.spread < 20) 
        exit_cond = ((kst > kst_signal * 0.95) & (kst < -0))
        kst_short_signals = self.process_signal(entry_cond=entry_cond,
                                               exit_cond=exit_cond)
        # london_newyork_signals = (london_cumret.ffill() > 0)[newyork_cumret.notna()].fillna(False)

        signals_long = (kst_long_signals.iloc[:, 0]).fillna(False).astype(int)
        signals_short = (kst_short_signals.iloc[:, 0]).fillna(False).astype(int)
        signals = pd.DataFrame(signals_long, columns = ['USOIL'])

        # Take profit and cut loss
        # signals = self.take_profit(signals=signals, df=self.df, usd=300)
        signals = self.cut_loss(signals=signals, df=self.df, usd=20)
        # print(signals.tail(20))
        # # print((self.df[['open', 'close']]).tail(20))
        # x = [1, 2, 3, 4, 5]
        # y = [60, 70, 80, 90, 100]

        # plt.bar(x, y, 
        #         # width=1.0, align='center'
        #         )
        # plt.xlabel("Subject")
        # plt.ylabel("Scores")
        # plt.title("Student A")
        # plt.show()
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