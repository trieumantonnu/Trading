import pandas as pd
import numpy as np
import glob
import os

class Methods:
    def __init__(self):
        pass

    def convert_margin_cash(self, symbols, reference_df):
        '''
        Convert profit margin to profit in USD.
        '''
        df_usd_per_lot = pd.DataFrame(index=reference_df.index)
        df_usd_per_lot['USOIL'] = 1000
        df_usd_per_lot['XAUUSD'] = 100
        df_usd_per_lot['BTCUSD'] = 1

        self.df_usd_per_lot = df_usd_per_lot[symbols]


    def mask_sessions(self, session_start, session_end, reference_df):
        '''
        This aim to filter periods in some specific trading sessions
        '''

        df_ = pd.DataFrame(reference_df.index.time, index=reference_df.index)
        dict_session = dict()

        for session in list(session_start.keys()):
            start_ = (df_ > session_start[session]) & (df_.shift() <= session_start[session])
            end_ = (df_ >= session_end[session]) & (df_.shift() < session_end[session])
            dict_session[session] = self.process_signal(start_, end_)

        self.df_session = pd.concat(dict_session, axis=1).droplevel(axis=1, level=1).fillna(False)
        

    def process_signal(self, entry_cond: pd.DataFrame, exit_cond: pd.DataFrame) -> pd.DataFrame:
        """
        Vectorized (loop-free) computation of position state given entry/exit boolean DataFrames.
        Returns True between entry and exit, False elsewhere.

        Parameters:
            entry_cond (pd.DataFrame): Boolean DataFrame of entry signals
            exit_cond (pd.DataFrame): Boolean DataFrame of exit signals

        Returns:
            pd.DataFrame: Boolean position DataFrame (True = in position)
        """
        # Ensure DataFrames are aligned
        assert entry_cond.shape == exit_cond.shape
        assert entry_cond.index.equals(exit_cond.index)
        assert entry_cond.columns.equals(exit_cond.columns)

        # +1 for entry, -1 for exit, 0 for no signal
        # 1-0=1 true, 1-1=0-->-1 false, 0-1=-1 false, 0-0=0 ffill
        signal = entry_cond.astype(int) - exit_cond.astype(int)
        signal = signal.where(
                            (entry_cond.astype(int)==0) | \
                            (entry_cond.astype(int)==1) & (exit_cond.astype(int)==0),
                             -1)

        return (signal).replace({1: True, -1: False, 0: np.nan}).ffill().fillna(False)
    
    def control_trading(self):
        '''
        After the data is mapped to signals, there will be some trading constraints 
        regarding the number of holding orders, floating capital,...
        '''
        return
    
    def take_profit(self, signals, df, usd=None, pct=None):
        last_change_info = df.open.ffill()[
                                    (signals.diff()!=0) & (signals.diff().notna())
                                    ].replace(0, np.nan).ffill()
        if usd:
            diff_df = ((df.close.ffill()-last_change_info) * self.df_usd_per_lot) * signals
            exceed_cond = diff_df >= usd
            new_signal = self.process_signal((signals!=0) & ((signals.diff()!=0)), 
                                             exceed_cond | (signals==0)).fillna(False).astype(int) * signals
        elif pct:
            diff_df = ((df.close.ffill()/last_change_info - 1) * 100) * signals
            exceed_cond = diff_df >= pct
            new_signal = self.process_signal((signals!=0) & ((signals.diff()!=0)), 
                                             exceed_cond | (signals==0)).fillna(False).astype(int) * signals
            
        return new_signal
    
    def cut_loss(self, signals, df, usd=None, pct=None):
        last_change_info = df.open.ffill()[
                                    (signals.diff()!=0) & (signals.diff().notna())
                                    ].replace(0, np.nan).ffill()
        if usd:
            diff_df = ((df.close.ffill()-last_change_info) * self.df_usd_per_lot) * signals
            exceed_cond = diff_df <= -usd
            new_signal = self.process_signal((signals!=0) & ((signals.diff()!=0)), 
                                             exceed_cond | (signals==0)).fillna(False).astype(int) * signals
        elif pct:
            diff_df = ((df.close.ffill()/last_change_info - 1) * 100) * signals
            exceed_cond = diff_df <= -pct
            new_signal = self.process_signal((signals!=0) & ((signals.diff()!=0)), 
                                             exceed_cond | (signals==0)).fillna(False).astype(int) * signals
            
        return new_signal
