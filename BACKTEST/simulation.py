import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import config
from config import *
import BACKTEST
from BACKTEST.methods import Methods
from BACKTEST.plot import PlotlyChartFramework

class Simulation(Methods, PlotlyChartFramework):
    def __init__(self, symbols, positions=pd.DataFrame()):
        self.symbols = symbols
        self.positions = positions.fillna(0).astype(int)
        self.figures = []


    def add_data(self, folder_path, freq, lag, method, from_time=None, to_time=None):
        '''
        method = {'':''}
        '''
        dataframes = dict()
        freq = str(freq) + 'T'

        for symbol in self.symbols:
            file_path = os.path.join(folder_path, symbol + '.csv')
            file_ = pd.read_csv(file_path, sep='\t')
            # <DATE>	<TIME>	<OPEN>	<HIGH>	<LOW>	<CLOSE>	<TICKVOL>	<VOL>	<SPREAD>
            file_.columns = [col.replace('<', '').replace('>', '').lower() for col in file_.columns]
            file_['datetime'] = pd.to_datetime(file_['date'] + ' ' + file_['time'])
            file_.set_index('datetime', inplace=True)
            file_.drop(columns=['date', 'time'], inplace=True)
            file_ = file_.reset_index().drop_duplicates(subset='datetime', keep='last').set_index('datetime')
            # file_.date = pd.to_datetime(file_.date)
            # file_.set_index('date', inplace=True)
            file__ = file_.resample(freq).agg(method).shift(lag)
            dataframes[symbol] = file__

        combined_df = pd.concat(dataframes, axis=1)
        swapped_df = combined_df.swaplevel(axis=1).sort_index(axis=1, level=0)
        sorted_df = swapped_df.sort_index(axis=0).loc[from_time:to_time]

        self.df = sorted_df

        self.mask_sessions(session_start=session_start,
                           session_end=session_end,
                           reference_df=self.df)        

        return sorted_df
    
    def output_pnl(self):
        diff_pos = self.positions.diff()

        logret = np.log(self.df.close - self.df.spread/1000) - np.log(self.df.close).shift()
        logret_sell = np.log(self.df.close) - np.log(self.df.close - self.df.spread/1000).shift()
        self.ret = self.df.close - self.df.close.shift()
        self.ret_sell = self.df.close - (self.df.close).shift()

        # change positions pnl
        ret_by_gap = self.df.open - self.df.close.shift()
        self.pnl_change_positions = (diff_pos.shift() * ret_by_gap)



        london_ret = logret[self.df_session['London']].reindex(index=logret.index)
        newyork_ret = logret[self.df_session['New York']].reindex(index=logret.index)
        sydney_ret = logret[self.df_session['Sydney']].reindex(index=logret.index)
        tokyo_ret = logret[self.df_session['Tokyo']].reindex(index=logret.index)

        london_cumret = london_ret.groupby(london_ret.iloc[:, 0].isna().cumsum()).cumsum()
        newyork_cumret = newyork_ret.groupby(newyork_ret.iloc[:, 0].isna().cumsum()).cumsum()
        sydney_cumret = sydney_ret.groupby(sydney_ret.iloc[:, 0].isna().cumsum()).cumsum()
        tokyo_cumret = tokyo_ret.groupby(tokyo_ret.iloc[:, 0].isna().cumsum()).cumsum()
        # 0011100
        self.pnl_long = self.ret[self.process_signal(self.positions == 1, self.positions != 1).shift()].reindex(index=self.ret.index)
        self.pnl_short = self.ret_sell[self.process_signal(self.positions == -1, self.positions != -1).shift()].reindex(index=self.ret.index)
        self.pnl_london = self.pnl_long[self.df_session['London']].reindex(index=self.ret.index).\
                        fillna(-self.pnl_short[self.df_session['London']])
        self.pnl_newyork = self.pnl_long[self.df_session['New York']].reindex(index=self.ret.index).\
                        fillna(-self.pnl_short[self.df_session['New York']])
        self.pnl_sydney = self.pnl_long[self.df_session['Sydney']].reindex(index=self.ret.index).\
                        fillna(-self.pnl_short[self.df_session['Sydney']])
        self.pnl_tokyo = self.pnl_long[self.df_session['Tokyo']].reindex(index=self.ret.index).\
                        fillna(-self.pnl_short[self.df_session['Tokyo']])

    def draw_stats_table(self, stats_dict):
        # Format values based on type
        def format_val(v):
            if isinstance(v, (float, int)):
                return f"{v:.4f}"
            elif isinstance(v, list):
                return ", ".join(str(i) for i in v)
            else:
                return str(v)

        # Compute max width for each column
        col1_width = max(len("Statistic"), max(len(str(k)) for k in stats_dict))
        col2_width = max(len("Value"), max(len(format_val(v)) for v in stats_dict.values()))

        # Table headers and borders
        separator = f"|{'-' * (col1_width + 2)}|{'-' * (col2_width + 2)}|"
        header = f"| {'Statistic'.ljust(col1_width)} | {'Value'.ljust(col2_width)} |"

        # Print the table
        print(separator)
        print(header)
        print(separator)
        for stat, val in stats_dict.items():
            val_str = format_val(val)
            row = f"| {stat.ljust(col1_width)} | {val_str.rjust(col2_width)} |"
            print(row)
        print(separator)

    def get_drawdown(self, cumulative_pnl):
        cummax_pnl = cumulative_pnl.copy().cummax()
        cumpnl_ratio = np.maximum((cumulative_pnl.copy() + 7500)/(cummax_pnl + 7500), 0)
        dd = 1 - cumpnl_ratio
        # dd = 1 - cumulative_pnl/cummax_pnl
        return dd.replace([-np.inf, np.inf], np.nan)


    def get_statistics(self):
        '''
        Statistics table: 
        - Sharpe
        - Profit
        - Winrate
        - Winrate KC
        - Turnover
        - No. orders
        - Intraday no. orders
        - Mean margin per order
        - Risk/Margin-reward
        - Drawdown

        Long-short analysis table
        - 

        '''
        self.output_pnl()
        self.convert_margin_cash(self.symbols,
                                 self.df)
        self.pnl = (
                    self.pnl_long.fillna(0) - self.pnl_short.fillna(0) - self.pnl_change_positions
                    ) * self.df_usd_per_lot
        self.pnl = self.pnl[self.pnl.copy().replace(0, np.nan).notna().cumsum() >= 1].dropna()
        self.margin = self.df.close * margin[self.symbols[0]]/10


        # log positions and pnl
        to_save_df = pd.concat([self.positions, self.pnl], axis=1)
        to_save_df.columns = pd.MultiIndex.from_tuples([
                                                        ("positions", lvl1) for lvl1 in self.positions.columns
                                                        ] + \
                                                        [
                                                        ("pnl", lvl1) for lvl1 in self.pnl.columns
                                                        ]
                                                        )
        log_filename = self.symbols[0]
        log_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'log'))
        log_path = os.path.join(log_dir, log_filename + '.csv')
        to_save_df.to_csv(log_path)

        # compute stats
        self.change_long = ((self.positions==1) & (self.positions.shift()!=1))
        self.change_short = ((self.positions==-1) & (self.positions.shift()!=-1))
        # temporarily for long only, need to add short
        nanret_withpos_indices = list(
                                        set(self.pnl.index).difference(set(self.pnl.dropna().index)).\
                                        difference(set(self.positions[self.positions==0].dropna().index))
                                    )
        self.pnl.loc[nanret_withpos_indices] = 0
        self.cumpnl_bytrade = self.pnl[self.positions.shift().fillna(0)!=0].groupby(
                                                                self.pnl[self.positions.shift().fillna(0)!=0].iloc[:, 0].isna().cumsum()
                                                                ).cumsum()
        lastpnl_bytrade = self.cumpnl_bytrade.shift()[
                                                        (self.cumpnl_bytrade.isna()) & \
                                                        (self.cumpnl_bytrade.shift().notna())
                                                        ].dropna()
        describe_pnl_bytrade = lastpnl_bytrade.describe()
        self.cumulative_pnl = self.pnl.cumsum()

        # cumulative_pnl = self.cumulative_pnl
        # cummax_pnl = cumulative_pnl.cummax()
        # cumpnl_ratio = np.maximum((cumulative_pnl + 7500)/(cummax_pnl + 7500), 0)
        # dd = 1 - cumpnl_ratio
        # dd.replace([-np.inf, np.inf], np.nan)

        self.dd = self.get_drawdown(self.cumulative_pnl)
        self.daily_pnl = self.pnl.groupby(self.pnl.index.date).sum()
        self.trade_margin = self.margin * (
                                            self.change_long.astype(int) + self.change_short.astype(int)
                                            )
        winrate = ((lastpnl_bytrade>0).sum()/describe_pnl_bytrade.loc['count'])[0]
        mean_win = lastpnl_bytrade[lastpnl_bytrade>0].dropna().mean()
        mean_loss = lastpnl_bytrade[lastpnl_bytrade<=0].dropna().mean()
        winrate_kc = winrate - ((1 - winrate)/(mean_win/-mean_loss))
        self.daily_reward_risk = (
                            self.daily_pnl/\
                                ((self.trade_margin).groupby(self.df.index.date).sum())
                            ).replace([np.inf, -np.inf], np.nan)
        
        for date_ in self.positions[self.positions.diff().fillna(0)!=0].dropna().index:
            pos_ = self.positions[self.positions.diff().fillna(0)!=0].dropna().loc[date_][0]
            price_ = self.df.close.ffill()[self.positions.diff().fillna(0)!=0].dropna().loc[date_][0]

            if pos_ > 0:
                print('Enter LONG at', date_,  'near', price_)
            elif pos_ < 0:
                print('Enter SHORT at', date_,  'near', price_)
            elif pos_ == 0:
                print('Close positions at', date_,  'near', price_)
                print('===============================================')


        # Construct stats table
        self.stats = dict()
        self.stats['Trading Symbols'] = self.symbols
        self.stats['Sharpe'] = (
                                self.daily_pnl.mean().div(self.daily_pnl.std()) * \
                                np.sqrt(trading_days[self.symbols[0]])
                                ).iloc[0]
        self.stats['PnL'] = self.pnl.sum().iloc[0]
        self.stats['Turnover (daily)'] = self.positions.fillna(0).diff().abs().sum(axis=1).sum()/\
                                            self.df.groupby(self.df.index.date).size().shape[0]
        self.stats['Mean Reward/Risk'] = self.daily_reward_risk.mean()[0]
        self.stats['Median Drawdown'] = self.dd.dropna().median()[0]
        self.stats['Winrate'] = winrate
        self.stats['Winrate KC'] = winrate_kc[0]
        self.stats['No. Long Trades'] = self.change_long.sum()[0]
        self.stats['No. Short Trades'] = self.change_short.sum()[0]
        self.stats['No. Trades'] = describe_pnl_bytrade.loc['count'][0]
        self.stats['Mean Margin/trade'] = self.trade_margin.replace(0, np.nan).mean()[0]
        self.stats['Lowest PnL/trade'] = describe_pnl_bytrade.loc['min'][0]
        self.stats['P25 PnL/trade'] = describe_pnl_bytrade.loc['25%'][0]
        self.stats['Mean PnL/trade'] = describe_pnl_bytrade.loc['mean'][0]
        self.stats['P75 PnL/trade'] = describe_pnl_bytrade.loc['75%'][0]
        self.stats['Max PnL/trade'] = describe_pnl_bytrade.loc['max'][0]


        self.draw_stats_table(self.stats)

    
    def get_outputs(self):
        self.get_statistics()

        self.plot_performance(cumulative_pnl = self.cumulative_pnl, 
                              positions = self.positions.loc[self.pnl.index], 
                              drawdown=self.dd, 
                              daily_reward_risk = self.daily_reward_risk,
                              stats=self.stats,
                              title='Performance Visualization', 
                              browser_name=None)
        return
        
# add analysis by order - winrate, long-short, number, intraday number, open time, closing time, holding period
# modify format for multiple symbols
# add plots analyzing pnl/pos in different sessions
# add bigger timeframe function
# add 1m to cutloss
# solve problem of adding positions with session's constraint, 
# if one side out then one side left then trade occurs in the non favorable sessions.
# think more about .shift() of positions while calculating the pnl