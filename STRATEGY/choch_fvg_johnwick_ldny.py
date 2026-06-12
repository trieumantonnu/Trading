import pandas as pd
import numpy as np
import os
import sys
import warnings

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import BACKTEST
from BACKTEST.simulation import Simulation
from BACKTEST.methods import Methods
import config
from config import *

warnings.filterwarnings("ignore")


class generateSignal(Methods):
    """
    Improved London/New York ICT-style alpha.

    Core logic:
        1. Trade only London and New York.
        2. Find first CHoCH of the session.
        3. After CHoCH, wait for same-direction FVG.
        4. Wait for FVG retest/rejection.
        5. Require John Wick candle or strong displacement confirmation.
        6. Enter with FVG/ATR-based stop and take-profit.
        7. One trade per session to reduce churn.

    Signal:
        1  = long
        -1 = short
        0  = flat
    """

    def __init__(self, data, session_start, session_end):
        self.data = data
        self.session_start = session_start
        self.session_end = session_end

        self.mask_sessions(
            session_start=session_start,
            session_end=session_end,
            reference_df=self.data
        )

        self.convert_margin_cash(
            self.data.columns.levels[1],
            self.data
        )

    # -------------------------------------------------
    # Basic helpers
    # -------------------------------------------------

    def _field(self, name):
        return self.df[name].astype(float)

    def _trade_session_mask(self):
        london = self.df_session["London"].reindex(self.df.index).fillna(False).astype(bool)
        newyork = self.df_session["New York"].reindex(self.df.index).fillna(False).astype(bool)
        return london | newyork

    def _session_id(self, trade_mask):
        start_new_session = trade_mask & (~trade_mask.shift(1).fillna(False))
        sid = start_new_session.cumsum()
        sid = sid.where(trade_mask, np.nan)
        return sid

    def _mask_to_df(self, mask, columns):
        return pd.DataFrame(
            np.repeat(mask.values.reshape(-1, 1), len(columns), axis=1),
            index=mask.index,
            columns=columns
        ).astype(bool)

    def _ffill_by_session(self, df, session_id):
        out = pd.DataFrame(index=df.index, columns=df.columns, dtype=float)

        for sid in session_id.dropna().unique():
            idx = session_id[session_id == sid].index
            out.loc[idx] = df.loc[idx].ffill()

        return out

    def _first_true_by_session(self, cond, session_id):
        out = pd.DataFrame(False, index=cond.index, columns=cond.columns)

        for col in cond.columns:
            for sid in session_id.dropna().unique():
                idx = session_id[session_id == sid].index
                s = cond.loc[idx, col].fillna(False)

                if s.any():
                    first_idx = s[s].index[0]
                    out.loc[first_idx, col] = True

        return out

    def _bars_since_event(self, event, session_id):
        out = pd.DataFrame(np.nan, index=event.index, columns=event.columns)

        for col in event.columns:
            for sid in session_id.dropna().unique():
                idx = session_id[session_id == sid].index
                counter = np.nan
                values = []

                for t in idx:
                    if bool(event.loc[t, col]):
                        counter = 0
                    elif not pd.isna(counter):
                        counter += 1

                    values.append(counter)

                out.loc[idx, col] = values

        return out

    def _bars_in_session(self, session_id, columns):
        out = pd.DataFrame(np.nan, index=session_id.index, columns=columns)

        for sid in session_id.dropna().unique():
            idx = session_id[session_id == sid].index
            values = np.arange(len(idx))
            out.loc[idx, :] = values.reshape(-1, 1)

        return out

    def _bars_to_session_end(self, session_id, columns):
        out = pd.DataFrame(np.nan, index=session_id.index, columns=columns)

        for sid in session_id.dropna().unique():
            idx = session_id[session_id == sid].index
            values = np.arange(len(idx))[::-1]
            out.loc[idx, :] = values.reshape(-1, 1)

        return out

    # -------------------------------------------------
    # Indicators and regime filters
    # -------------------------------------------------

    def atr(self, window=14):
        high = self._field("high")
        low = self._field("low")
        close = self._field("close")

        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()

        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        return tr.rolling(window).mean()

    def trend_filter(self, fast=24, slow=96, slope_window=12):
        """
        Regime filter to avoid random countertrend FVGs.

        Long bias:
            close > slow EMA, fast EMA > slow EMA, slow EMA slope positive.

        Short bias:
            close < slow EMA, fast EMA < slow EMA, slow EMA slope negative.
        """
        close = self._field("close")

        ema_fast = close.ewm(span=fast, adjust=False).mean()
        ema_slow = close.ewm(span=slow, adjust=False).mean()
        slow_slope = ema_slow - ema_slow.shift(slope_window)

        long_bias = (
            (close > ema_slow)
            & (ema_fast > ema_slow)
            & (slow_slope > 0)
        )

        short_bias = (
            (close < ema_slow)
            & (ema_fast < ema_slow)
            & (slow_slope < 0)
        )

        return long_bias.fillna(False), short_bias.fillna(False)

    def spread_filter(self, window=48, quantile=0.85):
        """
        Avoid expensive execution periods.
        Slightly stricter than before, but not too restrictive.
        """
        spread = self._field("spread")
        threshold = spread.rolling(window).quantile(quantile)
        return (spread <= threshold).fillna(True)

    # -------------------------------------------------
    # CHoCH logic
    # -------------------------------------------------

    def confirmed_swings(self, left=2, right=2):
        """
        Confirmed swing high/low.

        A swing at candle t-right is confirmed at candle t.
        """
        high = self._field("high")
        low = self._field("low")

        window = left + right + 1

        shifted_high = high.shift(right)
        shifted_low = low.shift(right)

        rolling_high = high.rolling(window=window).max()
        rolling_low = low.rolling(window=window).min()

        confirmed_high = shifted_high.where(shifted_high == rolling_high)
        confirmed_low = shifted_low.where(shifted_low == rolling_low)

        last_swing_high = confirmed_high.ffill()
        last_swing_low = confirmed_low.ffill()

        return last_swing_high, last_swing_low

    def first_choch(self, trade_mask, session_id, left=2, right=2):
        close = self._field("close")
        columns = close.columns

        trade_df = self._mask_to_df(trade_mask, columns)
        last_swing_high, last_swing_low = self.confirmed_swings(left=left, right=right)

        bull_break = (
            trade_df
            & last_swing_high.notna()
            & (close > last_swing_high)
            & (close.shift(1) <= last_swing_high.shift(1))
        )

        bear_break = (
            trade_df
            & last_swing_low.notna()
            & (close < last_swing_low)
            & (close.shift(1) >= last_swing_low.shift(1))
        )

        first_bull = self._first_true_by_session(bull_break, session_id)
        first_bear = self._first_true_by_session(bear_break, session_id)

        choch_dir = pd.DataFrame(0, index=close.index, columns=columns, dtype=float)
        choch_dir[first_bull] = 1
        choch_dir[first_bear] = -1

        conflict = first_bull & first_bear
        choch_dir[conflict] = 0

        active_choch_dir = self._ffill_by_session(
            choch_dir.replace(0, np.nan),
            session_id
        ).fillna(0)

        first_choch_event = choch_dir != 0
        bars_since_choch = self._bars_since_event(first_choch_event, session_id)

        return choch_dir, active_choch_dir, bars_since_choch

    # -------------------------------------------------
    # FVG logic
    # -------------------------------------------------

    def fair_value_gap(self):
        high = self._field("high")
        low = self._field("low")

        bull_fvg = low > high.shift(2)
        bull_lower = high.shift(2).where(bull_fvg)
        bull_upper = low.where(bull_fvg)

        bear_fvg = high < low.shift(2)
        bear_lower = high.where(bear_fvg)
        bear_upper = low.shift(2).where(bear_fvg)

        return bull_fvg, bear_fvg, bull_lower, bull_upper, bear_lower, bear_upper

    def active_fvg_after_choch(
        self,
        active_choch_dir,
        bars_since_choch,
        session_id,
        atr_,
        min_fvg_atr=0.08,
        max_fvg_atr=1.20,
        max_bars_after_choch=36
    ):
        """
        Keep only useful FVGs.

        Too tiny FVGs are noise.
        Too large FVGs often occur after exhausted displacement.
        """
        bull_fvg, bear_fvg, bull_lower, bull_upper, bear_lower, bear_upper = self.fair_value_gap()

        bull_size = (bull_upper - bull_lower).abs()
        bear_size = (bear_upper - bear_lower).abs()

        valid_after_choch = (
            bars_since_choch.notna()
            & (bars_since_choch >= 0)
            & (bars_since_choch <= max_bars_after_choch)
        )

        bull_size_ok = (
            (bull_size >= min_fvg_atr * atr_)
            & (bull_size <= max_fvg_atr * atr_)
        )

        bear_size_ok = (
            (bear_size >= min_fvg_atr * atr_)
            & (bear_size <= max_fvg_atr * atr_)
        )

        valid_bull_fvg = (
            bull_fvg
            & bull_size_ok
            & (active_choch_dir == 1)
            & valid_after_choch
        )

        valid_bear_fvg = (
            bear_fvg
            & bear_size_ok
            & (active_choch_dir == -1)
            & valid_after_choch
        )

        active_bull_lower = self._ffill_by_session(
            bull_lower.where(valid_bull_fvg),
            session_id
        )

        active_bull_upper = self._ffill_by_session(
            bull_upper.where(valid_bull_fvg),
            session_id
        )

        active_bear_lower = self._ffill_by_session(
            bear_lower.where(valid_bear_fvg),
            session_id
        )

        active_bear_upper = self._ffill_by_session(
            bear_upper.where(valid_bear_fvg),
            session_id
        )

        return (
            valid_bull_fvg,
            valid_bear_fvg,
            active_bull_lower,
            active_bull_upper,
            active_bear_lower,
            active_bear_upper
        )

    # -------------------------------------------------
    # Candle confirmation
    # -------------------------------------------------

    def john_wick_candle(self, wick_body_mult=1.0, wick_range_frac=0.30):
        """
        John Wick candle.

        Long:
            large lower wick and close in upper half.

        Short:
            large upper wick and close in lower half.
        """
        open_ = self._field("open")
        high = self._field("high")
        low = self._field("low")
        close = self._field("close")

        body = (close - open_).abs()
        candle_range = (high - low).replace(0, np.nan)

        upper_body = pd.DataFrame(
            np.maximum(open_.values, close.values),
            index=open_.index,
            columns=open_.columns
        )

        lower_body = pd.DataFrame(
            np.minimum(open_.values, close.values),
            index=open_.index,
            columns=open_.columns
        )

        upper_wick = high - upper_body
        lower_wick = lower_body - low

        body_safe = body.replace(0, np.nan)
        close_location = (close - low) / candle_range

        bullish_jw = (
            (lower_wick >= wick_body_mult * body_safe)
            & ((lower_wick / candle_range) >= wick_range_frac)
            & (close_location >= 0.55)
        )

        bearish_jw = (
            (upper_wick >= wick_body_mult * body_safe)
            & ((upper_wick / candle_range) >= wick_range_frac)
            & (close_location <= 0.45)
        )

        return bullish_jw.fillna(False), bearish_jw.fillna(False)

    def displacement_candle(self, atr_, mult=0.75):
        """
        Strong candle confirmation.

        This is allowed as an alternative to John Wick only when the FVG setup is clean.
        """
        open_ = self._field("open")
        high = self._field("high")
        low = self._field("low")
        close = self._field("close")

        candle_range = high - low
        close_location = (close - low) / candle_range.replace(0, np.nan)

        bull_disp = (
            (close > open_)
            & (candle_range >= mult * atr_)
            & (close_location >= 0.65)
        )

        bear_disp = (
            (close < open_)
            & (candle_range >= mult * atr_)
            & (close_location <= 0.35)
        )

        return bull_disp.fillna(False), bear_disp.fillna(False)

    def fvg_retest_or_rejection(
        self,
        active_bull_lower,
        active_bull_upper,
        active_bear_lower,
        active_bear_upper,
        atr_,
        tolerance_atr=0.10
    ):
        """
        Retest condition.

        Long:
            price trades into or near bullish FVG,
            then closes above midpoint.

        Short:
            price trades into or near bearish FVG,
            then closes below midpoint.
        """
        high = self._field("high")
        low = self._field("low")
        close = self._field("close")

        bull_mid = (active_bull_lower + active_bull_upper) / 2
        bear_mid = (active_bear_lower + active_bear_upper) / 2

        tolerance = tolerance_atr * atr_

        bull_retest = (
            active_bull_lower.notna()
            & active_bull_upper.notna()
            & (low <= active_bull_upper + tolerance)
            & (close >= bull_mid)
        )

        bear_retest = (
            active_bear_lower.notna()
            & active_bear_upper.notna()
            & (high >= active_bear_lower - tolerance)
            & (close <= bear_mid)
        )

        return bull_retest.fillna(False), bear_retest.fillna(False)

    # -------------------------------------------------
    # Stateful position builder with TP/SL
    # -------------------------------------------------

    def build_positions_with_risk(
        self,
        long_entry,
        short_entry,
        trade_df,
        session_id,
        active_bull_lower,
        active_bear_upper,
        atr_,
        max_hold_bars=30,
        cooldown_bars=8,
        stop_atr_buffer=0.25,
        reward_risk=1.50,
        one_trade_per_session=True
    ):
        close = self._field("close")
        high = self._field("high")
        low = self._field("low")

        signals = pd.DataFrame(0, index=close.index, columns=close.columns, dtype=int)

        for col in close.columns:
            pos = 0
            hold = 0
            cooldown = 0
            entry_price = np.nan
            stop_price = np.nan
            take_price = np.nan
            traded_sessions = set()

            for t in close.index:
                current_session = session_id.loc[t]

                in_session = bool(trade_df.loc[t, col]) if pd.notna(trade_df.loc[t, col]) else False
                le = bool(long_entry.loc[t, col]) if pd.notna(long_entry.loc[t, col]) else False
                se = bool(short_entry.loc[t, col]) if pd.notna(short_entry.loc[t, col]) else False

                c = close.loc[t, col]
                h = high.loc[t, col]
                l = low.loc[t, col]
                atr_val = atr_.loc[t, col]

                if cooldown > 0:
                    cooldown -= 1

                if pd.isna(current_session):
                    pos = 0
                    hold = 0
                    signals.loc[t, col] = 0
                    continue

                already_traded = (
                    one_trade_per_session
                    and current_session in traded_sessions
                )

                if pos == 0:
                    hold = 0

                    if cooldown == 0 and in_session and not already_traded:
                        if le and not se:
                            raw_stop = active_bull_lower.loc[t, col] - stop_atr_buffer * atr_val

                            if pd.notna(raw_stop) and raw_stop < c:
                                risk = c - raw_stop
                                entry_price = c
                                stop_price = raw_stop
                                take_price = c + reward_risk * risk
                                pos = 1
                                hold = 1
                                traded_sessions.add(current_session)

                        elif se and not le:
                            raw_stop = active_bear_upper.loc[t, col] + stop_atr_buffer * atr_val

                            if pd.notna(raw_stop) and raw_stop > c:
                                risk = raw_stop - c
                                entry_price = c
                                stop_price = raw_stop
                                take_price = c - reward_risk * risk
                                pos = -1
                                hold = 1
                                traded_sessions.add(current_session)

                elif pos == 1:
                    hold += 1

                    stop_hit = pd.notna(stop_price) and l <= stop_price
                    take_hit = pd.notna(take_price) and h >= take_price
                    session_exit = not in_session
                    time_exit = hold >= max_hold_bars
                    opposite = se

                    if stop_hit or take_hit or session_exit or time_exit or opposite:
                        pos = 0
                        hold = 0
                        cooldown = cooldown_bars
                        entry_price = np.nan
                        stop_price = np.nan
                        take_price = np.nan

                elif pos == -1:
                    hold += 1

                    stop_hit = pd.notna(stop_price) and h >= stop_price
                    take_hit = pd.notna(take_price) and l <= take_price
                    session_exit = not in_session
                    time_exit = hold >= max_hold_bars
                    opposite = le

                    if stop_hit or take_hit or session_exit or time_exit or opposite:
                        pos = 0
                        hold = 0
                        cooldown = cooldown_bars
                        entry_price = np.nan
                        stop_price = np.nan
                        take_price = np.nan

                signals.loc[t, col] = pos

        return signals

    # -------------------------------------------------
    # Diagnostics
    # -------------------------------------------------

    def print_diagnostics(
        self,
        trade_df,
        choch_dir,
        valid_bull_fvg,
        valid_bear_fvg,
        bullish_jw,
        bearish_jw,
        bull_retest,
        bear_retest,
        long_entry,
        short_entry,
        signals
    ):
        print("\n========== Improved Alpha Diagnostics ==========")
        print("Trade-session bars:")
        print(trade_df.sum())

        print("\nFirst bullish CHoCH count:")
        print((choch_dir == 1).sum())

        print("\nFirst bearish CHoCH count:")
        print((choch_dir == -1).sum())

        print("\nValid bullish FVG after CHoCH count:")
        print(valid_bull_fvg.sum())

        print("\nValid bearish FVG after CHoCH count:")
        print(valid_bear_fvg.sum())

        print("\nBullish John Wick count:")
        print(bullish_jw.sum())

        print("\nBearish John Wick count:")
        print(bearish_jw.sum())

        print("\nBullish FVG retest/rejection count:")
        print(bull_retest.sum())

        print("\nBearish FVG retest/rejection count:")
        print(bear_retest.sum())

        print("\nRaw long entry count:")
        print(long_entry.sum())

        print("\nRaw short entry count:")
        print(short_entry.sum())

        print("\nFinal nonzero position bars:")
        print((signals != 0).sum())

        print("\nApprox. number of long entries:")
        print(((signals == 1) & (signals.shift(1).fillna(0) != 1)).sum())

        print("\nApprox. number of short entries:")
        print(((signals == -1) & (signals.shift(1).fillna(0) != -1)).sum())
        print("================================================\n")

    # -------------------------------------------------
    # Main signal function
    # -------------------------------------------------

    def generate_signals(self):
        self.df = self.data.copy()
        self.df = self.df.replace(0, np.nan)

        close = self._field("close")
        columns = close.columns

        trade_mask = self._trade_session_mask()
        session_id = self._session_id(trade_mask)
        trade_df = self._mask_to_df(trade_mask, columns)

        bars_in_session = self._bars_in_session(session_id, columns)
        bars_to_end = self._bars_to_session_end(session_id, columns)

        # Avoid first few bars and last few bars of London/NY.
        session_quality = (
            (bars_in_session >= 3)
            & (bars_to_end >= 6)
        ).fillna(False)

        atr_ = self.atr(window=14)

        long_bias, short_bias = self.trend_filter(
            fast=24,
            slow=96,
            slope_window=12
        )

        choch_dir, active_choch_dir, bars_since_choch = self.first_choch(
            trade_mask=trade_mask,
            session_id=session_id,
            left=2,
            right=2
        )

        (
            valid_bull_fvg,
            valid_bear_fvg,
            active_bull_lower,
            active_bull_upper,
            active_bear_lower,
            active_bear_upper
        ) = self.active_fvg_after_choch(
            active_choch_dir=active_choch_dir,
            bars_since_choch=bars_since_choch,
            session_id=session_id,
            atr_=atr_,
            min_fvg_atr=0.05,
            max_fvg_atr=1.20,
            max_bars_after_choch=36
        )

        bullish_jw, bearish_jw = self.john_wick_candle(
            wick_body_mult=1.0,
            wick_range_frac=0.30
        )

        bull_disp, bear_disp = self.displacement_candle(
            atr_=atr_,
            mult=0.75
        )

        bull_retest, bear_retest = self.fvg_retest_or_rejection(
            active_bull_lower=active_bull_lower,
            active_bull_upper=active_bull_upper,
            active_bear_lower=active_bear_lower,
            active_bear_upper=active_bear_upper,
            atr_=atr_,
            tolerance_atr=0.10
        )

        spread_ok = self.spread_filter(
            window=48,
            quantile=0.85
        )

        recent_choch = (
            bars_since_choch.notna()
            & (bars_since_choch >= 0)
            & (bars_since_choch <= 36)
        )

        long_entry = (
            trade_df
            & session_quality
            & long_bias
            & spread_ok
            & (active_choch_dir == 1)
            & recent_choch
            & active_bull_lower.notna()
            & bull_retest
            & (bullish_jw | bull_disp)
        ).fillna(False)

        short_entry = (
            trade_df
            & session_quality
            & short_bias
            & spread_ok
            & (active_choch_dir == -1)
            & recent_choch
            & active_bear_upper.notna()
            & bear_retest
            & (bearish_jw | bear_disp)
        ).fillna(False)

        signals = self.build_positions_with_risk(
            long_entry=long_entry,
            short_entry=short_entry,
            trade_df=trade_df,
            session_id=session_id,
            active_bull_lower=active_bull_lower,
            active_bear_upper=active_bear_upper,
            atr_=atr_,
            max_hold_bars=30,
            cooldown_bars=8,
            stop_atr_buffer=0.25,
            reward_risk=1.25,
            one_trade_per_session=True
        )

        self.print_diagnostics(
            trade_df=trade_df,
            choch_dir=choch_dir,
            valid_bull_fvg=valid_bull_fvg,
            valid_bear_fvg=valid_bear_fvg,
            bullish_jw=bullish_jw,
            bearish_jw=bearish_jw,
            bull_retest=bull_retest,
            bear_retest=bear_retest,
            long_entry=long_entry,
            short_entry=short_entry,
            signals=signals
        )

        # # Optional hard loss cap from your inherited framework.
        # # You may comment this out if the internal TP/SL already controls risk better.
        # signals = self.cut_loss(
        #     signals=signals,
        #     df=self.df,
        #     usd=25
        # )

        return signals


if __name__ == "__main__":
    from config import session_start, session_end

    symbols = ["USOIL"]

    sim = Simulation(symbols)

    folder_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "DATA")
    )

    data = sim.add_data(
        folder_path=folder_path,
        freq=5,
        lag=1,
        method={
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "spread": "max",
            "tickvol": "sum"
        },
        # from_time="2025-07-01",
        # to_time="2025-08-18 16:50:00"
    )

    print(data.tail())

    generator = generateSignal(
        data=data,
        session_start=session_start,
        session_end=session_end
    )

    sim.positions = generator.generate_signals()
    sim.get_outputs()