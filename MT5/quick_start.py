import time
from datetime import datetime, timezone
import pandas as pd
import MetaTrader5 as MT5

# ---- CONFIG: set these if you want to login explicitly via code ----
ACCOUNT_ID   = None            # e.g. 12345678  or leave None to use already-logged-in terminal
ACCOUNT_PWD  = None            # e.g. "password" (only if logging in via code)
ACCOUNT_SVR  = None            # e.g. "Exness-MT5Real" / "Exness-MT5Trial" (varies by account)
MT5_PATH     = None            # e.g. r"C:\Program Files\MetaTrader 5\terminal64.exe" (Windows only)
SYMBOL       = "XAUUSD"        # change to what exists on your Exness account
TIMEFRAME    = MT5.TIMEFRAME_M1

def init_mt5():
    # Initialize; if MT5 is running and logged in, a bare initialize() usually works.
    if MT5_PATH:
        ok = MT5.initialize(MT5_PATH)
    else:
        ok = MT5.initialize()
    if not ok:
        raise RuntimeError(f"MT5.initialize() failed: {MT5.last_error()}")

    # Optional explicit login (if your terminal is not logged in)
    if ACCOUNT_ID and ACCOUNT_PWD and ACCOUNT_SVR:
        if not MT5.login(login=ACCOUNT_ID, password=ACCOUNT_PWD, server=ACCOUNT_SVR):
            raise RuntimeError(f"MT5.login() failed: {MT5.last_error()}")

    # Ensure symbol is available/visible in Market Watch
    if not MT5.symbol_select(SYMBOL, True):
        raise RuntimeError(f"symbol_select failed for {SYMBOL}: {MT5.last_error()}")

def shutdown_mt5():
    MT5.shutdown()

init_mt5()

# --- Get a single latest tick (quote) ---
tick = MT5.symbol_info_tick(SYMBOL)
print("Latest tick:", tick)

# --- Get last N minutes OHLCV (here: 100 M1 bars) ---
rates = MT5.copy_rates_from_pos(SYMBOL, TIMEFRAME, 0, 100)
df = pd.DataFrame(rates)
if not df.empty:
    df['time'] = pd.to_datetime(df['time'], unit='s', utc=True).dt.tz_convert('UTC')
print(df.tail())

shutdown_mt5()
