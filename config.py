from datetime import time

margin = {'USOIL': 5, 'XAUUSD': 1, 'BTCUSD': 0.0025}

session_start = {
    "London": time(8, 0),     # 08:00 UTC
    "New York": time(13, 0),  # 13:00 UTC
    "Sydney": time(22, 0),    # 22:00 UTC (previous day)
    "Tokyo": time(23, 0)      # 23:00 UTC (previous day)
}

session_end = {
    "London": time(16, 0),    # 16:00 UTC
    "New York": time(21, 0),  # 21:00 UTC
    "Sydney": time(6, 0),     # 06:00 UTC (next day)
    "Tokyo": time(7, 0)       # 07:00 UTC (next day)
}

trading_days = {'USOIL': 252, 'XAUUSD': 252, 'BTCUSD': 365}