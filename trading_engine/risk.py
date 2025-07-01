# utils/risk.py

def calculate_sl_tp(entry_price: float, direction: str, atr: float, sl_factor: float = 1.5, tp_factor: float = 2.5):
    """
    Calculate SL/TP using ATR volatility.

    Returns:
        stop_loss, take_profit
    """
    if direction == "Long":
        stop_loss = entry_price - atr * sl_factor
        take_profit = entry_price + atr * tp_factor
    elif direction == "Short":
        stop_loss = entry_price + atr * sl_factor
        take_profit = entry_price - atr * tp_factor
    else:
        stop_loss = take_profit = None
    return stop_loss, take_profit

def calculate_sl_tp(signal, entry_price, sl_pct=0.01, tp_pct=0.02):
    if signal == 'Long':
        sl = entry_price * (1 - sl_pct)
        tp = entry_price * (1 + tp_pct)
    else:  # Short
        sl = entry_price * (1 + sl_pct)
        tp = entry_price * (1 - tp_pct)
    return sl, tp
