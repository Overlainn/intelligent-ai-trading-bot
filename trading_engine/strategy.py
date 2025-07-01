# utils/strategy.py

def should_enter_trade(prediction: str, confidence: float, ema_crossover: bool, vwap_crossover: bool, position_open: bool) -> str:
    """
    Decide if we should enter a trade based on model signal and indicator logic.

    Returns:
        'Long', 'Short', or None
    """
    if position_open:
        return None

    # Confirm signals with crossover
    if prediction == "Long" and confidence >= 0.6 and ema_crossover and vwap_crossover:
        return "Long"
    elif prediction == "Short" and confidence >= 0.6 and ema_crossover and vwap_crossover:
        return "Short"
    return None

def should_enter_trade(signal, current_position):
    if current_position is None:
        return True
    if current_position['side'] != signal:
        return True  # flip direction
    return False  # already in same position
