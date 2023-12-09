import pandas as pd
import numpy as np

############################### DETECT DECIMALS ######################################
def detect_max_decimals(data):
    max_decimals = 0
    for value in data:
        decimal_part = str(value).split('.')[-1]
        max_decimals = max(max_decimals, len(decimal_part))
    return max_decimals

############################### MOVING AVERAGES ######################################
def ma(data, window, ma_type=0):
    """
    Calcula diferentes tipos de medias móviles dependiendo del parámetro ma_type.

    :param data: Array de precios o valores.
    :param window: Tamaño de la ventana (número de periodos).
    :param ma_type: Tipo de media móvil (0 = SMA, 1 = EMA, 2 = SMMA, 3 = LWMA).
    :return: Array con los valores de la media móvil.
    """
    if len(data) < window:
        return np.array([])

    if ma_type == 0:
        # SMA
        return simple_moving_average(data, window)
    elif ma_type == 1:
        # EMA
        return exponential_moving_average(data, window)
    elif ma_type == 2:
        # SMMA
        return smoothed_moving_average(data, window)
    elif ma_type == 3:
        # LWMA
        return linear_weighted_moving_average(data, window)
    else:
        raise ValueError("Invalid ma_type. Must be 0, 1, 2, or 3.")

def simple_moving_average(data, window):
    num_decimals = detect_max_decimals(data)
    sma = np.full(len(data), np.nan)
    sma_values = np.convolve(data, np.ones(window), 'valid') / window
    sma[window - 1:] = np.round(sma_values, num_decimals)
    return sma

def exponential_moving_average(data, window):
    num_decimals = detect_max_decimals(data)
    ema = np.full(len(data), np.nan)
    ema[window - 1] = np.mean(data[:window])
    alpha = 2 / (window + 1)
    for i in range(window, len(data)):
        ema[i] = np.round(data[i] * alpha + ema[i - 1] * (1 - alpha), num_decimals)
    return ema

def smoothed_moving_average(data, window):
    num_decimals = detect_max_decimals(data)
    smma = np.full(len(data), np.nan)
    smma[window - 1] = np.mean(data[:window])
    for i in range(window, len(data)):
        smma[i] = np.round((smma[i - 1] * (window - 1) + data[i]) / window, num_decimals)
    return smma


def linear_weighted_moving_average(data, window):
    num_decimals = detect_max_decimals(data)
    lwma = np.full(len(data), np.nan)
    weights = np.arange(1, window + 1)
    for i in range(window - 1, len(data)):
        lwma[i] = np.round(np.dot(data[i - window + 1:i + 1], weights) / weights.sum(), num_decimals)
    return lwma

############################### WILLIAMS PERCENT RANGE ######################################
def wpr(high, low, close, period=14):
    """
    Calcula el Williams %R y redondea los valores a enteros.

    :param high: Array de precios máximos.
    :param low: Array de precios mínimos.
    :param close: Array de precios de cierre.
    :param period: Número de periodos a considerar.
    :return: Array con los valores de Williams %R redondeados a enteros.
    """
    will_r = np.full(len(close), np.nan)

    for i in range(period - 1, len(close)):
        highest_high = np.max(high[i - period + 1:i + 1])
        lowest_low = np.min(low[i - period + 1:i + 1])
        will_r[i] = round(((highest_high - close[i]) / (highest_high - lowest_low)) * -100)

    return will_r

############################### RELATIVE STRENGTH INDEX ######################################
def rsi(close, period=14):
    # Inicializar los buffers de ganancias y pérdidas
    gain_buffer = np.zeros(len(close))
    loss_buffer = np.zeros(len(close))

    # Calcular las ganancias y pérdidas iniciales
    for i in range(1, period + 1):
        change = close[i] - close[i - 1]
        gain_buffer[i] = max(change, 0)
        loss_buffer[i] = abs(min(change, 0))

    # Calcular las medias de ganancias y pérdidas
    avg_gain = np.mean(gain_buffer[1:period + 1])
    avg_loss = np.mean(loss_buffer[1:period + 1])

    # Calcular el RSI
    rsi = np.zeros(len(close))
    for i in range(period + 1, len(close)):
        change = close[i] - close[i - 1]
        gain = max(change, 0)
        loss = abs(min(change, 0))

        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period

        if avg_loss != 0:
            rs = avg_gain / avg_loss
            rsi[i] = 100 - (100 / (1 + rs))
        else:
            rsi[i] = 100 if avg_gain != 0 else 50

    # Redondear los valores del RSI al entero más cercano
    rsi_rounded = np.round(rsi).astype(int)

    return rsi_rounded

############################### AVERAGE TRUE RANGE ######################################
def true_range(high, low, close):
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
    return tr

def atr(high, low, close, period=14):
    tr = true_range(high, low, close)
    atr = tr.rolling(window=period).mean()

    # Redondear los valores del ATR
    num_decimals = detect_max_decimals(close)
    atr_rounded = atr.round(num_decimals)

    return atr_rounded

############################### AVERAGE DIRECTIONAL INDEX ######################################
def adx(high, low, close, period=14):
    # Calcular +DM, -DM y True Range
    plus_dm = high.diff()
    minus_dm = low.diff()
    tr = np.maximum.reduce([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()])

    plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0.0)
    minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0.0)

    # Calcular el promedio de +DM, -DM y True Range
    tr_rolling = pd.Series(tr).rolling(window=period).mean()
    plus_dm_rolling = pd.Series(plus_dm).rolling(window=period).mean()
    minus_dm_rolling = pd.Series(minus_dm).rolling(window=period).mean()

    # Calcular +DI y -DI
    plus_di = 100 * plus_dm_rolling / tr_rolling
    minus_di = 100 * minus_dm_rolling / tr_rolling

    # Calcular el ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.rolling(window=period).mean()

    # Redondear los valores a un decimal
    plus_di_rounded = plus_di.round(1)
    minus_di_rounded = minus_di.round(1)
    adx_rounded = adx.round(1)

    return plus_di_rounded, minus_di_rounded, adx_rounded
