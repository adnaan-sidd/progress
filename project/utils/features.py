import pandas as pd

def calculate_technical_indicators(df):
    """Calculate technical indicators"""
    # RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # MACD
    ema12 = df['close'].ewm(span=12).mean()
    ema26 = df['close'].ewm(span=26).mean()
    df['macd'] = ema12 - ema26
    
    # Bollinger Bands
    df['sma20'] = df['close'].rolling(20).mean()
    std = df['close'].rolling(20).std()
    df['upper_bb'] = df['sma20'] + 2*std
    df['lower_bb'] = df['sma20'] - 2*std
    
    return df

def feature_engineering(df):
    """Feature selection for the model"""
    return df[[
        'sma20', 
        'rsi', 
        'macd', 
        'upper_bb', 
        'lower_bb'
    ]]
