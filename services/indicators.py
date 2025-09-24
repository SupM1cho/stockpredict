import pandas as pd

class TechnicalIndicators:
    def add_moving_averages(self, df):
        df["MA_20"] = df["Close"].rolling(window=20).mean()
        df["MA_50"] = df["Close"].rolling(window=50).mean()
        return df
    
    def add_rsi(self, df, window=14):
        delta = df["Close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
        rs = gain / loss
        df["RSI"] = 100 - (100 / (1 + rs))
        return df
    
    def add_macd(self, df, short=12, long=26, signal=9):
        df["EMA12"] = df["Close"].ewm(span=short).mean()
        df["EMA26"] = df["Close"].ewm(span=long).mean()
        df["MACD"] = df["EMA12"] - df["EMA26"]
        df["MACD_Signal"] = df["MACD"].ewm(span=signal).mean()
        return df
    
    def add_bollinger_bands(self, df, window=20):
        sma = df["Close"].rolling(window=window).mean()
        std = df["Close"].rolling(window=window).std()
        df["BB_Upper"] = sma + (2 * std)
        df["BB_Lower"] = sma - (2 * std)
        return df
