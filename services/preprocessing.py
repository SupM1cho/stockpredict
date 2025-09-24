class DataPreprocessor:
    def prepare_features(self, df):
        """Basic preprocessing (add returns, log, etc.)"""
        df["Return"] = df["Close"].pct_change()
        df = df.dropna()
        return df
