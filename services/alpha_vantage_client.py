import pandas as pd
import requests

class AlphaVantageClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://www.alphavantage.co/query"

    def get_daily_data(self, symbol: str, start_date, end_date):
        """Fetch daily adjusted stock data from Alpha Vantage API"""
        try:
            params = {
                "function": "TIME_SERIES_DAILY_ADJUSTED",
                "symbol": symbol,
                "outputsize": "full",
                "apikey": self.api_key
            }
            response = requests.get(self.base_url, params=params)
            data = response.json().get("Time Series (Daily)", {})
            
            df = pd.DataFrame.from_dict(data, orient="index", dtype=float)
            df.index = pd.to_datetime(df.index)
            df = df.rename(columns={
                "1. open": "Open",
                "2. high": "High",
                "3. low": "Low",
                "4. close": "Close",
                "6. volume": "Volume"
            })
            
            df = df.sort_index()
            df = df.loc[start_date:end_date]
            return df
        except Exception as e:
            print(f"Error fetching data: {e}")
            return None
