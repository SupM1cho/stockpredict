import pandas as pd

class StockPredictor:
    def __init__(self, model):
        self.model = model

    def predict_future(self, data, days=7):
        last_date = data.index[-1]
        future_dates = pd.date_range(start=last_date, periods=days+1, freq="B")[1:]
        
        # Dummy implementation (replace with real Spark predictions)
        last_price = data["Close"].iloc[-1]
        predictions = [last_price * (1 + 0.01*i) for i in range(days)]
        
        return pd.DataFrame({"Date": future_dates, "Predicted": predictions})
