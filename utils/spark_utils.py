from pyspark.sql import SparkSession

class SparkSessionManager:
    def __init__(self, app_name="StockPredictionApp"):
        self.app_name = app_name
        self.spark = None

    def get_session(self):
        if self.spark is None:
            self.spark = SparkSession.builder \
                .appName(self.app_name) \
                .getOrCreate()
        return self.spark
