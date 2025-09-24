# models/evaluator.py
from pyspark.ml.evaluation import RegressionEvaluator

class ModelEvaluator:
    def __init__(self):
        self.metrics = ["mae", "rmse", "r2"]

    def evaluate(self, model, data, label_col="label", prediction_col="prediction"):
        """
        Evaluasi model menggunakan MAE, RMSE, R².
        """
        results = {}
        for metric in self.metrics:
            evaluator = RegressionEvaluator(
                labelCol=label_col,
                predictionCol=prediction_col,
                metricName=metric
            )
            results[metric.upper()] = evaluator.evaluate(model.transform(data))
        return results

    def backtest(self, model, train_data, test_data, label_col="label", prediction_col="prediction"):
        """
        Backtesting: latih model di train_data, uji di test_data.
        """
        model = model.fit(train_data)
        predictions = model.transform(test_data)
        return self.evaluate(model, test_data, label_col, prediction_col)
