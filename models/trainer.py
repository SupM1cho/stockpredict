from pyspark.ml.regression import LinearRegression, RandomForestRegressor, GBTRegressor
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import SparkSession

class SparkModelTrainer:
    def __init__(self, spark: SparkSession):
        self.spark = spark

    def train(self, data, model_type="Linear Regression", train_split=0.8):
        # Convert Pandas → Spark
        sdf = self.spark.createDataFrame(data.reset_index())
        features = ["Open", "High", "Low", "Volume"]
        assembler = VectorAssembler(inputCols=features, outputCol="features")
        sdf = assembler.transform(sdf).select("features", "Close")

        # Split data
        train, test = sdf.randomSplit([train_split, 1-train_split])

        # Choose model
        if model_type == "Linear Regression":
            model = LinearRegression(featuresCol="features", labelCol="Close")
        elif model_type == "Random Forest":
            model = RandomForestRegressor(featuresCol="features", labelCol="Close")
        else:
            model = GBTRegressor(featuresCol="features", labelCol="Close")

        trained_model = model.fit(train)
        predictions = trained_model.transform(test)

        metrics = {
            "mae": trained_model.summary.meanAbsoluteError if hasattr(trained_model, "summary") else 0,
            "rmse": trained_model.summary.rootMeanSquaredError if hasattr(trained_model, "summary") else 0,
            "r2": trained_model.summary.r2 if hasattr(trained_model, "summary") else 0
        }
        return trained_model, metrics
