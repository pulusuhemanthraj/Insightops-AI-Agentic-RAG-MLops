"""PySpark-ready ETL module.

This module is intentionally lightweight so it can run in portfolio environments.
If PySpark is installed, it reads raw CSV files and writes parquet features.
"""
from src.utils.config import RAW_DIR, PROCESSED_DIR


def run_spark_etl() -> None:
    try:
        from pyspark.sql import SparkSession
        from pyspark.sql.functions import col, count, avg
    except ImportError as exc:
        raise RuntimeError("Install pyspark to run Spark ETL: pip install pyspark") from exc

    spark = SparkSession.builder.appName("InsightOpsSparkETL").getOrCreate()
    events = spark.read.option("header", True).csv(str(RAW_DIR / "product_events.csv"))
    metrics = events.groupBy("product_area", "event_name").agg(
        count("event_id").alias("event_count"),
        avg(col("latency_ms").cast("double")).alias("avg_latency_ms"),
    )
    metrics.write.mode("overwrite").parquet(str(PROCESSED_DIR / "spark_event_metrics.parquet"))
    spark.stop()


if __name__ == "__main__":
    run_spark_etl()
