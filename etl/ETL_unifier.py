import pandas as pd
import os

DATA_DIR = "data/raw"
OUTPUT_DIR = "data/clean"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Simulacion: combinar restaurant_sales + reviews + empleados
df_sales = pd.read_csv(os.path.join(DATA_DIR, "restaurant_sales.csv"))
df_reviews = pd.read_csv(os.path.join(DATA_DIR, "reviews.csv"))
df_employees = pd.read_csv(os.path.join(DATA_DIR, "store_employees.csv"))

df_sales["date"] = pd.to_datetime(df_sales["date"])
df_reviews["date"] = pd.to_datetime(df_reviews["date"])
df_employees["hire_date"] = pd.to_datetime(df_employees["hire_date"])

# KPI por tienda
sales_summary = df_sales.groupby("store_id")["daily_sales"].agg(["mean", "sum"]).reset_index()
sales_summary.columns = ["store_id", "avg_daily_sales", "total_month_sales"]
review_scores = df_reviews.groupby("store_id")["rating"].mean().reset_index()
review_scores.columns = ["store_id", "avg_customer_rating"]
employee_counts = df_employees.groupby("store_id")["employee_id"].count().reset_index()
employee_counts.columns = ["store_id", "num_employees"]

store_metrics = df_sales[["store_id", "brand"]].drop_duplicates() \
    .merge(sales_summary, on="store_id", how="left") \
    .merge(review_scores, on="store_id", how="left") \
    .merge(employee_counts, on="store_id", how="left")

store_metrics.fillna({"avg_customer_rating": 0, "num_employees": 0}, inplace=True)
store_metrics.to_csv(os.path.join(OUTPUT_DIR, "unified_store_metrics.csv"), index=False)
print(" ETL terminado")
