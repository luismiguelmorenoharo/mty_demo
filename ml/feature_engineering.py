import pandas as pd
import os

INPUT = "data/clean/unified_store_metrics.csv"
OUTPUT = "data/features/final_dataset.csv"

df = pd.read_csv(INPUT)
df["sales_per_employee"] = df["total_month_sales"] / df["num_employees"].replace(0, 1)
df["rating_gap"] = (4.5 - df["avg_customer_rating"]).clip(lower=0)
threshold = df["total_month_sales"].quantile(0.25)
df["is_low_performing_store"] = (df["total_month_sales"] < threshold).astype(int)

os.makedirs("data/features", exist_ok=True)
df.to_csv(OUTPUT, index=False)
print("✅ Feature engineering listo")
