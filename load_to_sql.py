from sqlalchemy import create_engine
import pandas as pd
from urllib.parse import quote_plus

# -------------------------------
# MySQL connection
# -------------------------------
username = "root"
password = quote_plus("Mathan@123")
host = "localhost"
database = "amazon_sales_analytics"

engine = create_engine(f"mysql+pymysql://{username}:{password}@{host}/{database}")

# Test connection
try:
    with engine.connect() as conn:
        print("✅ MySQL connection successful")
except Exception as e:
    print("❌ Connection failed:", e)

# -------------------------------
# Load cleaned transactions
# -------------------------------
transactions_df = pd.read_csv("data/cleaned_amazon_transactions.csv")
transactions_df.to_sql("transactions", engine, if_exists="replace", index=False)
print("✅ Transactions table loaded/replaced")

# -------------------------------
# Load products catalog
# -------------------------------
products_df = pd.read_csv("data/amazon_india_products_catalog.csv")
products_df.to_sql("products", engine, if_exists="replace", index=False)
print("✅ Products table loaded/replaced")

# -------------------------------
# Create customers table from transactions
# -------------------------------
customers_df = transactions_df[[
    "customer_id", "customer_city", "customer_state",
    "customer_tier", "customer_spending_tier", "customer_age_group"
]].drop_duplicates()
customers_df.to_sql("customers", engine, if_exists="replace", index=False)
print("✅ Customers table created/replaced")

# -------------------------------
# Create time_dimension table
# -------------------------------
time_df = transactions_df[["order_date", "order_month", "order_year", "order_quarter"]].drop_duplicates()
time_df.to_sql("time_dimension", engine, if_exists="replace", index=False)
print("✅ Time dimension table created/replaced")

print("🎉 All data successfully loaded into MySQL database.")
