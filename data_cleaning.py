import pandas as pd
import numpy as np
from datetime import datetime
import os

# -------------------------------
# Load the Dataset
# -------------------------------
file_path = r"data/amazon_india_complete_2015_2025.csv"
df = pd.read_csv(file_path)

# -------------------------------
# Q1: Standardize Dates (order_date)
# -------------------------------
def parse_date(x):
    """Try multiple date formats safely."""
    date_formats = [
    "%d/%m/%Y",   
    "%d-%m-%Y",   
    "%Y/%m/%d",   
    "%Y-%m-%d",   
    "%m/%d/%Y",   
    "%m-%d-%Y",   
    "%d %b %Y",   
    "%d %B %Y",   
    "%b %d, %Y",  
    "%B %d, %Y",  
    "%d.%m.%Y",   
    "%Y.%m.%d",   
    "%d-%b-%Y",   
    "%d-%B-%Y",   
    "%Y%m%d",     
]

    for fmt in date_formats:
        try:
            return datetime.strptime(str(x), fmt)
        except (ValueError, TypeError):
            continue
    return pd.NaT

df["order_date"] = df["order_date"].apply(parse_date)
df["order_date"] = df["order_date"].dt.strftime("%Y-%m-%d")


# -------------------------------
# Q2: Clean Price Column (original_price_inr)
# -------------------------------
def clean_price(x):
    if pd.isna(x):
        return np.nan
    x = str(x).replace("₹", "").replace(",", "").strip()
    if x.lower() in ["price on request", "na", "nan", "none", ""]:
        return np.nan
    try:
        return float(x)
    except ValueError:
        return np.nan

df["original_price_inr"] = df["original_price_inr"].apply(clean_price)

# -------------------------------
# Q3: Standardize Ratings (customer_rating & product_rating)
# -------------------------------
def clean_rating(x):
    if pd.isna(x):
        return np.nan
    x = str(x).lower().strip()
    if "star" in x:
        return float(x.split()[0])  # "4 stars" → 4.0
    if "/" in x:
        try:
            num, denom = x.split("/")
            return (float(num) / float(denom)) * 5  # e.g. 3/5 → 3.0
        except:
            return np.nan
    try:
        return float(x)
    except ValueError:
        return np.nan

df["customer_rating"] = df["customer_rating"].apply(clean_rating)
df["product_rating"] = df["product_rating"].apply(clean_rating)

# -------------------------------
# Q4: Standardize City Names (customer_city)
# -------------------------------
city_mapping = {
    "Bengaluru": "Bangalore",
    "New Delhi": "Delhi",
    "Bombay": "Mumbai",
}

df["customer_city"] = df["customer_city"].replace(city_mapping)
df["customer_city"] = df["customer_city"].str.title()

# -------------------------------
# Q5: Standardize Boolean Columns
# -------------------------------
def clean_bool(x):
    if pd.isna(x):
        return False
    x = str(x).strip().lower()
    return x in ["yes", "true", "1", "y"]

bool_columns = ["is_prime_member", "is_festival_sale", "is_prime_eligible"]
for col in bool_columns:
    df[col] = df[col].apply(clean_bool)

# -------------------------------
# Q6: Standardize Product Categories (category)
# -------------------------------
category_mapping = {
    "Electronicss": "Electronics",
    "Electronic": "Electronics",
    "ELECTRONICS": "Electronics",
    "Electronics & Accessories": "Electronics"
}

df["category"] = df["category"].replace(category_mapping)
df["category"] = df["category"].str.title()

# -------------------------------
# Q7: Clean Delivery Days (delivery_days)
# -------------------------------
def clean_delivery(x):
    if pd.isna(x):
        return np.nan
    x = str(x).lower().strip()
    if "same" in x:
        return 0
    try:
        val = int(x)
        if 0 <= val <= 30:  # realistic delivery window
            return val
        return np.nan
    except:
        return np.nan

df["delivery_days"] = df["delivery_days"].apply(clean_delivery)

# -------------------------------
# Q8: Handle Duplicates
# -------------------------------
# Using transaction_id + product_id + order_date to identify dupes
df = df[~df.duplicated(subset=["transaction_id", "product_id", "order_date"], keep="first")]

# -------------------------------
# Q9: Correct Price Outliers
# -------------------------------
Q1 = df["original_price_inr"].quantile(0.25)
Q3 = df["original_price_inr"].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

df.loc[(df["original_price_inr"] < lower_bound) | (df["original_price_inr"] > upper_bound), 
       "original_price_inr"] = np.nan

# -------------------------------
# Q10: Standardize Payment Methods (payment_method)
# -------------------------------
payment_mapping = {
    "UPI": "UPI",
    "PhonePe": "UPI",
    "Google Pay": "UPI",
    "Credit Card": "Credit Card",
    "CC": "Credit Card",
    "Debit Card": "Debit Card",
    "Cash On Delivery": "Cash",
    "COD": "Cash",
}

df["payment_method"] = df["payment_method"].replace(payment_mapping)
df["payment_method"] = df["payment_method"].str.title()

# -------------------------------
# Save the Cleaned Data
# -------------------------------
output_path = r"data/cleaned_amazon_transactions.csv"
os.makedirs("data", exist_ok=True)  # ensure 'data' folder exists
df.to_csv(output_path, index=False)

print(f"✅ Data cleaning complete. Saved as {output_path}")
