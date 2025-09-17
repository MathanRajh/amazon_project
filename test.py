import pandas as pd
import numpy as np
from datetime import datetime

# -------------------------------
# Load the Dataset
# -------------------------------

file_path = r"data\amazon_india_products_catalog.csv"
df = pd.read_csv(file_path)

print(df.columns.tolist())


