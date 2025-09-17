# streamlit_dashboard_30q.py
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
from urllib.parse import quote_plus
import numpy as np

st.set_page_config(page_title="Amazon Analytics Dashboard", layout="wide")

# ---------------- DB CONNECT ----------------
DB_USER = "root"
DB_PASS = quote_plus("Mathan@123")
DB_HOST = "localhost"
DB_NAME = "amazon_sales_analytics"

engine = create_engine(f"mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}/{DB_NAME}")

@st.cache_data
def load_data():
    transactions = pd.read_sql("SELECT * FROM transactions", engine, parse_dates=["order_date"])
    customers = pd.read_sql("SELECT * FROM customers", engine)
    products = pd.read_sql("SELECT * FROM products", engine)
    
    df = transactions.merge(customers, on="customer_id", how="left") \
                     .merge(products, on="product_id", how="left")
    
    # Clean categories and brands (use whichever column exists)
    # prefer merged columns with suffixes you had earlier
    cat_col = None
    for c in ("category_x", "category", "category_y"):
        if c in df.columns:
            cat_col = c
            break
    brand_col = None
    for b in ("brand_x", "brand", "brand_y"):
        if b in df.columns:
            brand_col = b
            break
    age_col = None
    for a in ("customer_age_group_x", "customer_age_group", "customer_age_group_y"):
        if a in df.columns:
            age_col = a
            break

    df['category_clean'] = df[cat_col].astype(str).str.strip().str.title() if cat_col else ""
    df['brand_clean'] = df[brand_col].astype(str).str.strip().str.title() if brand_col else ""
    df['age_group_clean'] = df[age_col].astype(str).str.strip().str.title() if age_col else ""
    
    # Unit price calc (avoid divide by zero)
    if 'discounted_price_inr' in df.columns and 'quantity' in df.columns:
        df['unit_price_calc'] = df.apply(lambda r: r['discounted_price_inr']/r['quantity'] if pd.notna(r['discounted_price_inr']) and r['quantity'] and r['quantity']>0 else np.nan, axis=1)
    else:
        df['unit_price_calc'] = np.nan
    
    return df

df = load_data()

# ---------------- HELPER FUNCTION ----------------
def plot_bar(series, title, max_items=10):
    series = series.sort_values(ascending=False).head(max_items)
    st.write(title)
    st.bar_chart(series)

def plot_line(series, title):
    st.write(title)
    st.line_chart(series)

def plot_box(df_, x, y, title):
    fig, ax = plt.subplots(figsize=(12,6))
    sns.boxplot(data=df_, x=x, y=y, ax=ax)
    plt.xticks(rotation=90)
    plt.title(title)
    st.pyplot(fig)

def safe_metric(label, value):
    try:
        st.metric(label, value)
    except Exception:
        st.write(f"{label}: {value}")

# ---------------- QUESTIONS ----------------

def q1(): # Executive Summary
    st.subheader("Q1: Executive Summary")
    total_rev = df['final_amount_inr'].sum() if 'final_amount_inr' in df.columns else 0
    active_customers = df['customer_id'].nunique() if 'customer_id' in df.columns else 0
    aov = df['final_amount_inr'].mean() if 'final_amount_inr' in df.columns else 0
    safe_metric("Total Revenue", f"{total_rev:,.0f}")
    safe_metric("Active Customers", f"{active_customers:,}")
    safe_metric("Avg Order Value", f"{aov:,.2f}")
    if 'category_clean' in df.columns and 'final_amount_inr' in df.columns:
        plot_bar(df.groupby('category_clean')['final_amount_inr'].sum(), "Top Categories by Revenue")
    else:
        st.write("Category or revenue column missing.")

def q2(): # Real-time Performance
    st.subheader("Q2: Real-time Business Performance")
    if 'order_date' not in df.columns or 'final_amount_inr' not in df.columns:
        st.write("Required columns missing.")
        return
    month = df['order_date'].dt.to_period('M').max()
    df_month = df[df['order_date'].dt.to_period('M')==month]
    safe_metric("Revenue This Month", f"{df_month['final_amount_inr'].sum():,.0f}")
    safe_metric("Orders This Month", f"{len(df_month):,}")
    safe_metric("Active Customers (Month)", f"{df_month['customer_id'].nunique():,}")

def q3(): # Strategic Overview (simple market share by category)
    st.subheader("Q3: Strategic Overview - Market Share by Category")
    if 'category_clean' in df.columns and 'final_amount_inr' in df.columns:
        cat = df.groupby('category_clean')['final_amount_inr'].sum().sort_values(ascending=False)
        st.write("Top categories market share")
        st.bar_chart((cat / cat.sum()).head(10))
    else:
        st.write("category or revenue missing")

def q4(): # Financial Performance (simple revenue breakdown)
    st.subheader("Q4: Financial Performance - Revenue by Category & Avg Price")
    if 'category_clean' in df.columns and 'final_amount_inr' in df.columns:
        rev = df.groupby('category_clean')['final_amount_inr'].sum().sort_values(ascending=False)
        st.write("Revenue by category")
        st.bar_chart(rev.head(10))
    else:
        st.write("category or revenue missing")
    if 'category_clean' in df.columns and 'unit_price_calc' in df.columns:
        avg_price = df.groupby('category_clean')['unit_price_calc'].mean().sort_values(ascending=False)
        st.write("Average unit price by category")
        st.bar_chart(avg_price.head(10))

def q5(): # Growth Analytics - simple YoY revenue growth and top categories
    st.subheader("Q5: Growth Analytics - YoY Revenue & Top Categories")
    if 'order_year' in df.columns and 'final_amount_inr' in df.columns:
        yearly = df.groupby('order_year')['final_amount_inr'].sum().sort_index()
        plot_line(yearly, "Yearly Revenue")
        pct = yearly.pct_change().fillna(0) * 100
        st.write("Year-over-year growth (%)")
        st.bar_chart(pct)
    else:
        st.write("order_year or revenue missing")
    # top categories
    if 'category_clean' in df.columns and 'final_amount_inr' in df.columns:
        plot_bar(df.groupby('category_clean')['final_amount_inr'].sum(), "Top Categories (total revenue)")
    else:
        st.write("Category or revenue missing")

def q6(): # Revenue Trend
    st.subheader("Q6: Revenue Trend (monthly heatmap simplified)")
    if 'order_year' in df.columns and 'order_month' in df.columns and 'final_amount_inr' in df.columns:
        monthly = df.groupby(['order_year','order_month'])['final_amount_inr'].sum().reset_index()
        pivot = monthly.pivot(index='order_year', columns='order_month', values='final_amount_inr').fillna(0)
        st.write("Monthly revenue pivot (rows=year, cols=month)")
        st.dataframe(pivot)
    else:
        st.write("Required columns missing")

def q7(): # Category Performance (simple)
    st.subheader("Q7: Category Performance - Revenue & Market Share")
    if 'category_clean' in df.columns and 'final_amount_inr' in df.columns:
        cat = df.groupby('category_clean')['final_amount_inr'].sum().sort_values(ascending=False)
        st.write("Revenue by category (top 10)")
        st.bar_chart(cat.head(10))
        st.write("Market share - top categories")
        st.bar_chart((cat / cat.sum()).head(10))
    else:
        st.write("Category or revenue missing")

def q8(): # Geographic revenue analysis simplified by state
    st.subheader("Q8: Geographic Revenue (Top states)")
    state_col = None
    for c in ("customer_state_x","customer_state","customer_state_y"):
        if c in df.columns:
            state_col = c
            break
    if state_col and 'final_amount_inr' in df.columns:
        state_rev = df.groupby(state_col)['final_amount_inr'].sum().sort_values(ascending=False)
        st.write("Top states by revenue")
        st.bar_chart(state_rev.head(20))
    else:
        st.write("customer_state or revenue missing")

def q9(): # Festival Sales
    st.subheader("Q9: Festival Sales - simple compare")
    if 'is_festival_sale' in df.columns and 'final_amount_inr' in df.columns:
        plot_bar(df.groupby('is_festival_sale')['final_amount_inr'].sum(), "Festival vs Non-Festival Sales")
    else:
        st.write("is_festival_sale or revenue missing")

def q10(): # Price Optimization - price elasticity simplified
    st.subheader("Q10: Price vs Demand (scatter)")
    if 'discounted_price_inr' in df.columns and 'quantity' in df.columns:
        sample = df.dropna(subset=['discounted_price_inr','quantity']).sample(min(2000, len(df)), random_state=1) if len(df)>2000 else df.dropna(subset=['discounted_price_inr','quantity'])
        fig, ax = plt.subplots(figsize=(10,6))
        ax.scatter(sample['discounted_price_inr'], sample['quantity'], alpha=0.4)
        ax.set_xlabel("Discounted Price (INR)")
        ax.set_ylabel("Quantity")
        ax.set_title("Price vs Demand (scatter)")
        st.pyplot(fig)
        corr = df[['discounted_price_inr','quantity']].corr().iloc[0,1]
        st.write(f"Correlation (price vs quantity): {corr:.3f}")
    else:
        st.write("discounted_price_inr or quantity missing")

def q11(): # RFM Dashboard (simple summary)
    st.subheader("Q11: RFM - simple scatter")
    if 'order_date' not in df.columns or 'transaction_id' not in df.columns or 'final_amount_inr' not in df.columns:
        st.write("Required columns missing")
        return
    rfm = df.groupby("customer_id").agg({
        "order_date":"max",
        "transaction_id":"count",
        "final_amount_inr":"sum"
    }).reset_index().rename(columns={"order_date":"last_purchase","transaction_id":"frequency","final_amount_inr":"monetary"})
    rfm['recency'] = (df['order_date'].max() - pd.to_datetime(rfm['last_purchase'])).dt.days
    fig, ax = plt.subplots(figsize=(10,6))
    scatter = ax.scatter(rfm['recency'], rfm['monetary'], s=np.sqrt(rfm['frequency']), alpha=0.5)
    ax.set_xlabel("Recency (days)")
    ax.set_ylabel("Monetary (total)")
    ax.set_title("RFM scatter (recency vs monetary, size~frequency)")
    st.pyplot(fig)

def q12(): # Customer Journey (first vs repeat simple)
    st.subheader("Q12: Customer Journey - first vs repeat orders")
    if 'customer_id' in df.columns and 'order_date' in df.columns:
        first = df.groupby('customer_id')['order_date'].min().reset_index().rename(columns={'order_date':'first_order'})
        orders = df.groupby('customer_id').size().reset_index(name='orders')
        merged = first.merge(orders, on='customer_id')
        st.write("Distribution of number of orders per customer")
        st.bar_chart(merged['orders'].value_counts().sort_index())
    else:
        st.write("Required columns missing")

def q13(): # Prime membership analysis (already present earlier)
    st.subheader("Q13: Prime Membership - simple comparison")
    if 'is_prime_member' in df.columns and 'final_amount_inr' in df.columns:
        mean_by_prime = df.groupby('is_prime_member')['final_amount_inr'].mean()
        st.write("Avg order value by Prime membership (0=No,1=Yes)")
        st.bar_chart(mean_by_prime)
    else:
        st.write("Required columns missing")

def q14(): # Retention & Cohorts (simple)
    st.subheader("Q14: Cohort retention (simplified)")
    if 'customer_id' not in df.columns or 'order_date' not in df.columns:
        st.write("Required columns missing")
        return
    df_cohort = df.copy()
    df_cohort['order_period'] = df_cohort['order_date'].dt.to_period('M')
    first_order = df_cohort.groupby('customer_id')['order_date'].min().dt.to_period('M').rename('cohort')
    df_cohort = df_cohort.join(first_order, on='customer_id')
    cohort_counts = df_cohort.groupby(['cohort','order_period'])['customer_id'].nunique().unstack(fill_value=0)
    st.write("Cohort counts (rows=cohort month)")
    st.dataframe(cohort_counts.head(12))

def q15(): # Demographics & Behavior simplified
    st.subheader("Q15: Demographics & Behavior")
    if 'age_group_clean' in df.columns and 'final_amount_inr' in df.columns:
        plot_bar(df.groupby('age_group_clean')['final_amount_inr'].sum(), "Revenue by Age Group")
    else:
        st.write("Age group or revenue missing")

def q16(): # Product Performance - top products
    st.subheader("Q16: Product Performance - Top Products")
    prod_col = None
    for c in ("product_name_x","product_name","product_name_y"):
        if c in df.columns:
            prod_col = c
            break
    if prod_col and 'final_amount_inr' in df.columns:
        top = df.groupby(prod_col)['final_amount_inr'].sum().sort_values(ascending=False)
        st.write("Top products by revenue")
        st.bar_chart(top.head(20))
    else:
        st.write("product name or revenue missing")

def q17(): # Brand Analytics - simplified
    st.subheader("Q17: Brand Analytics")
    if 'brand_clean' in df.columns and 'final_amount_inr' in df.columns:
        plot_bar(df.groupby('brand_clean')['final_amount_inr'].sum(), "Top Brands by Revenue", max_items=20)
    else:
        st.write("Brand or revenue missing")

def q18(): # Inventory optimization simple: launch_year vs revenue
    st.subheader("Q18: Inventory & Product Lifecycle (launch year)")
    if 'launch_year' in df.columns and 'final_amount_inr' in df.columns:
        life = df.groupby('launch_year')['final_amount_inr'].sum().sort_index()
        plot_line(life, "Revenue by Launch Year")
    else:
        st.write("launch_year or revenue missing")

def q19(): # Product Rating & Review (simple)
    st.subheader("Q19: Ratings impact on sales")
    rating_col = None
    for c in ('rating','customer_rating','product_rating'):
        if c in df.columns:
            rating_col = c
            break
    if rating_col and 'final_amount_inr' in df.columns:
        avg = df.groupby(rating_col)['final_amount_inr'].sum().sort_index()
        plot_line(avg, f"Revenue by {rating_col}")
    else:
        st.write("Rating or revenue missing")

def q20(): # New Product Launch Dashboard (simple)
    st.subheader("Q20: New Product Launch - top recent launches")
    if 'launch_year' in df.columns and 'final_amount_inr' in df.columns:
        recent = df[df['launch_year']>= (df['launch_year'].max()-2)].groupby('product_id')['final_amount_inr'].sum().sort_values(ascending=False)
        st.write("Top recent launches by revenue")
        st.bar_chart(recent.head(20))
    else:
        st.write("launch_year or revenue missing")

def q21(): # Delivery performance (simple)
    st.subheader("Q21: Delivery Performance")
    if 'delivery_days' in df.columns:
        st.write("Delivery days distribution")
        st.bar_chart(df['delivery_days'].dropna().value_counts().sort_index().head(50))
    else:
        st.write("delivery_days missing")

def q22(): # Payment analytics
    st.subheader("Q22: Payment Methods")
    if 'payment_method' in df.columns:
        st.write("Payment method share")
        st.bar_chart(df['payment_method'].value_counts().head(20))
    else:
        st.write("payment_method missing")

def q23(): # Returns & cancellations
    st.subheader("Q23: Returns & Cancellations")
    if 'return_status' in df.columns:
        st.write("Return status counts")
        st.bar_chart(df['return_status'].value_counts().head(20))
    else:
        st.write("return_status missing")

def q24(): # Customer Service (placeholder simple metric)
    st.subheader("Q24: Customer Service - simple metrics")
    st.write("No direct customer service columns available; show proxies if present.")
    if 'customer_rating' in df.columns:
        st.write("Average customer rating:")
        st.write(df['customer_rating'].mean())

def q25(): # Supply chain (placeholder simple)
    st.subheader("Q25: Supply Chain - placeholder")
    st.write("No supplier-level data available in the dataset.")

def q26(): # Predictive Analytics (placeholder)
    st.subheader("Q26: Predictive Analytics - placeholder")
    st.write("Modeling not implemented in this simple dashboard. Use historical revenue for naive forecast.")
    if 'order_date' in df.columns and 'final_amount_inr' in df.columns:
        monthly = df.set_index('order_date').resample('M')['final_amount_inr'].sum()
        st.line_chart(monthly.tail(36))

def q27(): # Market intelligence (simple)
    st.subheader("Q27: Market Intelligence - top categories/brands")
    if 'category_clean' in df.columns:
        st.write("Top categories")
        st.bar_chart(df['category_clean'].value_counts().head(20))
    else:
        st.write("category missing")

def q28(): # Cross-sell / association (very simple)
    st.subheader("Q28: Cross-sell - simple co-purchase proxy")
    if 'transaction_id' in df.columns and 'product_id' in df.columns:
        # simple: products per transaction distribution
        items_per_txn = df.groupby('transaction_id')['product_id'].nunique()
        st.write("Distribution: number of distinct products per transaction")
        st.bar_chart(items_per_txn.value_counts().sort_index())
    else:
        st.write("transaction_id or product_id missing")

def q29(): # Seasonal planning (simple seasonal totals)
    st.subheader("Q29: Seasonal Planning - monthly totals")
    if 'order_month' in df.columns and 'final_amount_inr' in df.columns:
        month_totals = df.groupby('order_month')['final_amount_inr'].sum().sort_index()
        st.bar_chart(month_totals)
    else:
        st.write("order_month or revenue missing")

def q30(): # Command center (summary KPIs)
    st.subheader("Q30: Business Intelligence Command Center")
    total_rev = df['final_amount_inr'].sum() if 'final_amount_inr' in df.columns else 0
    total_txns = df['transaction_id'].nunique() if 'transaction_id' in df.columns else 0
    total_customers = df['customer_id'].nunique() if 'customer_id' in df.columns else 0
    st.metric("Total Revenue", f"{total_rev:,.0f}")
    st.metric("Total Transactions", f"{total_txns:,}")
    st.metric("Total Customers", f"{total_customers:,}")

# ---------------- SIDEBAR ----------------
question = st.sidebar.selectbox("Select Question (1-30)", list(range(1,31)))

# ---------------- RUN ----------------
globals()[f"q{question}"]()
