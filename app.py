# streamlit_30q.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine
from urllib.parse import quote_plus

st.set_page_config("Amazon Sales Analytics (30 Qs)", layout="wide")
sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 5)

# -------------------------
# DB CONFIG - edit these
# -------------------------
DB_USER = "root"
DB_PASS = quote_plus("Mathan@123")   # replace with your password, or use st.secrets
DB_HOST = "localhost"
DB_NAME = "amazon_sales_analytics"

ENGINE_STR = f"mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}/{DB_NAME}"
engine = create_engine(ENGINE_STR, pool_pre_ping=True)

# -------------------------
# Load & prepare data
# -------------------------
@st.cache_data
def load_data():
    # Read tables; parse order_date as date
    tx = pd.read_sql("SELECT * FROM transactions", engine, parse_dates=["order_date"])
    prod = pd.read_sql("SELECT * FROM products", engine)
    cust = pd.read_sql("SELECT * FROM customers", engine)
    time_dim = pd.read_sql("SELECT * FROM time_dimension", engine)

    # Normalize transaction column names to match your list if needed
    # Ensure numeric columns exist
    for col in ["final_amount_inr", "discounted_price_inr", "original_price_inr", "quantity", "delivery_days"]:
        if col in tx.columns:
            tx[col] = pd.to_numeric(tx[col], errors="coerce")

    # Merge prefer product & customer authoritative columns
    df = tx.merge(prod, on="product_id", how="left", suffixes=("", "_prod"))
    df = df.merge(cust, on="customer_id", how="left", suffixes=("", "_cust"))

    # Simple canonical names for plotting (choose existing columns)
    if "product_name" in df.columns:
        df["product_name_c"] = df["product_name"]
    elif "product_name_x" in df.columns:
        df["product_name_c"] = df["product_name_x"]
    else:
        df["product_name_c"] = None

    # category:
    for c in ("category", "category_x", "category_y"):
        if c in df.columns:
            df["category_c"] = df[c].astype(str).str.strip().replace({"nan": np.nan})
            break
    else:
        df["category_c"] = None

    # brand:
    for b in ("brand", "brand_x", "brand_y"):
        if b in df.columns:
            df["brand_c"] = df[b].astype(str).str.strip().replace({"nan": np.nan})
            break
    else:
        df["brand_c"] = None

    # age group:
    for a in ("customer_age_group", "customer_age_group_x", "customer_age_group_y"):
        if a in df.columns:
            df["age_group_c"] = df[a].astype(str).str.strip().replace({"nan": np.nan})
            break
    else:
        df["age_group_c"] = None

    # city/state canonical
    for cc in ("customer_city", "customer_city_x", "customer_city_y"):
        if cc in df.columns:
            df["city_c"] = df[cc]
            break
    else:
        df["city_c"] = None
    for ss in ("customer_state", "customer_state_x", "customer_state_y"):
        if ss in df.columns:
            df["state_c"] = df[ss]
            break
    else:
        df["state_c"] = None

    # unit price (fallback calculation)
    if ("discounted_price_inr" in df.columns) and ("quantity" in df.columns):
        df["unit_price_calc"] = df["discounted_price_inr"] / df["quantity"].replace({0: np.nan})
    else:
        df["unit_price_calc"] = np.nan

    # derived time columns if missing
    if "order_year" not in df.columns and "order_date" in df.columns:
        df["order_year"] = df["order_date"].dt.year
    if "order_month" not in df.columns and "order_date" in df.columns:
        df["order_month"] = df["order_date"].dt.month
    if "order_quarter" not in df.columns and "order_date" in df.columns:
        df["order_quarter"] = df["order_date"].dt.quarter

    # final_amount fallback if not present
    if "final_amount_inr" not in df.columns and "subtotal_inr" in df.columns:
        df["final_amount_inr"] = df["subtotal_inr"]

    return df, tx, prod, cust, time_dim

try:
    df, tx, prod, cust, time_dim = load_data()
    st.sidebar.success("Connected & data loaded")
except Exception as e:
    st.sidebar.error(f"Data load error: {e}")
    st.stop()

# -------------------------
# small helpers
# -------------------------
def safe_bar(series, title=None, max_items=20):
    series = series.dropna()
    if series.empty:
        st.write("No data to show")
        return
    series = series.sort_values(ascending=False).head(max_items)
    st.bar_chart(series)

def safe_line(series, title=None):
    series = series.dropna()
    if series.empty:
        st.write("No data to show")
        return
    st.line_chart(series)

def safe_table(df_table, max_rows=200):
    if df_table.empty:
        st.write("No data to show")
    else:
        st.dataframe(df_table.head(max_rows))

def safe_show_matplotlib():
    st.pyplot(plt.gcf()); plt.clf()

# -------------------------
# Implementations Q1 - Q30 (simple & minimal)
# -------------------------
def q1_executive_summary():
    st.header("Q1 — Executive Summary")
    total_rev = df["final_amount_inr"].sum(skipna=True)
    ao = df["final_amount_inr"].mean(skipna=True)
    active_customers = df["customer_id"].nunique()
    # YoY growth simple
    yearly = df.groupby("order_year")["final_amount_inr"].sum().sort_index()
    yoy = yearly.pct_change() * 100
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Revenue (INR)", f"{total_rev:,.0f}")
    col2.metric("Active Customers", f"{active_customers:,}")
    col3.metric("Avg Order Value", f"{ao:,.2f}")
    st.subheader("Yearly revenue and YoY%")
    st.table(pd.concat([yearly.rename("revenue"), yoy.rename("yoy_pct")], axis=1).fillna(0).round(2))
    st.subheader("Top Categories by Revenue")
    safe_bar(df.groupby("category_c")["final_amount_inr"].sum(), None, max_items=10)

def q2_realtime_monitor():
    st.header("Q2 — Real-time Business Performance (current month)")
    if df["order_date"].isna().all():
        st.write("No order_date data")
        return
    current_month = df["order_date"].dt.to_period("M").max()
    dfm = df[df["order_date"].dt.to_period("M") == current_month]
    st.metric("Current month", str(current_month))
    st.metric("Revenue this month", f"{dfm['final_amount_inr'].sum():,.0f}")
    st.metric("Orders this month", f"{len(dfm):,}")
    st.metric("Active customers this month", f"{dfm['customer_id'].nunique():,}")

def q3_strategic_overview():
    st.header("Q3 — Strategic Overview")
    st.write("Market share by category (simple %) and Top 10 categories")
    cat_rev = df.groupby("category_c")["final_amount_inr"].sum()
    cat_pct = (cat_rev / cat_rev.sum() * 100).sort_values(ascending=False)
    safe_table(cat_pct.rename("pct").reset_index().head(10))
    safe_bar(cat_rev, "Top categories", max_items=10)

def q4_financial_performance():
    st.header("Q4 — Financial Performance (simple)")
    st.write("Revenue by category and basic breakdown")
    safe_bar(df.groupby("category_c")["final_amount_inr"].sum(), max_items=12)
    st.write("Basic profit margin not available (no cost fields) — showing revenue trend")
    safe_line(df.groupby("order_year")["final_amount_inr"].sum())

def q5_growth_analytics():
    st.header("Q5 — Growth Analytics")
    st.write("Category revenue consolidated (normalize variants)")
    # normalize variants: lowercase stripped mapping common typos
    df["category_norm"] = df["category_c"].astype(str).str.strip().str.title().replace({
        "Electronic": "Electronics", "Electronics & Accessories": "Electronics", "Electronicss": "Electronics"
    })
    safe_bar(df.groupby("category_norm")["final_amount_inr"].sum(), max_items=12)

def q6_revenue_trend():
    st.header("Q6 — Revenue Trend (monthly / yearly)")
    yearly = df.groupby("order_year")["final_amount_inr"].sum().sort_index()
    monthly = df.groupby([df["order_date"].dt.to_period("M")])["final_amount_inr"].sum().sort_index()
    st.subheader("Yearly")
    safe_line(yearly)
    st.subheader("Monthly (last 36 months)")
    safe_line(monthly.tail(36).rename_axis("month").reset_index().set_index("order_date")["final_amount_inr"])

def q7_category_performance():
    st.header("Q7 — Category Performance")
    grouped = df.groupby("category_c")["final_amount_inr"].agg(["sum","count"]).sort_values("sum", ascending=False)
    st.dataframe(grouped.head(20))
    safe_bar(grouped["sum"], max_items=15)

def q8_geographic_revenue():
    st.header("Q8 — Geographic Revenue (state / top cities)")
    state_rev = df.groupby("state_c")["final_amount_inr"].sum().sort_values(ascending=False)
    safe_bar(state_rev, max_items=15)
    st.subheader("Top cities")
    city_rev = df.groupby("city_c")["final_amount_inr"].sum().sort_values(ascending=False)
    safe_bar(city_rev, max_items=15)

def q9_festival_sales():
    st.header("Q9 — Festival Sales Impact")
    if "is_festival_sale" in df.columns:
        fest = df.groupby(["is_festival_sale"])["final_amount_inr"].sum()
        st.table(fest)
        safe_bar(fest)
    else:
        st.write("No is_festival_sale column in dataset")

def q10_price_optimization():
    st.header("Q10 — Price vs Demand (simple)")
    if ("discounted_price_inr" in df.columns) and ("quantity" in df.columns):
        sample = df.dropna(subset=["discounted_price_inr","quantity"]).sample(min(2000, len(df)))
        st.write("Scatter: unit price vs quantity (sample)")
        st.scatter_chart(sample[["discounted_price_inr","quantity"]])
        st.write("Correlation:")
        st.write(sample[["discounted_price_inr","quantity"]].corr().round(3))
    else:
        st.write("discounted_price_inr or quantity missing")

def q11_rfm():
    st.header("Q11 — Customer Segmentation (RFM)")
    if not {"customer_id","order_date","final_amount_inr"}.issubset(df.columns):
        st.write("Required columns missing")
        return
    latest = df["order_date"].max()
    rfm = df.groupby("customer_id").agg(Recency=("order_date", lambda x: (latest - x.max()).days),
                                       Frequency=("transaction_id","count"),
                                       Monetary=("final_amount_inr","sum")).reset_index()
    st.dataframe(rfm.head(10))
    fig, ax = plt.subplots()
    sns.scatterplot(data=rfm.sample(min(2000,len(rfm))), x="Frequency", y="Monetary", size="Recency", alpha=0.6, ax=ax)
    st.pyplot(fig)

def q12_customer_journey():
    st.header("Q12 — Customer Journey (simple)")
    # show distribution of number of categories per customer
    cust_cat = df.groupby("customer_id")["category_c"].nunique().dropna()
    st.write("Distribution: # distinct categories visited per customer")
    st.bar_chart(cust_cat.value_counts().sort_index())

def q13_prime_membership():
    st.header("Q13 — Prime Membership Analysis")
    if "is_prime_member" in df.columns:
        avg_by_prime = df.groupby("is_prime_member")["final_amount_inr"].agg(["mean","count"])
        st.table(avg_by_prime)
        st.write("Avg order value comparison")
        safe_bar(df.groupby("is_prime_member")["final_amount_inr"].mean())
    else:
        st.write("No is_prime_member column")

def q14_retention_clv():
    st.header("Q14 — CLV & Cohort (simple)")
    clv = df.groupby("customer_id")["final_amount_inr"].sum().sort_values(ascending=False)
    st.write("Top 10 CLV customers")
    st.table(clv.head(10).rename("CLV").reset_index())
    st.subheader("CLV distribution (histogram)")
    st.bar_chart(pd.cut(clv, bins=20).value_counts().sort_index())

def q15_demographics_behavior():
    st.header("Q15 — Demographics & Behavior")
    if df["age_group_c"].notna().any():
        safe_bar(df.groupby("age_group_c")["final_amount_inr"].sum(), max_items=12)
    else:
        st.write("No age group data")

def q16_product_performance():
    st.header("Q16 — Product Performance")
    prod_rev = df.groupby("product_id")["final_amount_inr"].sum().sort_values(ascending=False)
    safe_bar(prod_rev, max_items=15)
    st.write("Top products (ID + name)")
    top = prod_rev.head(15).reset_index()
    top["name"] = top["product_id"].map(df.set_index("product_id")["product_name_c"].to_dict())
    st.table(top)

def q17_brand_analytics():
    st.header("Q17 — Brand Analytics")
    if df["brand_c"].notna().any():
        safe_bar(df.groupby("brand_c")["final_amount_inr"].sum(), max_items=20)
    else:
        st.write("No brand data")

def q18_inventory_lifecycle():
    st.header("Q18 — Inventory & Product Lifecycle (launch_year)")
    if "launch_year" in df.columns:
        life = df.groupby("launch_year")["final_amount_inr"].sum().sort_index()
        safe_line(life)
    else:
        st.write("No launch_year column")

def q19_product_rating_reviews():
    st.header("Q19 — Rating & Reviews Impact")
    if "customer_rating" in df.columns:
        sns.boxplot(x=df["customer_rating"].dropna())
        plt.title("Customer rating distribution")
        safe_show_matplotlib()
        st.write("Correlation between rating and revenue per transaction (simple)")
        if {"customer_rating","final_amount_inr"}.issubset(df.columns):
            st.write(df[["customer_rating","final_amount_inr"]].corr().round(3))
    else:
        st.write("No customer_rating column")

def q20_new_product_launch():
    st.header("Q20 — New Product Launch Performance")
    if "launch_year" in df.columns:
        launch_perf = df.groupby("launch_year")["final_amount_inr"].sum().sort_index()
        safe_line(launch_perf)
    else:
        st.write("No launch_year info")

def q21_delivery_performance():
    st.header("Q21 — Delivery Performance")
    if "delivery_days" in df.columns:
        st.write("Delivery days distribution")
        st.bar_chart(df["delivery_days"].dropna().value_counts().sort_index())
        if "customer_rating" in df.columns:
            perf = df.groupby("delivery_days")["customer_rating"].mean().dropna()
            st.line_chart(perf)
    else:
        st.write("No delivery_days")

def q22_payment_analytics():
    st.header("Q22 — Payment Analytics")
    if "payment_method" in df.columns:
        safe_bar(df.groupby("payment_method")["final_amount_inr"].sum(), max_items=15)
    else:
        st.write("No payment_method column")

def q23_returns_cancellations():
    st.header("Q23 — Returns & Cancellations")
    if "return_status" in df.columns:
        ret = df["return_status"].value_counts()
        safe_bar(ret, max_items=20)
    else:
        st.write("No return_status")

def q24_customer_service():
    st.header("Q24 — Customer Service (proxy via returns & ratings)")
    if "customer_rating" in df.columns:
        st.write("Average rating by state (if available)")
        if df["state_c"].notna().any():
            st.table(df.groupby("state_c")["customer_rating"].mean().sort_values(ascending=False).head(10))
        else:
            st.write("No state info")
    else:
        st.write("No rating info")

def q25_supply_chain():
    st.header("Q25 — Supply Chain (proxy: product weight & revenue)")
    if "weight_kg" in df.columns:
        w = df.groupby("product_id").agg(weight=("weight_kg","first"), revenue=("final_amount_inr","sum")).reset_index()
        st.table(w.sort_values("revenue", ascending=False).head(10))
    else:
        st.write("No weight_kg column available")

def q26_predictive_analytics():
    st.header("Q26 — Predictive Analytics (simple linear trend projection)")
    yearly = df.groupby("order_year")["final_amount_inr"].sum().dropna().sort_index()
    if len(yearly) >= 3:
        st.line_chart(yearly)
        st.write("Simple linear trend (OLS) projection for next year")
        x = np.arange(len(yearly))
        y = yearly.values
        m, b = np.polyfit(x, y, 1)
        next_val = m * (len(x)) + b
        st.write(f"Projected next-year revenue (approx): {next_val:,.0f}")
    else:
        st.write("Not enough yearly points to project")

def q27_market_intelligence():
    st.header("Q27 — Market Intelligence (basic)")
    st.write("Top categories + brands (revenue share)")
    safe_table(df.groupby("category_c")["final_amount_inr"].sum().sort_values(ascending=False).head(10).rename("revenue").reset_index())

def q28_cross_sell_upsell():
    st.header("Q28 — Cross-sell / Association (simple co-purchase counts)")
    # simple approach: for orders with same transaction_id or order_id, count category pairs
    key = "transaction_id" if "transaction_id" in df.columns else "order_id" if "order_id" in df.columns else None
    if key:
        order_groups = df.groupby(key)["category_c"].apply(lambda s: list(s.dropna().unique()))
        from collections import Counter
        pair_counts = Counter()
        for cats in order_groups:
            cats = [c for c in cats if pd.notna(c)]
            for i in range(len(cats)):
                for j in range(i+1, len(cats)):
                    pair_counts[tuple(sorted((cats[i], cats[j])))] += 1
        pc = pd.DataFrame(pair_counts.items(), columns=["pair","count"]).sort_values("count", ascending=False).head(20)
        st.table(pc)
    else:
        st.write("No order grouping key available")

def q29_seasonal_planning():
    st.header("Q29 — Seasonal Planning (monthly seasonality)")
    monthly = df.groupby(df["order_date"].dt.month)["final_amount_inr"].sum()
    monthly.index = range(1,13)
    st.bar_chart(monthly)

def q30_bi_command_center():
    st.header("Q30 — BI Command Center (key KPIs)")
    total_rev = df["final_amount_inr"].sum()
    customers_count = df["customer_id"].nunique()
    orders_count = len(df)
    avg_order = df["final_amount_inr"].mean()
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Revenue", f"{total_rev:,.0f}")
    col2.metric("Customers", f"{customers_count:,}")
    col3.metric("Orders", f"{orders_count:,}")
    col4.metric("Avg Order", f"{avg_order:,.2f}")
    st.write("Small overview charts")
    st.line_chart(df.groupby("order_year")["final_amount_inr"].sum())

# map question numbers to functions
Q_FUNCS = {
    1: q1_executive_summary, 2: q2_realtime_monitor, 3: q3_strategic_overview,
    4: q4_financial_performance, 5: q5_growth_analytics, 6: q6_revenue_trend,
    7: q7_category_performance, 8: q8_geographic_revenue, 9: q9_festival_sales,
    10: q10_price_optimization, 11: q11_rfm, 12: q12_customer_journey,
    13: q13_prime_membership, 14: q14_retention_clv, 15: q15_demographics_behavior,
    16: q16_product_performance, 17: q17_brand_analytics, 18: q18_inventory_lifecycle,
    19: q19_product_rating_reviews, 20: q20_new_product_launch,
    21: q21_delivery_performance, 22: q22_payment_analytics, 23: q23_returns_cancellations,
    24: q24_customer_service, 25: q25_supply_chain, 26: q26_predictive_analytics,
    27: q27_market_intelligence, 28: q28_cross_sell_upsell, 29: q29_seasonal_planning,
    30: q30_bi_command_center
}

# -------------------------
# Sidebar & run selected question
# -------------------------
st.sidebar.title("Amazon Analytics — Select")
qsel = st.sidebar.number_input("Question (1-30)", min_value=1, max_value=30, value=1, step=1)
st.sidebar.write("Choose a question and press Run")
if st.sidebar.button("Run selected question"):
    try:
        st.write(f"Running Question {qsel}")
        Q_FUNCS[qsel]()   # run the chosen function (they access global df)
    except Exception as e:
        st.error(f"Error running Q{qsel}: {e}")

# show small help / dataset info
with st.expander("Dataset info & columns"):
    st.write("Transactions rows:", len(df))
    st.write("Columns (sample):")
    st.write(df.columns.tolist())
    st.write("You can edit DB credentials at top of the file if needed.")
