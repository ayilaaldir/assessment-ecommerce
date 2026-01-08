import os
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import json
from urllib.request import urlopen

st.set_page_config(page_title="Overview", layout="wide")
st.title("Overview")

APP_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(APP_DIR, "..", ".."))
DATA_DIR = os.path.join(ROOT_DIR, "data", "processed")

FINAL_ORDERS_PATH = os.path.join(DATA_DIR, "final_orders.csv")
FINAL_REVIEWS_PATH = os.path.join(DATA_DIR, "final_reviews.csv")


@st.cache_data
def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data
def load_brazil_states_geojson_online():
    url = "https://raw.githubusercontent.com/codeforamerica/click_that_hood/master/public/data/brazil-states.geojson"
    with urlopen(url) as response:
        return json.load(response)


def safe_to_datetime(df: pd.DataFrame, cols: list[str]) -> None:
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")


def pct_change(a: float, b: float) -> float:
    if a is None or not np.isfinite(a) or a <= 0:
        return np.nan
    return ((b - a) / a) * 100.0


def format_pct(x: float) -> str:
    if not np.isfinite(x):
        return "N/A"
    sign = "+" if x > 0 else ""
    return f"{sign}{x:.1f}%"


def label_trend(x: float, tol: float = 2.0) -> str:
    if not np.isfinite(x):
        return "N/A"
    if x > tol:
        return "Increase"
    if x < -tol:
        return "Decrease"
    return "Flat"


if not os.path.exists(FINAL_ORDERS_PATH):
    st.error(f"Missing file: {FINAL_ORDERS_PATH}")
    st.stop()

if not os.path.exists(FINAL_REVIEWS_PATH):
    st.error(f"Missing file: {FINAL_REVIEWS_PATH}")
    st.stop()

orders_df = load_csv(FINAL_ORDERS_PATH)
reviews_df = load_csv(FINAL_REVIEWS_PATH)

safe_to_datetime(
    orders_df,
    [
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ],
)

st.subheader("Filters")

if "order_purchase_timestamp" not in orders_df.columns:
    st.error("order_purchase_timestamp not found in final_orders.csv")
    st.stop()

tmp = orders_df.dropna(subset=["order_purchase_timestamp"]).copy()
tmp["month_dt"] = tmp["order_purchase_timestamp"].dt.to_period("M").dt.to_timestamp()

min_m = tmp["month_dt"].min()
max_m = tmp["month_dt"].max()

default_start = pd.Timestamp("2017-01-01")
default_end   = pd.Timestamp("2018-08-31")

default_start = max(default_start, min_m)
default_end   = min(default_end, max_m)

f1, f2 = st.columns(2)
with f1:
    start_m = st.date_input(
        "Start month",
        value=default_start.date(),
        min_value=min_m.date(),
        max_value=max_m.date(),
        key="global_start_month",
    )
with f2:
    end_m = st.date_input(
        "End month",
        value=default_end.date(),
        min_value=min_m.date(),
        max_value=max_m.date(),
        key="global_end_month",
    )

start_m = pd.to_datetime(start_m)
end_m = pd.to_datetime(end_m)

if start_m > end_m:
    st.warning("Start month is after end month. Please fix the range.")
    st.stop()

mask = (tmp["month_dt"] >= start_m) & (tmp["month_dt"] <= end_m)
orders_f = tmp.loc[mask].copy()

orders_in_range = set(orders_f["order_id"].dropna().unique().tolist())
reviews_f = reviews_df[reviews_df["order_id"].isin(orders_in_range)].copy()

POP_MONTHS = 3
prev_end = (start_m.to_period("M") - 1).to_timestamp()
prev_start = (start_m.to_period("M") - POP_MONTHS).to_timestamp()

dataset_min_month = tmp["month_dt"].min()

STABLE_START = pd.Timestamp("2017-01-01")
has_prev_window = (prev_start >= dataset_min_month) and (start_m >= STABLE_START)

if has_prev_window:
    prev_mask = (tmp["month_dt"] >= prev_start) & (tmp["month_dt"] <= prev_end)
    orders_prev = tmp.loc[prev_mask].copy()

    orders_prev_ids = set(orders_prev["order_id"].dropna().unique().tolist())
    reviews_prev = reviews_df[reviews_df["order_id"].isin(orders_prev_ids)].copy()
else:
    orders_prev = pd.DataFrame()
    reviews_prev = pd.DataFrame()

st.divider()

st.subheader("General Overview")

col1, col2, col3 = st.columns(3)
col1.metric("Total Orders", f"{len(orders_f):,}")

if "review_score" in reviews_f.columns and len(reviews_f) > 0:
    avg_rating = reviews_f["review_score"].dropna().mean()
    col2.metric("Average Rating", f"{avg_rating:.2f} / 5")
else:
    avg_rating = np.nan
    col2.metric("Average Rating", "N/A")

if "payment_value" in orders_f.columns:
    total_payment = orders_f["payment_value"].fillna(0).sum()
    col3.metric("Total Payment Value", f"{total_payment:,.2f}")
else:
    total_payment = np.nan
    col3.metric("Total Payment Value", "N/A")

st.divider()

st.markdown("### Delivery performance")

if {"delivery_status", "order_estimated_delivery_date", "order_delivered_customer_date"}.issubset(set(orders_f.columns)):
    delivered = orders_f[orders_f["delivery_status"].isin(["On Time", "Late"])].copy()
    delivered = delivered.dropna(subset=["order_estimated_delivery_date", "order_delivered_customer_date"])

    total_delivered = len(delivered)
    late = delivered[delivered["delivery_status"] == "Late"].copy()
    on_time = delivered[delivered["delivery_status"] == "On Time"].copy()

    late_rate = (len(late) / total_delivered) * 100.0 if total_delivered > 0 else np.nan

    delivered["diff_days"] = (delivered["order_delivered_customer_date"] - delivered["order_estimated_delivery_date"]).dt.days
    avg_late_days = float(delivered.loc[delivered["delivery_status"] == "Late", "diff_days"].mean()) if len(late) > 0 else np.nan
    avg_early_days = float((on_time["order_estimated_delivery_date"] - on_time["order_delivered_customer_date"]).dt.days.mean()) if len(on_time) > 0 else np.nan

    d1, d2, d3 = st.columns(3)
    d1.metric("Late delivery rate", "N/A" if not np.isfinite(late_rate) else f"{late_rate:.1f}%")
    d2.metric("Avg late days", "N/A" if not np.isfinite(avg_late_days) else f"{avg_late_days:.1f} days")
    d3.metric("Avg early days", "N/A" if not np.isfinite(avg_early_days) else f"{avg_early_days:.1f} days")
else:
    st.write("Delivery performance insight is not available (missing delivery columns).")

st.divider()

st.subheader("Delivery Performance Details")

needed = {"delivery_status", "order_estimated_delivery_date", "order_delivered_customer_date"}
missing = needed - set(orders_f.columns)

if missing:
    st.info(f"Missing columns for delivery details: {sorted(list(missing))}")
else:
    delivered = orders_f[orders_f["delivery_status"].isin(["On Time", "Late"])].copy()
    delivered = delivered.dropna(subset=["order_estimated_delivery_date", "order_delivered_customer_date"])

    delivered["delivery_diff_days"] = (
        (delivered["order_delivered_customer_date"] - delivered["order_estimated_delivery_date"]).dt.days
    )

    colA, colB = st.columns(2)

    with colA:
        st.markdown("#### On Time (delivered on/before estimate)")
        on_time = delivered[delivered["delivery_status"] == "On Time"].copy()
        if len(on_time) == 0:
            st.write("No rows.")
        else:
            on_time["early_days"] = (on_time["order_estimated_delivery_date"] - on_time["order_delivered_customer_date"]).dt.days
            st.write(
                {
                    "count": int(len(on_time)),
                    "avg_early_days": round(float(on_time["early_days"].mean()), 2),
                    "median_early_days": float(on_time["early_days"].median()),
                }
            )

    with colB:
        st.markdown("#### Late (delivered after estimate)")
        late = delivered[delivered["delivery_status"] == "Late"].copy()
        if len(late) == 0:
            st.write("No rows.")
        else:
            late["late_days"] = late["delivery_diff_days"]
            st.write(
                {
                    "count": int(len(late)),
                    "avg_late_days": round(float(late["late_days"].mean()), 2),
                    "median_late_days": float(late["late_days"].median()),
                }
            )

    st.divider()

    st.markdown("#### Not Delivered breakdown")
    not_delivered = orders_f[orders_f["delivery_status"] == "Not Delivered"].copy()

    if len(not_delivered) == 0:
        st.write("No rows.")
    else:
        if "order_status" in not_delivered.columns:
            status_counts = not_delivered["order_status"].value_counts().reset_index()
            status_counts.columns = ["order_status", "count"]

            left_nd, right_nd = st.columns([1, 1])

            with left_nd:
                st.dataframe(status_counts, use_container_width=True, height=300)

            with right_nd:
                st.bar_chart(status_counts.set_index("order_status")["count"])
        else:
            st.info("order_status column not found to break down Not Delivered pipeline.")

st.divider()

st.subheader("Top & Bottom Product Categories")

if "product_category_name_english" not in reviews_f.columns or len(reviews_f) == 0:
    st.write("Category breakdown is not available (missing category column or no reviews in selected range).")
else:
    cat_counts = (
        reviews_f["product_category_name_english"]
        .fillna("unknown")
        .value_counts()
        .reset_index()
    )
    cat_counts.columns = ["category", "count"]

    top10 = cat_counts.head(10).copy()
    bottom10 = cat_counts.tail(10).copy()

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("#### Top 10 categories")
        st.dataframe(top10, use_container_width=True, height=320)

        if len(top10) > 0:
            best = top10.iloc[0]
            share = (best["count"] / len(reviews_f)) * 100.0

            best_avg_rating = np.nan
            if "review_score" in reviews_f.columns:
                best_avg_rating = reviews_f.loc[
                    reviews_f["product_category_name_english"].fillna("unknown") == best["category"],
                    "review_score"
                ].dropna().mean()

            rating_txt = "N/A" if not np.isfinite(best_avg_rating) else f"{best_avg_rating:.2f} / 5"

            st.write(
                f"Highest category is **{best['category']}** with **{int(best['count']):,}** reviews "
                f"({share:.1f}% of reviews in selected range). Average rating: **{rating_txt}**."
            )

    with c2:
        st.markdown("#### Bottom 10 categories")
        st.dataframe(bottom10, use_container_width=True, height=320)

        if len(bottom10) > 0:
            worst = bottom10.iloc[-1]
            share = (worst["count"] / len(reviews_f)) * 100.0

            worst_avg_rating = np.nan
            if "review_score" in reviews_f.columns:
                worst_avg_rating = reviews_f.loc[
                    reviews_f["product_category_name_english"].fillna("unknown") == worst["category"],
                    "review_score"
                ].dropna().mean()

            rating_txt = "N/A" if not np.isfinite(worst_avg_rating) else f"{worst_avg_rating:.2f} / 5"

            st.write(
                f"Lowest category is **{worst['category']}** with **{int(worst['count']):,}** reviews "
                f"({share:.1f}% of reviews in selected range). Average rating: **{rating_txt}**."
            )

st.divider()

st.subheader("Trends and Geography")

left, right = st.columns(2)

with left:
    st.markdown("#### Monthly Orders Trend")
    trend = orders_f.groupby("month_dt").size().reset_index(name="orders")
    trend["month"] = trend["month_dt"].dt.to_period("M").astype(str)
    st.line_chart(trend.set_index("month")["orders"])
    st.dataframe(trend[["month", "orders"]], use_container_width=True)

with right:
    st.markdown("#### Orders by State (Choropleth Map)")

    if "customer_state" not in orders_f.columns:
        st.info("customer_state column not found in final_orders.csv")
    else:
        state_counts = orders_f["customer_state"].value_counts().reset_index()
        state_counts.columns = ["UF", "orders"]

        geojson = load_brazil_states_geojson_online()

        fig = px.choropleth(
            state_counts,
            geojson=geojson,
            locations="UF",
            featureidkey="properties.sigla",
            color="orders",
            color_continuous_scale="Blues",
            hover_name="UF",
            hover_data={"orders": True},
            projection="mercator",
        )
        fig.update_layout(height=350, margin={"r": 0, "t": 0, "l": 0, "b": 0})
        fig.update_geos(fitbounds="locations", visible=False)

        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(
            state_counts.sort_values("orders", ascending=False),
            use_container_width=True,
            height=295,
        )

        st.markdown("#### Geography insight")

        if "customer_state" in orders_f.columns and len(orders_f) > 0:
            state_counts = orders_f["customer_state"].value_counts()
            top_state = state_counts.index[0]
            top_orders = int(state_counts.iloc[0])
            top_share = (top_orders / len(orders_f)) * 100.0
            st.write(f"Top state is **{top_state}** with **{top_orders:,}** orders ({top_share:.1f}% of selected range orders).")
        else:
            st.write("Geography insight is not available (no customer_state or no orders in range).")