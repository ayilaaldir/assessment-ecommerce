import os
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Forecast", layout="wide")
st.title("Forecast")


APP_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(APP_DIR, "..", ".."))
DATA_DIR = os.path.join(ROOT_DIR, "data", "processed")

FORECAST_ALL_1M_PATH = os.path.join(DATA_DIR, "final_forecast_table_all_1m.csv")
FINAL_CATEGORY_MONTHLY_PATH = os.path.join(DATA_DIR, "final_category_monthly.csv")


@st.cache_data
def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def _safe_float(x, default=np.nan) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _safe_int(x, default=0) -> int:
    try:
        return int(round(float(x)))
    except Exception:
        return default


def compute_forecast_insights(row: pd.Series, hist: pd.DataFrame) -> dict:
    """
    Returns a dict of insight strings computed from:
    - 1-month forecast row (naive baseline: forecast == last)
    - category monthly history
    """
    insights = {}

    last_orders = _safe_int(row.get("last_orders", np.nan))
    last_rev = _safe_float(row.get("last_revenue", np.nan))
    trend_orders = str(row.get("orders_trend", "N/A"))
    trend_rev = str(row.get("revenue_trend", "N/A"))
    chg_orders_3m = _safe_float(row.get("orders_change_pct_3m", np.nan))
    chg_rev_3m = _safe_float(row.get("revenue_change_pct_3m", np.nan))

    insights["method"] = (
        "Forecast method: this deployed 1-month forecast uses a naive baseline (next month equals last month). "
        "It was chosen because deep models (LSTM/GRU/BiLSTM) performed worse on this dataset with short history per category. "
        "Interpretation: this forecast is a conservative 'hold-last-month' estimate, best for short-horizon planning."
    )

    if np.isfinite(chg_orders_3m):
        insights["momentum_orders"] = (
            f"Recent demand momentum (orders): {trend_orders} over the last 3 months "
            f"({chg_orders_3m:.2f}% change vs previous period). "
            "Use this to understand whether the category is heating up or cooling down recently."
        )
    else:
        insights["momentum_orders"] = (
            "Recent demand momentum (orders): not enough data to compute the 3-month change reliably for this category."
        )

    if np.isfinite(chg_rev_3m):
        insights["momentum_revenue"] = (
            f"Recent revenue momentum: {trend_rev} over the last 3 months "
            f"({chg_rev_3m:.2f}% change). "
            "If revenue momentum differs from order momentum, pricing or product mix may be shifting."
        )
    else:
        insights["momentum_revenue"] = (
            "Recent revenue momentum: not enough data to compute the 3-month change reliably for this category."
        )

    if last_orders > 0 and np.isfinite(last_rev):
        aov = last_rev / last_orders
        insights["aov"] = (
            f"Revenue per order (AOV proxy) last month: {aov:,.2f}. "
            "If orders are flat but revenue rises, AOV is likely improving (higher basket size or price mix)."
        )
    else:
        insights["aov"] = (
            "Revenue per order (AOV proxy): cannot compute because last month orders or revenue is missing/zero."
        )

    if hist is None or len(hist) == 0:
        insights["history"] = (
            "Historical pattern: no monthly history rows found for this category, so volatility and peak detection are unavailable."
        )
        return insights

    hist = hist.sort_values("order_month").copy()

    months_available = int(hist["order_month"].nunique())
    full_months = pd.date_range(hist["order_month"].min(), hist["order_month"].max(), freq="MS")
    missing_months = int(len(full_months) - hist["order_month"].nunique())

    tail = hist.tail(12).copy() if months_available >= 12 else hist.copy()

    ord_mean = float(tail["orders"].mean()) if tail["orders"].notna().any() else np.nan
    ord_std = float(tail["orders"].std()) if tail["orders"].notna().any() else np.nan
    rev_mean = float(tail["revenue"].mean()) if tail["revenue"].notna().any() else np.nan
    rev_std = float(tail["revenue"].std()) if tail["revenue"].notna().any() else np.nan

    vol_lines = []
    if np.isfinite(ord_mean) and ord_mean > 0 and np.isfinite(ord_std):
        cv_orders = ord_std / ord_mean
        vol_lines.append(f"Orders volatility (last {len(tail)} months): CV={cv_orders:.2f} (higher = more unstable).")
    else:
        vol_lines.append("Orders volatility: insufficient data.")

    if np.isfinite(rev_mean) and rev_mean > 0 and np.isfinite(rev_std):
        cv_rev = rev_std / rev_mean
        vol_lines.append(f"Revenue volatility (last {len(tail)} months): CV={cv_rev:.2f} (higher = more unstable).")
    else:
        vol_lines.append("Revenue volatility: insufficient data.")

    peak_orders_idx = hist["orders"].idxmax() if hist["orders"].notna().any() else None
    peak_rev_idx = hist["revenue"].idxmax() if hist["revenue"].notna().any() else None

    peak_orders_txt = "Peak orders month: N/A"
    if peak_orders_idx is not None and peak_orders_idx in hist.index:
        peak_orders_month = hist.loc[peak_orders_idx, "order_month"]
        peak_orders_val = hist.loc[peak_orders_idx, "orders"]
        peak_orders_txt = f"Peak orders month: {peak_orders_month.strftime('%Y-%m')} ({int(peak_orders_val):,} orders)."

    peak_rev_txt = "Peak revenue month: N/A"
    if peak_rev_idx is not None and peak_rev_idx in hist.index:
        peak_rev_month = hist.loc[peak_rev_idx, "order_month"]
        peak_rev_val = hist.loc[peak_rev_idx, "revenue"]
        peak_rev_txt = f"Peak revenue month: {peak_rev_month.strftime('%Y-%m')} ({float(peak_rev_val):,.2f})."

    recent_note = "Recent direction check: insufficient data."
    if months_available >= 6:
        last3 = hist.tail(3)
        prev3 = hist.iloc[-6:-3]
        last3_avg = float(last3["orders"].mean())
        prev3_avg = float(prev3["orders"].mean())
        if prev3_avg > 0:
            pct = ((last3_avg - prev3_avg) / prev3_avg) * 100.0
            label = "up" if pct > 2 else ("down" if pct < -2 else "flat")
            recent_note = (
                f"Recent direction check (orders): last 3-month avg {last3_avg:,.1f} vs prev 3-month avg {prev3_avg:,.1f} "
                f"({pct:.1f}%, {label})."
            )
        else:
            recent_note = "Recent direction check (orders): previous 3-month average is zero, cannot compute percent change."

    coverage_note = (
        f"Data coverage: {months_available} months available for this category. "
        f"Missing months in the range: {missing_months}. "
        "Short and sparse histories are one main reason simple baselines can outperform deep models here."
    )

    insights["history"] = "\n".join(
        [coverage_note, recent_note, peak_orders_txt, peak_rev_txt] + vol_lines
    )

    insights["interpretation"] = (
        "Interpretation note: because the deployed forecast equals last month, use it as a short-horizon planning baseline. "
        "The 'trend' labels come from recent historical momentum; if the category is volatile or has missing months, "
        "treat trend signals cautiously and validate with the historical chart."
    )

    return insights

st.write(
    "Deployed 1-month forecast uses naive baseline (next month equals last month) because it outperformed LSTM/GRU/BiLSTM on this dataset with short history."
)

if not os.path.exists(FORECAST_ALL_1M_PATH):
    st.error("Missing file:")
    st.write(FORECAST_ALL_1M_PATH)
    st.stop()

forecast_1m = load_csv(FORECAST_ALL_1M_PATH)

if "last_month" in forecast_1m.columns:
    forecast_1m["last_month"] = pd.to_datetime(forecast_1m["last_month"], errors="coerce")
if "forecast_month" in forecast_1m.columns:
    forecast_1m["forecast_month"] = pd.to_datetime(forecast_1m["forecast_month"], errors="coerce")

if "category" not in forecast_1m.columns:
    st.error("forecast table must contain a 'category' column.")
    st.stop()

categories = sorted(forecast_1m["category"].dropna().unique().tolist())
cat = st.selectbox("Search category:", categories)

st.divider()

sub = forecast_1m[forecast_1m["category"] == cat]
if len(sub) == 0:
    st.info("No forecast row found for selected category.")
    st.stop()

row = sub.iloc[0]

st.subheader("1-month forecast summary")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Last month", str(row["last_month"].date()) if pd.notna(row.get("last_month")) else "N/A")
c2.metric("Forecast month", str(row["forecast_month"].date()) if pd.notna(row.get("forecast_month")) else "N/A")
c3.metric("Forecast orders", f"{_safe_int(row.get('forecast_orders')):,}")
c4.metric("Forecast revenue", f"{_safe_float(row.get('forecast_revenue')):,.2f}")

st.write(
    {
        "orders_trend": row.get("orders_trend", "N/A"),
        "orders_change_pct_3m": row.get("orders_change_pct_3m", "N/A"),
        "revenue_trend": row.get("revenue_trend", "N/A"),
        "revenue_change_pct_3m": row.get("revenue_change_pct_3m", "N/A"),
    }
)

st.divider()

hist_df = None
if os.path.exists(FINAL_CATEGORY_MONTHLY_PATH):
    monthly_df = load_csv(FINAL_CATEGORY_MONTHLY_PATH)

    if "order_month" in monthly_df.columns:
        monthly_df["order_month"] = pd.to_datetime(monthly_df["order_month"] + "-01", errors="coerce")

    if "product_category_name_english" in monthly_df.columns:
        cat_col = "product_category_name_english"
    elif "category" in monthly_df.columns:
        cat_col = "category"
    else:
        cat_col = None

    if cat_col is not None and {"orders", "revenue", "order_month"}.issubset(set(monthly_df.columns)):
        hist_df = (
            monthly_df[monthly_df[cat_col] == cat]
            .dropna(subset=["order_month"])
            .sort_values("order_month")
            .copy()
        )
        hist_df = hist_df[["order_month", "orders", "revenue"]]
else:
    hist_df = None

st.subheader("Insights")

ins = compute_forecast_insights(row, hist_df)

with st.expander("Method insight (why this forecast approach)", expanded=True):
    st.write(ins["method"])

with st.expander("Category insight (selected category)", expanded=True):
    st.write(ins["momentum_orders"])
    st.write(ins["momentum_revenue"])
    st.write(ins["aov"])
    st.write(ins["interpretation"])

with st.expander("Historical pattern insight (based on monthly history)", expanded=True):
    st.write(ins["history"])

st.divider()

st.subheader("Historical Data")

if hist_df is None or len(hist_df) == 0:
    st.info("No historical rows found for this category (final_category_monthly.csv).")
else:
    min_hist = hist_df["order_month"].min()
    max_hist = hist_df["order_month"].max()

    h1, h2 = st.columns(2)
    with h1:
        hist_start = st.date_input(
            "History start",
            value=min_hist.date(),
            min_value=min_hist.date(),
            max_value=max_hist.date(),
            key="hist_start",
        )
    with h2:
        hist_end = st.date_input(
            "History end",
            value=max_hist.date(),
            min_value=min_hist.date(),
            max_value=max_hist.date(),
            key="hist_end",
        )

    hist_start = pd.to_datetime(hist_start)
    hist_end = pd.to_datetime(hist_end)

    if hist_start > hist_end:
        st.warning("History start is after history end. Please fix the range.")
        st.stop()

    hist_view = hist_df[(hist_df["order_month"] >= hist_start) & (hist_df["order_month"] <= hist_end)].copy()

    if len(hist_view) == 0:
        st.info("No historical rows in the selected history range.")
    else:
        left, right = st.columns([2, 1])

        with left:
            st.markdown("### Chart")
            chart_df = hist_view.set_index("order_month")[["orders", "revenue"]]
            st.line_chart(chart_df)

        with right:
            st.markdown("### Table")
            st.dataframe(hist_view, use_container_width=True, height=330)