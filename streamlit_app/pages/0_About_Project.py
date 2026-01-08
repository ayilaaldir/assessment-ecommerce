import os
import pandas as pd
import streamlit as st

st.title("Project Overview")

APP_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(APP_DIR, "..", ".."))
DATA_DIR = os.path.join(ROOT_DIR, "data", "processed")

FINAL_ORDERS_PATH = os.path.join(DATA_DIR, "final_orders.csv")
FINAL_REVIEWS_PATH = os.path.join(DATA_DIR, "final_reviews.csv")

@st.cache_data
def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

st.markdown(
    """
### About this project (Olist E-Commerce Analytics)

This dashboard analyzes the **Brazilian Olist e-commerce dataset** by integrating multiple raw tables
(orders, customers, items, payments, products, sellers, and reviews) into analytics-ready datasets.
The objective of this project is to provide a clear overview of **business performance**, **customer satisfaction**,
and **short-term trends** through exploratory analysis, sentiment analysis, and forecasting.

### Analysis period: Jan 2017 – Aug 2018
The analysis is limited to **1 January 2017 to 31 August 2018** for the following reasons:
- Data before 2017 contains fewer transactions and is not sufficiently representative for trend analysis.
- The dataset coverage ends in 2018, and **August 2018 is the last month with consistently complete data**
  across orders, payments, delivery information, and customer reviews.
- Using this time window ensures stable KPIs and avoids bias caused by incomplete months.

### Project contents
The dashboard is built using the following processed datasets:

- `final_orders.csv`  
  Order-level data containing purchase timestamps, delivery information,
  payment value, customer location, and delivery status.
  This table is the main source for KPI calculation and operational analysis.

- `final_reviews.csv`  
  Review-level data containing ratings, review text, and associated product categories.
  This dataset is used for customer satisfaction analysis and sentiment modeling.

- `final_category_monthly.csv`  
  Aggregated category–month dataset containing total orders and revenue per category.
  This table is created to support time-series analysis and forecasting preparation.

- `final_forecast_table_all_1m.csv`  
  Forecast output table containing next-month baseline predictions and trend indicators
  for each product category, used in the forecasting and insights section of the dashboard.

### Methods and analysis
- **Exploratory Data Analysis (EDA)**:
  order volume, revenue (payment value), product category performance, delivery status,
  and geographic distribution by state.
- **Sentiment analysis**:
  customer reviews are converted into binary sentiment labels based on ratings
  (negative: 1–2, positive: 4–5), supported by a multilingual transformer model
  trained on Portuguese review text.
- **Forecasting**:
  monthly order and revenue trends are modeled at the category level using a
  short-term baseline approach to identify near-future direction rather than long-term prediction.

### Limitations
- The analysis is limited to a relatively short time range (2017–2018), as earlier periods contain
  sparse and unstable transaction data.
- Many customer reviews do not contain textual feedback, therefore sentiment analysis is performed
  only on reviews with available text.
- Monthly time series at the product category level are relatively short, which limits the effectiveness
  of complex deep learning forecasting models and favors simpler baseline approaches.
- Geographic data may contain duplicate or aggregated entries, which can affect the granularity
  of location-based analysis.
"""
)