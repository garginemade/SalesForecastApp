# Sales Forecasting
# Gargi Nemade

# Task 7 — Deployment: Interactive Dashboard using Streamlit
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# load dataset
df = pd.read_csv("train.csv")

# convert to time series format
df["Order Date"] = pd.to_datetime(
    df["Order Date"],
    dayfirst=True
)
df["Year"] = df["Order Date"].dt.year


# Home Page
st.set_page_config(
    page_title="Sales Forecast Dashboard",
    layout="wide"
)

st.title("Sales Forecast Dashboard")

st.markdown("""
### Welcome!

This dashboard contains four sections:

- Sales Overview
- Forecast Explorer
- Anomaly Report
- Product Demand Segments

Use the **sidebar** to navigate between pages.
""")

# Side Bar
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    [
        "Sales Overview",
        "Forecast Explorer",
        "Anomaly Report",
        "Product Demand Segments"
    ]
)

# Page 1 — Sales Overview Dashboard
if page == "Sales Overview":

    st.title("Sales Overview Dashboard")
    st.write("Superstore Sales Analysis")

    # Total Sales by Year
    st.subheader("Total Sales by Year")

    yearly_sales = (
        df.groupby("Year")["Sales"]
        .sum()
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(8,5))

    ax.bar(
        yearly_sales["Year"],
        yearly_sales["Sales"]
    )

    ax.set_title("Total Sales by Year")
    ax.set_xlabel("Year")
    ax.set_ylabel("Sales")

    st.pyplot(fig)

    # Monthly Sales Trend
    st.subheader("Monthly Sales Trend")

    monthly_sales = (
        df.groupby(
            pd.Grouper(
                key="Order Date",
                freq="M"
            )
        )["Sales"]
        .sum()
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(10,5))

    ax.plot(
        monthly_sales["Order Date"],
        monthly_sales["Sales"],
        marker="o",
        linewidth=2
    )

    ax.set_title("Monthly Sales Trend")
    ax.set_xlabel("Date")
    ax.set_ylabel("Sales")

    plt.xticks(rotation=45)

    st.pyplot(fig)

    # Interactive Filters
    st.subheader("Sales by Region and Category")

    col1, col2 = st.columns(2)

    with col1:
        region = st.selectbox(
            "Select Region",
            sorted(df["Region"].unique())
        )

    with col2:
        category = st.selectbox(
            "Select Category",
            sorted(df["Category"].unique())
        )

    filtered = df[
        (df["Region"] == region) &
        (df["Category"] == category)
    ]

    sales = (
        filtered.groupby("Sub-Category")["Sales"]
        .sum()
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(10,5))

    ax.bar(
        sales["Sub-Category"],
        sales["Sales"]
    )

    ax.set_title(f"{region} - {category} Sales")

    ax.set_xlabel("Sub-Category")
    ax.set_ylabel("Sales")
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)

    # Filtered Data
    st.subheader("Filtered Dataset")
    st.dataframe(filtered)

    # Summary Metrics
    st.subheader("Summary")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Total Sales",
        f"${filtered['Sales'].sum():,.0f}"
    )

    col2.metric(
        "Total Orders",
        filtered.shape[0]
    )

    col3.metric(
        "Average Sales",
        f"${filtered['Sales'].mean():,.2f}"
    )


# Page 2 — Forecast Explorer
elif page == "Forecast Explorer":

    st.title("Forecast Explorer")

    # Create Monthly Sales
    sales_df = (
        df.groupby(pd.Grouper(key="Order Date", freq="M"))["Sales"]
        .sum()
        .reset_index()
    )

    st.title("Sales Overview Dashboard")
    st.write("Superstore Sales Analysis")

    # Total Sales by Year
    st.subheader("Total Sales by Year")

    yearly_sales = (
        df.groupby("Year")["Sales"]
        .sum()
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(8,5))

    ax.bar(
        yearly_sales["Year"],
        yearly_sales["Sales"]
    )

    ax.set_title("Total Sales by Year")
    ax.set_xlabel("Year")
    ax.set_ylabel("Sales")

    st.pyplot(fig)

    # Monthly Sales Trend
    st.subheader("Monthly Sales Trend")

    monthly_sales = (
        df.groupby(
            pd.Grouper(
                key="Order Date",
                freq="M"
            )
        )["Sales"]
        .sum()
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(10,5))

    ax.plot(
        monthly_sales["Order Date"],
        monthly_sales["Sales"],
        marker="o",
        linewidth=2
    )

    ax.set_title("Monthly Sales Trend")
    ax.set_xlabel("Date")
    ax.set_ylabel("Sales")

    plt.xticks(rotation=45)

    st.pyplot(fig)

    # Interactive Filters
    st.subheader("Sales by Region and Category")

    col1, col2 = st.columns(2)

    with col1:
        region = st.selectbox(
            "Select Region",
            sorted(df["Region"].unique())
        )

    with col2:
        category = st.selectbox(
            "Select Category",
            sorted(df["Category"].unique())
        )

    filtered = df[
        (df["Region"] == region) &
        (df["Category"] == category)
    ]

    sales = (
        filtered.groupby("Sub-Category")["Sales"]
        .sum()
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(10,5))

    ax.bar(
        sales["Sub-Category"],
        sales["Sales"]
    )

    ax.set_title(f"{region} - {category} Sales")

    ax.set_xlabel("Sub-Category")

    ax.set_ylabel("Sales")

    plt.xticks(rotation=45)

    plt.tight_layout()

    st.pyplot(fig)

    # Filtered Data
    st.subheader("Filtered Dataset")

    st.dataframe(filtered)

    # Summary Metrics
    st.subheader("Summary")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Total Sales",
        f"${filtered['Sales'].sum():,.0f}"
    )

    col2.metric(
        "Total Orders",
        filtered.shape[0]
    )

    col3.metric(
    "Average Sales",
    f"${filtered['Sales'].mean():,.2f}"
   )

    # Lag Features
    sales_df["Lag_1"] = sales_df["Sales"].shift(1)
    sales_df["Lag_2"] = sales_df["Sales"].shift(2)
    sales_df["Lag_3"] = sales_df["Sales"].shift(3)

    # Rolling Mean
    sales_df["Rolling_Mean_3"] = sales_df["Sales"].rolling(3).mean()

    # Date Features
    sales_df["Month"] = sales_df["Order Date"].dt.month
    sales_df["Quarter"] = sales_df["Order Date"].dt.quarter

    sales_df = sales_df.dropna()

    X = sales_df.drop(columns=["Order Date", "Sales"])
    y = sales_df["Sales"]

    X_train = X[:-3]
    X_test = X[-3:]

    y_train = y[:-3]
    y_test = y[-3:]

    model = XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )

    model.fit(X_train, y_train)

    # Forecast Type
    forecast_type = st.selectbox(
        "Forecast By",
        ["Category", "Region"],
        key="forecast_type"
    )

    if forecast_type == "Category":
        option = st.selectbox(
            "Select Category",
            sorted(df["Category"].unique()),
            key="forecast_category"
        )

        filtered_df = df[df["Category"] == option]

    else:

        option = st.selectbox(
            "Select Region",
            sorted(df["Region"].unique()),
            key="forecast_region"
        )

        filtered_df = df[df["Region"] == option]

    # Forecast Horizon
    months = st.slider(
        "Forecast Horizon",
        1,
        3,
        3,
        key="forecast_horizon"
    )

    forecast_data = (
        filtered_df.groupby(
            pd.Grouper(key="Order Date", freq="M")
        )["Sales"]
        .sum()
        .reset_index()
    )

    forecast_data["Lag_1"] = forecast_data["Sales"].shift(1)
    forecast_data["Lag_2"] = forecast_data["Sales"].shift(2)
    forecast_data["Lag_3"] = forecast_data["Sales"].shift(3)

    forecast_data["Rolling_Mean_3"] = (
        forecast_data["Sales"]
        .rolling(3)
        .mean()
    )

    forecast_data["Month"] = forecast_data["Order Date"].dt.month
    forecast_data["Quarter"] = forecast_data["Order Date"].dt.quarter

    forecast_data = forecast_data.dropna()

    X = forecast_data.drop(columns=["Order Date", "Sales"])
    y = forecast_data["Sales"]

    X_train = X[:-3]
    X_test = X[-3:]

    y_train = y[:-3]
    y_test = y[-3:]

    model.fit(X_train, y_train)

    prediction = model.predict(X_test)
    prediction = prediction[:months]

    forecast_months = [
        f"Month {i+1}"
        for i in range(months)
    ]

    forecast_df = pd.DataFrame({
        "Forecast Month": forecast_months,
        "Predicted Sales": prediction
    })

    st.subheader("Forecast Output")
    st.dataframe(forecast_df)

    fig, ax = plt.subplots(figsize=(8,5))

    ax.plot(
        forecast_months,
        prediction,
        marker="o",
        linewidth=2
    )

    ax.set_title(f"{option} Sales Forecast")
    ax.set_xlabel("Forecast Month")
    ax.set_ylabel("Predicted Sales")
    ax.grid(True)

    st.pyplot(fig)

    # Model Performance
    xgb_mae = mean_absolute_error(y_test, prediction)
    xgb_rmse = np.sqrt(mean_squared_error(y_test, prediction))

    st.subheader("Model Performance")

    col1, col2 = st.columns(2)

    col1.metric(
        "MAE",
        f"{xgb_mae:,.2f}"
    )

    col2.metric(
        "RMSE",
        f"{xgb_rmse:,.2f}"
    )

# Page 3 — Anomaly Report
elif page == "Anomaly Report":

    st.title("Anomaly Report")

    from sklearn.ensemble import IsolationForest

    # Weekly Sales
    weekly_sales = (
        df.groupby(
            pd.Grouper(key="Order Date", freq="W")
        )["Sales"]
        .sum()
        .reset_index()
    )

    # Isolation Forest
    iso = IsolationForest(
        contamination=0.05,
        random_state=42
    )

    weekly_sales["Anomaly"] = iso.fit_predict(
        weekly_sales[["Sales"]]
    )

    anomalies = weekly_sales[
        weekly_sales["Anomaly"] == -1
    ]

    st.subheader("Weekly Sales with Detected Anomalies")

    fig, ax = plt.subplots(figsize=(12,5))

    ax.plot(
        weekly_sales["Order Date"],
        weekly_sales["Sales"],
        label="Weekly Sales"
    )

    ax.scatter(
        anomalies["Order Date"],
        anomalies["Sales"],
        color="red",
        s=80,
        label="Anomaly"
    )

    ax.set_title("Weekly Sales Anomalies")
    ax.set_xlabel("Week")
    ax.set_ylabel("Sales")
    ax.legend()

    st.pyplot(fig)

    st.subheader("Detected Anomalies")

    anomaly_table = anomalies[
        ["Order Date", "Sales"]
    ].copy()

    anomaly_table["Order Date"] = (
        anomaly_table["Order Date"]
        .dt.strftime("%d-%m-%Y")
    )

    st.dataframe(anomaly_table)

    st.metric(
        "Total Anomalies",
        len(anomaly_table)
    )

# Page 4 — Product Demand Segments
elif page == "Product Demand Segments":

    st.title("Product Demand Segments")

    cluster_df = pd.read_csv("cluster_results.csv")
    pca_df = pd.read_csv("pca_results.csv")

    st.subheader("Demand Cluster Visualization")

    fig, ax = plt.subplots(figsize=(10,6))

    scatter = ax.scatter(
        pca_df["PC1"],
        pca_df["PC2"],
        c=pca_df["Cluster"],
        s=120
    )

    for i in range(len(pca_df)):
        ax.text(
            pca_df.loc[i, "PC1"],
            pca_df.loc[i, "PC2"],
            pca_df.loc[i, "Sub-Category"],
            fontsize=8
        )

    ax.set_title("Product Demand Clusters")
    ax.set_xlabel("Principal Component 1")
    ax.set_ylabel("Principal Component 2")

    st.pyplot(fig)

    cluster_names = {
        0: "High Volume, Stable Demand",
        1: "Growing Demand",
        2: "Low Volume, High Volatility",
        3: "Declining Demand"
    }

    cluster_df["Demand Cluster"] = (
        cluster_df["Cluster"]
        .map(cluster_names)
    )

    st.subheader("Sub-Categories by Demand Cluster")

    st.dataframe(
        cluster_df[
            [
                "Sub-Category",
                "Demand Cluster"
            ]
        ]
    )

    st.subheader("Cluster Summary")

    st.dataframe(
        cluster_df["Demand Cluster"]
        .value_counts()
        .reset_index()
        .rename(
            columns={
                "index":"Demand Cluster",
                "Demand Cluster":"Number of Sub-Categories"
            }
        )
    )