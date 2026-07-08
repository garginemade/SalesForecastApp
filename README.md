# 📈 Sales Forecasting and Demand Analysis Dashboard

An end-to-end Data Science project that analyzes historical Superstore sales data, forecasts future sales using multiple machine learning and statistical models, detects anomalies, segments product demand, and presents insights through an interactive Streamlit dashboard.

---

## 🚀 Live Demo

🔗 **Streamlit App:**  
[https://your-streamlit-app.streamlit.app](https://salesforecastapp-garginemade-777.streamlit.app/)

---

## 📌 Project Overview

This project focuses on helping businesses make better inventory and sales planning decisions by:

- Analyzing historical sales trends
- Forecasting future sales
- Detecting unusual sales patterns
- Segmenting products based on demand
- Providing an interactive dashboard for business users

---

## 📊 Dataset

**Superstore Sales Dataset**

Files used:
- `train.csv`

---

## 🛠 Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- XGBoost
- Statsmodels (SARIMA)
- Facebook Prophet
- Streamlit

---

## 📈 Project Workflow

### 1. Data Preprocessing
- Data cleaning
- Date conversion
- Feature engineering
- Monthly and weekly aggregation

### 2. Exploratory Data Analysis
- Yearly sales analysis
- Monthly sales trend
- Sales by region
- Sales by category

### 3. Sales Forecasting
Three forecasting models were implemented:

- SARIMA
- Facebook Prophet
- XGBoost

The models were compared using:

- MAE
- RMSE
- MAPE

**Best Performing Model:** XGBoost

---

### 4. Category & Region Forecasting

Forecasts were generated for:

- Furniture
- Technology
- Office Supplies
- West Region
- East Region

---

### 5. Anomaly Detection

Methods used:

- Isolation Forest
- Z-Score Detection

The project identifies unusually high or low sales weeks.

---

### 6. Product Demand Segmentation

K-Means Clustering was used to group products into demand segments such as:

- High Volume, Stable Demand
- Growing Demand
- Low Volume, High Volatility
- Declining Demand

---

### 7. Interactive Streamlit Dashboard

The dashboard includes:

- 📊 Sales Overview
- 📈 Forecast Explorer
- 🚨 Anomaly Report
- 📦 Product Demand Segments

---

## 📁 Repository Structure

```
SalesForecastApp/
│
├── app.py
├── analysis.ipynb
├── train.csv
├── cluster_results.csv
├── pca_results.csv
├── requirements.txt
├── README.md
├── summary.pdf
└── charts/
    ├── sales_overview.png
    ├── forecast.png
    ├── anomaly_detection.png
    └── product_clusters.png... many more
```

---

## ▶️ Run the Project Locally

Clone the repository:

```bash
git clone https://github.com/garginemade/SalesForecastApp.git
```

Go to the project folder:

```bash
cd SalesForecastApp
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```

---

## 📄 Business Report

A business-friendly executive summary is available in:

- `summary.pdf`

---

## 👩‍💻 Author

**Gargi Nemade**

---

## ⭐ Acknowledgements

- Superstore Sales Dataset
- Streamlit
- Scikit-learn
- XGBoost
- Facebook Prophet
- Statsmodels
