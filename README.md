# 📊 ESG Company Intelligence Platform

> Before investing in a company, investors and analysts often evaluate key indicators such as **Revenue, Market Capitalization, Growth Rate, Profit Margin, and ESG (Environmental, Social, Governance) performance**. However, analyzing multiple metrics simultaneously can be challenging and time-consuming.
>
> This project addresses that challenge by using Machine Learning to identify the **business persona** of a company based on its financial and ESG characteristics. Instead of looking at individual metrics separately, the platform groups companies into meaningful business archetypes and predicts the most likely persona for new companies.

---

## 🚀 Project Overview

The **ESG Company Intelligence Platform** is an end-to-end Data Science project that combines:

* Data Cleaning & Preprocessing
* Exploratory Data Analysis (EDA)
* Feature Engineering
* K-Means Clustering
* PCA Visualization
* Random Forest Classification
* Streamlit Deployment

The project analyzes **11,000 companies** across multiple industries and geographic regions using financial and ESG metrics.

The objective is to discover hidden company personas and build an intelligent system capable of classifying new companies into those personas.

---

## 🎯 Business Problem

Investors, analysts, and business researchers often ask:

* Which companies are high-growth innovators?
* Which companies are environmentally sustainable?
* Which companies consume large amounts of resources?
* Can financial and ESG metrics reveal hidden company archetypes?

This project uses machine learning to answer these questions.

---

## 🏗 Architecture Diagram

```text
                    ESG Company Dataset
                             │
                             ▼
                  Data Cleaning & Validation
                             │
                             ▼
                    Missing Value Handling
                             │
                             ▼
                      Feature Engineering
                             │
                             ▼
                    Log Transformation
                             │
                             ▼
                     Standardization
                             │
                             ▼
                     K-Means Clustering
                             │
                             ▼
                  Company Persona Discovery
                             │
                             ▼
                      PCA Visualization
                             │
                             ▼
                 Cluster Profiling & Insights
                             │
                             ▼
                Random Forest Classification
                             │
                             ▼
              Company Persona Prediction System
                             │
                             ▼
                    Streamlit Web Application
```

---

## 📂 Dataset Information

### Dataset Size

| Metric        | Value       |
| ------------- | ----------- |
| Rows          | 11,000      |
| Columns       | 16          |
| Industries    | 9           |
| Regions       | 7           |
| Years Covered | 2015 - 2025 |

---

### Features

| Column Name       | Description                         |
| ----------------- | ----------------------------------- |
| CompanyID         | Unique identifier                   |
| CompanyName       | Company name                        |
| Industry          | Industry sector                     |
| Region            | Geographic region                   |
| Year              | Reporting year                      |
| Revenue           | Annual revenue (Million USD)        |
| ProfitMargin      | Net profit margin (%)               |
| MarketCap         | Market capitalization (Million USD) |
| GrowthRate        | Revenue growth rate (%)             |
| ESG_Overall       | Overall ESG score                   |
| ESG_Environmental | Environmental score                 |
| ESG_Social        | Social score                        |
| ESG_Governance    | Governance score                    |
| CarbonEmissions   | Carbon emissions (Tons CO₂)         |
| WaterUsage        | Water usage (Cubic Meters)          |
| EnergyConsumption | Energy consumption (MWh)            |

---

## 🔍 Methodology

### 1. Data Understanding

* Analyzed dataset structure
* Identified missing values
* Examined feature distributions

### 2. Data Cleaning

* Handled missing GrowthRate values
* Removed data inconsistencies
* Prepared data for analysis

### 3. Feature Engineering

Applied logarithmic transformation to:

* Revenue
* MarketCap
* CarbonEmissions
* WaterUsage
* EnergyConsumption

This reduced skewness and improved clustering quality.

### 4. Standardization

Used StandardScaler to normalize features before clustering.

### 5. K-Means Clustering

Discovered hidden company groups using:

* Elbow Method
* Silhouette Score

Selected:

```text
K = 4
```

clusters.

### 6. PCA Visualization

Reduced dimensions from 8 features to 2 principal components for visualization.

### 7. Random Forest Classification

Trained a classifier to predict company personas for new companies.

---

## 📈 Key Results

### Cluster Distribution

| Persona                     | Companies |
| --------------------------- | --------: |
| Resource Intensive Giants   |     1,698 |
| Sustainable Service Firms   |     2,338 |
| Traditional Operators       |     4,378 |
| High-Performance Innovators |     2,586 |

---

## 🏭 Company Personas

### 🏭 Resource Intensive Giants

Characteristics:

* High Revenue
* High Market Cap
* High Carbon Emissions
* High Water Usage
* High Energy Consumption

Dominant Industries:

* Energy
* Manufacturing
* Utilities

---

### 🌱 Sustainable Service Firms

Characteristics:

* Strong ESG Performance
* Low Resource Consumption
* Moderate Profitability

Dominant Industries:

* Finance
* Retail
* Healthcare

---

### 🔧 Traditional Operators

Characteristics:

* Lower Growth
* Lower Profitability
* Average ESG Performance

Dominant Industries:

* Transportation
* Utilities
* Manufacturing

---

### 🚀 High-Performance Innovators

Characteristics:

* Highest Growth Rate
* Highest Profit Margin
* Highest ESG Score

Dominant Industries:

* Technology
* Healthcare
* Finance

---

## 🤖 Machine Learning Models

### K-Means Clustering

Purpose:

* Discover hidden company archetypes

Evaluation:

* Elbow Method
* Silhouette Score

---

### Random Forest Classifier

Purpose:

* Predict company persona for new companies

Performance:

```text
Accuracy: 97.05%
```

---

## 🔥 Key Insights

### Insight 1

High-Performance Innovators exhibited:

* Highest Growth Rate
* Highest Profit Margin
* Highest ESG Score

---

### Insight 2

Resource consumption metrics were more influential than Growth Rate in determining company personas.

Top drivers:

* Energy Consumption
* Market Capitalization
* Carbon Emissions
* Water Usage

---

### Insight 3

Companies with stronger ESG profiles tended to be associated with higher profitability and growth within the discovered clusters.

---

## 💻 Streamlit Application

The deployed application allows users to:

* Enter company financial metrics
* Enter ESG indicators
* Predict company persona
* View persona descriptions
* Explore business insights

---

## 🛠 Tech Stack

### Programming Language

* Python

### Data Analysis

* Pandas
* NumPy

### Visualization

* Matplotlib
* Seaborn

### Machine Learning

* Scikit-Learn
* K-Means
* PCA
* Random Forest

### Deployment

* Streamlit

---

## 📁 Project Structure

```text
ESG-Company-Intelligence-Platform/
│
├── dataset_link
│
├── ESG_Company_Segmentation.ipynb  
│
├── app/
│   └── app.py
│
├── models/
│   ├── company_persona_classifier.pkl
│   └── scaler.pkl
│
├── images/
│   ├── high_performance_innovator.png
│   └── sustainable_service_firm.png
│   ├── cluster_distribution.png
│   └── feature_importance.png
│
└── README.md
```

---

## ⚙ Installation

```bash
git clone https://github.com/KashishPundir/Company-Segmentation-and-Persona-Prediction.git

cd app

streamlit run app.py
```

---

## 👨‍💻 Author

**Kashish Pundir**

B.Tech Computer Science (Data Science)

Passionate about Data Science, Machine Learning, Analytics, and AI-driven decision systems.

---

## ⭐ If you found this project interesting, consider giving it a star!
