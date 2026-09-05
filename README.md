# 📊 ESG Company Intelligence Platform

**A rigorously benchmarked Machine Learning pipeline that discovers hidden business archetypes from financial and ESG data — and predicts them for new companies in real time.**

[![Live Demo](https://img.shields.io/badge/🚀_Live_Demo-Streamlit-FF4B4B?style=for-the-badge)](https://esg-company-intelligence.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.x-blue?style=for-the-badge&logo=python)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange?style=for-the-badge&logo=scikit-learn)](https://scikit-learn.org/)

🔗 **[Try the Live App →](https://esg-company-intelligence.streamlit.app/)**

---

## 💡 Why This Project

Before investing, analysts juggle a dozen metrics — Revenue, Market Cap, Growth Rate, Profit Margin, and ESG (Environmental, Social, Governance) performance — trying to form a single coherent picture of a company. Doing this manually across thousands of companies is slow and inconsistent.

This project solves that with machine learning: instead of scanning metrics one by one, it groups companies into clear **business personas** and instantly classifies any new company into the persona it matches — giving investors and analysts a fast, data-driven read on what kind of company they're looking at.

What sets this project apart is that **no single algorithm is assumed to be "the best."** Both the clustering stage and the classification stage are built as head-to-head model comparisons, with the winning model chosen on measurable evidence rather than convention.

---

## 🚀 Project Overview

The **ESG Company Intelligence Platform** is a complete, benchmarked data science pipeline covering:

* Data Quality Auditing & Unit-of-Analysis Design
* Feature Engineering (log transforms + standardization)
* **Multi-algorithm clustering comparison** (K-Means, Agglomerative, Gaussian Mixture)
* Multi-metric cluster validation (Silhouette, Calinski-Harabasz, Davies-Bouldin)
* PCA Visualization
* Business-driven cluster profiling & persona naming
* **Multi-algorithm classifier comparison** (6 models, 5-fold cross-validation)
* Held-out test evaluation & feature importance analysis
* **Live Streamlit Deployment**

The underlying dataset spans **11,000 company-year observations across 1,000 unique companies**, covering 9 industries and 7 regions from 2015–2025.

---

## 🎯 Business Questions Answered

* Which companies are high-growth innovators?
* Which companies are environmentally sustainable?
* Which companies are heavy resource consumers?
* Can financial and ESG metrics reveal hidden company archetypes — and can that segmentation be reproduced reliably for new companies?

---

## 🏗 Architecture

<img width="600" height="350" alt="image" src="https://github.com/user-attachments/assets/d33b69aa-c5a7-4ea8-8322-5f22a7312bab" />

---

## 📂 Dataset

| Metric                  | Value       |
| ------------------------ | ----------- |
| Company-year rows         | 11,000      |
| Unique companies           | 1,000       |
| Columns                    | 16          |
| Industries                 | 9           |
| Regions                    | 7           |
| Years Covered               | 2015 – 2025 |
| Rows used for clustering (latest snapshot per company) | 1,000 |

### Features

| Column Name         | Description                          |
| -------------------- | ------------------------------------ |
| CompanyID             | Unique identifier                    |
| CompanyName           | Company name                         |
| Industry               | Industry sector                      |
| Region                 | Geographic region                    |
| Year                    | Reporting year                       |
| Revenue                 | Annual revenue (Million USD)         |
| ProfitMargin            | Net profit margin (%)                |
| MarketCap               | Market capitalization (Million USD)  |
| GrowthRate              | Revenue growth rate (%)              |
| ESG_Overall             | Overall ESG score                    |
| ESG_Environmental       | Environmental score                  |
| ESG_Social              | Social score                         |
| ESG_Governance          | Governance score                     |
| CarbonEmissions         | Carbon emissions (Tons CO₂)          |
| WaterUsage              | Water usage (Cubic Meters)           |
| EnergyConsumption       | Energy consumption (MWh)             |

---

## 🔍 Methodology

**1. Data Quality Audit** — Checked missing values, duplicates, and categorical cardinality. `GrowthRate` was the only column with missing values (9.09%), expected for each company's first reporting year, where no prior year exists to compute growth from.

**2. Unit-of-Analysis Design** — Selected the latest year (2025) per company, collapsing 11,000 company-year rows into 1,000 clean company-level rows — the correct population for a "type of company" segmentation question.

**3. Feature Engineering** — Applied `log1p` transformation to Revenue, MarketCap, CarbonEmissions, WaterUsage, and EnergyConsumption to correct heavy right-skew, then standardized all 8 clustering features with `StandardScaler`.

**4. Clustering Model Comparison** — Rather than assuming K-Means was optimal, three algorithms (K-Means, Agglomerative/Ward, Gaussian Mixture) were each tested across `k = 2..8`, ranked jointly on Silhouette, Calinski-Harabasz, and Davies-Bouldin scores.

**5. PCA Visualization** — Reduced the 8 clustering features to 2 principal components, capturing **67.0% of total variance**, for visual inspection of cluster separation.

**6. Cluster Profiling & Persona Naming** — Business-meaningful persona names were assigned *after* inspecting each cluster's average feature profile — never assumed in advance.

**7. Classifier Model Comparison** — Six classifiers were benchmarked with stratified 5-fold cross-validation on accuracy, weighted F1, precision, and recall, so the "best model" claim is backed by cross-validated evidence, not a single train/test split.

**8. Final Evaluation & Interpretation** — The winning classifier was evaluated on a held-out test set, with a full classification report, confusion matrix, and permutation feature importance.

---

## 📈 Key Results

### Clustering Model Comparison (Top Result per Algorithm)

| Algorithm         | k | Silhouette ↑ | Calinski-Harabasz ↑ | Davies-Bouldin ↓ | Mean Rank ↓ |
| ------------------ | - | ------------ | -------------------- | ------------------ | ------------ |
| **K-Means**         | **4** | **0.2046** | **281.88** | **1.4064** | **2.67** |
| Agglomerative (Ward) | 2 | 0.2099 | 303.14 | 1.6565 | 4.33 |
| Gaussian Mixture     | 3 | 0.1010 | 155.07 | 2.1994 | 15.00 |

**K-Means with k = 4** was selected as the best-ranked configuration overall.

### Cluster Distribution (1,000 companies)

| Persona                     | Companies |
| ----------------------------- | --------: |
| Resource Intensive Giants      |       145 |
| Sustainable Service Firms      |       218 |
| Traditional Operators          |       380 |
| High-Performance Innovators    |       257 |

### Classifier Comparison (Stratified 5-Fold Cross-Validation)

| Model                    | CV Accuracy | CV F1 (Weighted) | CV Precision | CV Recall |
| -------------------------- | ----------: | -----------------: | -------------: | ----------: |
| **Logistic Regression**    | **97.80%**  | **0.9780**          | **0.9783**       | **0.978**   |
| SVM (RBF)                   | 95.80%      | 0.9579              | 0.9591           | 0.958        |
| Extra Trees                 | 94.20%      | 0.9417              | 0.9439           | 0.942        |
| HistGradientBoosting        | 93.40%      | 0.9340              | 0.9359           | 0.934        |
| Gradient Boosting           | 92.70%      | 0.9268              | 0.9298           | 0.927        |
| Random Forest               | 92.40%      | 0.9236              | 0.9266           | 0.924        |

**Logistic Regression** was the strongest performer and was selected as the production classifier.

### Held-Out Test Set Performance

```text
Best Classifier: Logistic Regression
Test Accuracy:          98.00%
Weighted F1-Score:      0.980
Weighted Precision:     0.980
Weighted Recall:        0.980
```

| Class (Persona)                | Precision | Recall | F1-Score | Support |
| -------------------------------- | ----------: | -------: | ---------: | --------: |
| High-Performance Innovators       | 0.98        | 0.98    | 0.98       | 51        |
| Resource Intensive Giants         | 0.97        | 0.97    | 0.97       | 29        |
| Sustainable Service Firms         | 1.00        | 0.98    | 0.99       | 44        |
| Traditional Operators             | 0.97        | 0.99    | 0.98       | 76        |

---

## 🏭 Company Personas

### 🚀 High-Performance Innovators
Highest Growth Rate · Highest Profit Margin · Highest ESG Score among all personas

**Dominant Industries:** Technology, Healthcare, Finance

### 🏭 Resource Intensive Giants
Highest Revenue & Market Cap · Highest Carbon Emissions, Water Usage, and Energy Consumption

**Dominant Industries:** Energy, Manufacturing, Utilities

### 🌱 Sustainable Service Firms
Strong ESG Performance · Lowest Resource Consumption · Moderate Profitability

**Dominant Industries:** Finance, Retail, Healthcare

### 🔧 Traditional Operators
Lower Growth · Lower Profitability · Below-average ESG Performance — the largest and most "average" segment

**Dominant Industries:** Transportation, Utilities, Manufacturing

---

## 🔥 Key Insights

1. **Growth, profitability, and ESG performance move together.** High-Performance Innovators lead on all three simultaneously, contradicting the idea that sustainability comes at the cost of financial performance.
2. **Resource-consumption metrics are the strongest classification signals.** Permutation importance ranks EnergyConsumption (0.169), Revenue (0.142), and MarketCap (0.138) as the top three drivers of persona assignment — ahead of ESG_Overall (0.053) and GrowthRate (0.052).
3. **Simplicity beat complexity for this task.** A linear Logistic Regression model outperformed every tree-based ensemble and SVM tested, achieving the highest cross-validated and held-out accuracy — a useful reminder that model complexity doesn't guarantee better performance on well-separated, standardized feature spaces.
4. **The clustering choice was evidence-based, not assumed.** K-Means only narrowly outranked Agglomerative Clustering on the combined metric ranking (2.67 vs. 4.33) and clearly outperformed Gaussian Mixture Models (15.00), justifying a genuine comparison rather than defaulting to the most common algorithm.

---

## 💻 Live Application

**🔗 [https://esg-company-intelligence.streamlit.app/](https://esg-company-intelligence.streamlit.app/)**

The deployed app lets users:

* Enter company financial metrics
* Enter ESG indicators
* Instantly predict the company's business persona
* View detailed persona descriptions
* Explore underlying business insights

---

## 🛠 Tech Stack

| Category           | Tools                                                                 |
| ------------------- | ---------------------------------------------------------------------- |
| Language             | Python                                                                  |
| Data Analysis        | Pandas, NumPy                                                           |
| Visualization         | Matplotlib, Seaborn                                                     |
| Clustering            | Scikit-Learn — K-Means, Agglomerative Clustering, Gaussian Mixture       |
| Classification         | Scikit-Learn — Logistic Regression, Random Forest, Extra Trees, Gradient Boosting, HistGradientBoosting, SVM |
| Dimensionality Reduction | PCA                                                                   |
| Model Persistence       | Joblib                                                                |
| Deployment              | Streamlit                                                             |

---

## 📁 Project Structure

```text
Company-Segmentation-and-Persona-Prediction/
│
├── data
│
├── ESG_Company_segmentation_model_comparison.ipynb
│
├── app/
│   └── app.py
│
├── models/
│   └── esg_company_segmentation_best_models.joblib
│
├── images/
│   ├── Classification Confidence.png
│   ├── Prediction result interpretation.png
│   ├── UI.png
│   ├── Clustered Company Persona.png
│   └── Feature importance.png
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation & Local Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/KashishPundir/Company-Segmentation-and-Persona-Prediction.git
   ```

2. **Navigate into the project folder**
   ```bash
   cd Company-Segmentation-and-Persona-Prediction
   ```

3. **Verify the trained models are present**
   ```bash
   dir models
   ```
4. **Installed libraries**
   ```bash
   python -m pip install -r requirements.txt
   ```
   
4. **Run the app locally**
   ```bash
   streamlit run app/app.py
   ```

> 💡 Prefer not to install anything? Use the **[live demo](https://esg-company-intelligence.streamlit.app/)** instead.

---

## 👨‍💻 Author

**Kashish Pundir**
B.Tech Computer Science (Data Science)

Passionate about Data Science, Machine Learning, Analytics, and AI-driven decision systems.

📫 [GitHub](https://github.com/KashishPundir) · 🔗 [Live App](https://esg-company-intelligence.streamlit.app/)

---

## ⭐ Support

If you found this project interesting or useful, consider giving it a **star** — it helps a lot!



