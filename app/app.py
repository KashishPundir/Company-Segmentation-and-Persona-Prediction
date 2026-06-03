import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -------------------------
# Load Model
# -------------------------
model = joblib.load(
    "models/company_persona_classifier.pkl"
)
scaler = joblib.load(
    "models/scaler.pkl"
)

# -------------------------
# Page Config
# -------------------------
st.set_page_config(
    page_title="ESG Company Intelligence Platform",
    page_icon="📊",
    layout="wide"
)

# -------------------------
# Header
# -------------------------
st.title("📊 ESG Company Intelligence Platform")

st.markdown("""
Predict a company's business persona using
financial and ESG indicators.
""")

# -------------------------
# Input Section
# -------------------------

st.subheader("Enter Company Metrics")

col1, col2 = st.columns(2)

with col1:

    revenue = st.number_input(
        "Revenue (Million USD)",
        min_value=0.0,
        value=5000.0,
        help="Annual revenue in millions of USD"
    )

    profit_margin = st.number_input(
        "Profit Margin (%)",
        value=10.0,
        help="Net profit margin percentage"
    )

    market_cap = st.number_input(
        "Market Cap (Million USD)",
        min_value=0.0,
        value=10000.0,
        help="Market capitalization in millions USD"
    )

    growth_rate = st.number_input(
        "Growth Rate (%)",
        value=5.0,
        help="Year-over-year revenue growth"
    )

with col2:

    esg = st.slider(
        "ESG Score",
        min_value=0.0,
        max_value=100.0,
        value=50.0,
        help="Overall ESG sustainability score"
    )

    carbon = st.number_input(
        "Carbon Emissions (Tons CO₂)",
        min_value=0.0,
        value=300000.0,
        help="Annual carbon emissions"
    )

    water = st.number_input(
        "Water Usage (Cubic Meters)",
        min_value=0.0,
        value=200000.0,
        help="Annual water usage"
    )

    energy = st.number_input(
        "Energy Consumption (MWh)",
        min_value=0.0,
        value=1200000.0,
        help="Annual energy consumption"
    )

# -------------------------
# Create Input Data
# -------------------------

input_data = pd.DataFrame({
    "Revenue": [np.log1p(revenue)],
    "ProfitMargin": [profit_margin],
    "MarketCap": [np.log1p(market_cap)],
    "GrowthRate": [growth_rate],
    "ESG_Overall": [esg],
    "CarbonEmissions": [np.log1p(carbon)],
    "WaterUsage": [np.log1p(water)],
    "EnergyConsumption": [np.log1p(energy)]
})

# -------------------------
# Prediction
# -------------------------

if st.button("🔍 Predict Company Persona"):

    prediction = model.predict(input_data)[0]

    cluster_names = {
        0: "Resource Intensive Giants",
        1: "Sustainable Service Firms",
        2: "Traditional Operators",
        3: "High-Performance Innovators"
    }

    persona = cluster_names[prediction]

    display_names = {
        "Resource Intensive Giants": "🏭 Resource Intensive Giants",
        "Sustainable Service Firms": "🌱 Sustainable Service Firms",
        "Traditional Operators": "🔧 Traditional Operators",
        "High-Performance Innovators": "🚀 High-Performance Innovators"
    }

    st.success(
        f"Predicted Persona: {display_names[persona]}"
    )

    descriptions = {

        "Resource Intensive Giants":
        """
        Large companies with significant market presence,
        high revenue, and heavy resource consumption.
        Common in Energy, Manufacturing and Utilities sectors.
        """,

        "Sustainable Service Firms":
        """
        Environmentally efficient firms with strong ESG
        performance and relatively low resource usage.
        Often found in Finance and Retail industries.
        """,

        "Traditional Operators":
        """
        Established businesses with moderate performance,
        lower profitability and slower growth.
        Common in Transportation and Manufacturing sectors.
        """,

        "High-Performance Innovators":
        """
        High-growth, highly profitable firms with strong
        ESG performance.
        Frequently found in Technology and Healthcare sectors.
        """
    }

    st.info(descriptions[persona])

    
