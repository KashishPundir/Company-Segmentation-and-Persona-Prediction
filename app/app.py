import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="ESG Company Intelligence Platform",
    page_icon="📊",
    layout="wide"
)


# ============================================================
# LOAD BEST MODEL ARTIFACT
# ============================================================

@st.cache_resource
def load_artifacts():

    # Project root
    BASE_DIR = Path(__file__).resolve().parent.parent

    MODEL_PATH = (
        BASE_DIR
        / "models"
        / "esg_company_segmentation_best_models.joblib"
    )

    if not MODEL_PATH.exists():
        st.error(
            f"Model artifact not found:\n{MODEL_PATH}"
        )
        st.stop()

    return joblib.load(MODEL_PATH)


artifacts = load_artifacts()


# ============================================================
# EXTRACT ARTIFACTS
# ============================================================

cluster_features = artifacts["cluster_features"]
log_features = artifacts["log_features"]

scaler = artifacts["scaler"]

clustering_algorithm = artifacts["clustering_algorithm"]
clustering_model = artifacts["clustering_model"]

classifier_name = artifacts["classifier_name"]
classifier = artifacts["classifier"]

cluster_names = artifacts["cluster_names"]


# ============================================================
# PERSONA DESCRIPTIONS
# ============================================================

descriptions = {

    "Resource Intensive Giants":
        """
        Large companies with significant market presence,
        high revenue, and heavy resource consumption.

        These companies tend to have substantial carbon emissions,
        water usage and energy consumption.
        """,

    "Sustainable Service Firms":
        """
        Companies characterized by stronger ESG performance
        and relatively lower resource consumption.

        They tend to operate with comparatively efficient
        environmental footprints.
        """,

    "Traditional Operators":
        """
        Established companies with comparatively lower
        profitability and slower growth.

        Their financial and ESG characteristics represent
        a more traditional operating profile.
        """,

    "High-Performance Innovators":
        """
        Companies showing strong profitability and growth
        together with strong ESG performance.

        These firms represent a combination of financial
        performance and sustainability characteristics.
        """
}


# ============================================================
# HEADER
# ============================================================

st.title("📊 ESG Company Intelligence Platform")

st.markdown(
    """
    **Discover a company's business persona using financial,
    ESG and resource-consumption indicators.**
    """
)

st.caption(
    f"Clustering model: {clustering_algorithm}  |  "
    f"Classifier: {classifier_name}"
)


# ============================================================
# INPUT SECTION
# ============================================================

st.subheader("🏢 Enter Company Metrics")

col1, col2 = st.columns(2)


with col1:

    revenue = st.number_input(
        "Revenue (Million USD)",
        min_value=0.0,
        value=5000.0,
        step=100.0,
        help="Annual company revenue in millions of USD."
    )

    profit_margin = st.number_input(
        "Profit Margin (%)",
        value=10.0,
        step=0.5,
        help="Net profit margin percentage."
    )

    market_cap = st.number_input(
        "Market Cap (Million USD)",
        min_value=0.0,
        value=10000.0,
        step=100.0,
        help="Company market capitalization in millions of USD."
    )

    growth_rate = st.number_input(
        "Growth Rate (%)",
        value=5.0,
        step=0.5,
        help="Year-over-year growth rate."
    )


with col2:

    esg = st.slider(
        "ESG Score",
        min_value=0.0,
        max_value=100.0,
        value=50.0,
        step=0.1,
        help="Overall ESG score."
    )

    carbon = st.number_input(
        "Carbon Emissions (Tons CO₂)",
        min_value=0.0,
        value=300000.0,
        step=10000.0,
        help="Annual carbon emissions."
    )

    water = st.number_input(
        "Water Usage (Cubic Meters)",
        min_value=0.0,
        value=200000.0,
        step=10000.0,
        help="Annual water consumption."
    )

    energy = st.number_input(
        "Energy Consumption (MWh)",
        min_value=0.0,
        value=1200000.0,
        step=10000.0,
        help="Annual energy consumption."
    )


# ============================================================
# PREPROCESSING FUNCTION
# ============================================================

def preprocess_input(data):

    data = data.copy()

    # Apply exactly the same log transformation
    # used during model training.
    for col in log_features:
        data[col] = np.log1p(data[col])

    # Keep the exact feature order used during training.
    data = data[cluster_features]

    # Apply the fitted training scaler.
    data_scaled = scaler.transform(data)

    return data_scaled


# ============================================================
# PREDICTION
# ============================================================

if st.button(
    "🔍 Predict Company Persona",
    use_container_width=True
):

    # --------------------------------------------------------
    # Create raw input DataFrame
    # --------------------------------------------------------

    input_data = pd.DataFrame({
        "Revenue": [revenue],
        "ProfitMargin": [profit_margin],
        "MarketCap": [market_cap],
        "GrowthRate": [growth_rate],
        "ESG_Overall": [esg],
        "CarbonEmissions": [carbon],
        "WaterUsage": [water],
        "EnergyConsumption": [energy]
    })


    # --------------------------------------------------------
    # Same preprocessing as training
    # --------------------------------------------------------

    X_new = preprocess_input(input_data)


    # --------------------------------------------------------
    # Clustering prediction
    # --------------------------------------------------------

    cluster_prediction = clustering_model.predict(X_new)[0]

    cluster_prediction = int(cluster_prediction)

    clustering_persona = cluster_names.get(
        cluster_prediction,
        f"Cluster {cluster_prediction}"
    )


    # --------------------------------------------------------
    # Supervised classifier prediction
    # --------------------------------------------------------

    classifier_prediction = classifier.predict(X_new)[0]

    classifier_prediction = int(classifier_prediction)

    classifier_persona = cluster_names.get(
        classifier_prediction,
        f"Cluster {classifier_prediction}"
    )


    # ========================================================
    # DISPLAY RESULT
    # ========================================================

    st.divider()

    st.subheader("🎯 Prediction Result")


    display_names = {

        "Resource Intensive Giants":
            "🏭 Resource Intensive Giants",

        "Sustainable Service Firms":
            "🌱 Sustainable Service Firms",

        "Traditional Operators":
            "🔧 Traditional Operators",

        "High-Performance Innovators":
            "🚀 High-Performance Innovators"
    }


    st.success(
        f"Predicted Persona: "
        f"{display_names.get(classifier_persona, classifier_persona)}"
    )

    # ========================================================
    # MODEL AGREEMENT
    # ========================================================

    if cluster_prediction == classifier_prediction:

        st.success(
            "✅ Clustering model and classifier agree on "
            "the company segment."
        )

    else:

        st.warning(
            "⚠️ The clustering model and classifier produced "
            "different cluster predictions."
        )


    # ========================================================
    # PERSONA DESCRIPTION
    # ========================================================

    st.subheader("📋 Persona Profile")

    st.info(
        descriptions.get(
            classifier_persona,
            "No description available for this persona."
        )
    )


    # ========================================================
    # INPUT SUMMARY
    # ========================================================

    with st.expander("🔎 View Input Data"):

        st.dataframe(
            input_data,
            use_container_width=True
        )


    # ========================================================
    # CLASSIFIER CONFIDENCE
    # ========================================================

    if hasattr(classifier, "predict_proba"):

        probabilities = classifier.predict_proba(X_new)[0]

        confidence = float(np.max(probabilities))

        st.subheader("📈 Classification Confidence")

        st.progress(
            min(confidence, 1.0)
        )

        st.write(
            f"Classifier confidence: **{confidence:.2%}**"
        )

        probability_df = pd.DataFrame({

            "Cluster": classifier.classes_.astype(int),

            "Probability": probabilities

        })

        probability_df["Persona"] = (
            probability_df["Cluster"]
            .map(cluster_names)
        )

        probability_df = probability_df[
            ["Cluster", "Persona", "Probability"]
        ]

        probability_df["Probability"] = (
            probability_df["Probability"]
            .map(lambda x: f"{x:.2%}")
        )

        with st.expander("View Cluster Probabilities"):

            st.dataframe(
                probability_df,
                use_container_width=True,
                hide_index=True
            )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "ESG Company Intelligence Platform | "
    "Machine Learning Model Selection + Deployment"
)