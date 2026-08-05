import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# Set Page Config
st.set_page_config(
    page_title="California Housing Intelligence",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# DATA LOADING & MODEL TRAINING
# -----------------------------------------------------------------------------
def remove_outliers(df: pd.DataFrame, col: str) -> pd.DataFrame:
    stats = df[col].describe()
    q25, q75 = stats['25%'], stats['75%']
    iqr = q75 - q25
    lower_bound = q25 - 1.5 * iqr
    upper_bound = q75 + 1.5 * iqr
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    return df[~df[col].isin(outliers[col])]

@st.cache_resource
def train_housing_model():
    df = pd.read_csv('housing.csv')
    df_clean = remove_outliers(df.copy(), "total_rooms")

    # Impute missing values
    imputer = SimpleImputer(missing_values=np.nan, strategy="median")
    df_clean.iloc[:, 4:5] = imputer.fit_transform(df_clean.iloc[:, 4:5])

    # Encode ocean_proximity
    label_encoder = LabelEncoder()
    df_clean["ocean_proximity"] = label_encoder.fit_transform(df_clean["ocean_proximity"])

    X = df_clean.drop("median_house_value", axis=1)
    y = df_clean["median_house_value"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=101
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_scaled, y_train)

    predictions = model.predict(X_test_scaled)
    
    return df_clean, model, scaler, label_encoder, X_test, y_test, predictions

df_housing, model, scaler, label_encoder, X_test, y_test, predictions = train_housing_model()

# -----------------------------------------------------------------------------
# SIDEBAR NAVIGATION
# -----------------------------------------------------------------------------
st.sidebar.title("📌 Navigation")
page = st.sidebar.radio(
    "Go to page:",
    ["🏡 Analytics Dashboard", "🔮 House Price Predictor", "ℹ️ About Project"]
)

st.sidebar.markdown("---")
st.sidebar.info("💡 **Tip**: Switch between pages to explore analytics or test predictions in real-time.")

# -----------------------------------------------------------------------------
# PAGE 1: ANALYTICS DASHBOARD
# -----------------------------------------------------------------------------
if page == "🏡 Analytics Dashboard":
    st.title("🏡 California Housing Analytics Dashboard")
    st.markdown("Interactive visualizations built with Plotly to analyze California housing prices.")

    # Performance Metrics
    st.subheader("📈 Model Metrics")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Intercept", f"{model.intercept_:.2f}")
    c2.metric("MAE", f"${metrics.mean_absolute_error(y_test, predictions):,.2f}")
    c3.metric("MSE", f"{metrics.mean_squared_error(y_test, predictions):,.2f}")
    c4.metric("RMSE", f"${np.sqrt(metrics.mean_squared_error(y_test, predictions)):,.2f}")

    st.divider()

    # Visualizations
    st.subheader("🗺️ Geographic Price Distribution")
    fig_map = px.scatter_mapbox(
        df_housing.sample(n=min(3000, len(df_housing)), random_state=42),
        lat="latitude",
        lon="longitude",
        color="median_house_value",
        size="population",
        color_continuous_scale=px.colors.cyclical.IceFire,
        size_max=15,
        zoom=5,
        mapbox_style="carto-positron",
        title="California Housing Map (Sampled)"
    )
    st.plotly_chart(fig_map, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🔥 Correlation Heatmap")
        corr_matrix = df_housing.corr().round(2)
        fig_corr = px.imshow(
            corr_matrix,
            text_auto=True,
            aspect="auto",
            color_continuous_scale="Viridis"
        )
        st.plotly_chart(fig_corr, use_container_width=True)

    with col2:
        st.subheader("📉 Residuals Distribution")
        residuals = y_test - predictions
        fig_hist = px.histogram(
            residuals,
            nbins=50,
            labels={'value': 'Residual Amount ($)'},
            marginal="rug"
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    st.subheader("🎯 Actual vs Predicted House Values")
    df_compare = pd.DataFrame({'Actual': y_test, 'Predicted': predictions}).reset_index(drop=True)
    fig_reg = px.scatter(
        df_compare,
        x='Actual',
        y='Predicted',
        trendline="ols",
        trendline_color_override="red",
        opacity=0.4
    )
    st.plotly_chart(fig_reg, use_container_width=True)

# -----------------------------------------------------------------------------
# PAGE 2: PRICE PREDICTOR FORM
# -----------------------------------------------------------------------------
elif page == "🔮 House Price Predictor":
    st.title("🔮 Estimate Your House Value")
    st.write("Fill in the house specifications below to get an estimated market price.")

    with st.form(key="prediction_form"):
        st.markdown("### 📍 Location & Proximity")
        col1, col2, col3 = st.columns(3)
        with col1:
            longitude = st.number_input("Longitude", value=-122.23, step=0.01, format="%.2f")
        with col2:
            latitude = st.number_input("Latitude", value=37.88, step=0.01, format="%.2f")
        with col3:
            ocean_prox = st.selectbox(
                "Ocean Proximity",
                options=["<1H OCEAN", "INLAND", "ISLAND", "NEAR BAY", "NEAR OCEAN"]
            )

        st.markdown("### 🏠 Property & Neighborhood Features")
        col4, col5, col6 = st.columns(3)
        with col4:
            housing_median_age = st.slider("House Median Age (years)", 1, 52, 25)
        with col5:
            total_rooms = st.number_input("Total Rooms", value=2000, step=10)
        with col6:
            total_bedrooms = st.number_input("Total Bedrooms", value=400, step=5)

        col7, col8, col9 = st.columns(3)
        with col7:
            population = st.number_input("Population Density", value=1200, step=50)
        with col8:
            households = st.number_input("Households Count", value=380, step=10)
        with col9:
            median_income = st.slider("Median Income ($10,000s)", 0.5, 15.0, 3.8, step=0.1)

        submit_btn = st.form_submit_button("🚀 Calculate Estimated Value", use_container_width=True)

    if submit_btn:
        # Map selected ocean proximity back to numerical label
        ocean_encoded = label_encoder.transform([ocean_prox])[0]

        input_data = pd.DataFrame([[
            longitude, latitude, housing_median_age, total_rooms,
            total_bedrooms, population, households, median_income, ocean_encoded
        ]], columns=[
            "longitude", "latitude", "housing_median_age", "total_rooms",
            "total_bedrooms", "population", "households", "median_income", "ocean_proximity"
        ])

        # Scale input & Predict
        scaled_input = scaler.transform(input_data)
        predicted_value = model.predict(scaled_input)[0]

        st.success("✨ Prediction Complete!")
        
        # Display Result Metric & Gauge Chart
        res_col1, res_col2 = st.columns([1, 2])
        
        with res_col1:
            st.metric(
                label="Estimated Median House Value", 
                value=f"${max(0, predicted_value):,.2f}"
            )
            st.write("**Entered Specifications:**")
            st.json({
                "Location": f"({latitude}, {longitude})",
                "Ocean Proximity": ocean_prox,
                "House Age": f"{housing_median_age} yrs",
                "Income Level": f"${median_income * 10000:,.0f}"
            })

        with res_col2:
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=max(0, predicted_value),
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Predicted Price vs State Max"},
                gauge={
                    'axis': {'range': [None, 500000]},
                    'bar': {'color': "#2E7D32"},
                    'steps': [
                        {'range': [0, 150000], 'color': "#ECEFF1"},
                        {'range': [150000, 350000], 'color': "#CFD8DC"},
                        {'range': [350000, 500000], 'color': "#B0BEC5"}
                    ]
                }
            ))
            fig_gauge.update_layout(height=300)
            st.plotly_chart(fig_gauge, use_container_width=True)

# -----------------------------------------------------------------------------
# PAGE 3: ABOUT PAGE
# -----------------------------------------------------------------------------
elif page == "ℹ️ About Project":
    st.title("ℹ️ About This Application")
    
    st.markdown("""
    ### 🎯 Overview
    This web application provides predictive analytics and interactive insights into **California Housing Prices**. 
    It is designed to replace traditional static machine learning workflows with fully dynamic visual dashboards and an instant price prediction engine.

    ---

    ### 🛠️ Tech Stack & Libraries
    * **Streamlit**: Web application framework.
    * **Plotly Express & Graph Objects**: Interactive plots, map rendering, and custom gauge indicators.
    * **Scikit-Learn**: Machine learning pipeline (Linear Regression, StandardScaler, SimpleImputer).
    * **Pandas & NumPy**: Data processing and statistical manipulation.

    ---

    ### 📊 Dataset Details
    * **Source**: 1990 California Census dataset (`housing.csv`).
    * **Total Records**: ~20,640 entries.
    * **Target Feature**: `median_house_value` ($).
    * **Preprocessing Pipeline**: Outlier filtering via $1.5 \times \text{IQR}$ rule, median missing value imputation, and label encoding for ocean proximity categories.

    ---

    ### 👨‍💻 Model Specs
    | Parameter | Value |
    | :--- | :--- |
    | **Algorithm** | Multiple Linear Regression |
    | **Train / Test Split** | 60% Train / 40% Test |
    | **Scaling Method** | `StandardScaler` |
    """)