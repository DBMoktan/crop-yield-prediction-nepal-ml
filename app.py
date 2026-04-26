import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
import os

# --- App Configuration ---
st.set_page_config(page_title="Nepal Crop Yield Prediction", page_icon="🌾", layout="wide", initial_sidebar_state="expanded")

# --- Custom CSS for Styling ---
st.markdown("""
    <style>
    /* Main container styling */
    .main {
        background-color: #FAFAFA;
    }
    
    /* Headings */
    h1, h2, h3 {
        color: #2E7D32;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* Result Card Styling */
    .result-card {
        background: linear-gradient(135deg, #43A047, #1B5E20);
        color: white;
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 20px 0;
    }
    .result-value {
        font-size: 48px;
        font-weight: bold;
        margin: 10px 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #E8F5E9;
    }
    
    /* Buttons */
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 10px 24px;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #45a049;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        transform: scale(1.02);
    }
    </style>
""", unsafe_allow_html=True)

# --- Load Data ---
@st.cache_data
def load_data():
    df_ml = pd.read_csv('cleaned_dataset_for_ml.csv')
    df_res = pd.read_csv('model_results.csv')
    df_insights = pd.read_csv('data_insights.csv')
    return df_ml, df_res, df_insights

try:
    df_ml, df_res, df_insights = load_data()
except Exception as e:
    st.error(f"Error loading data. Make sure all CSV files are in the same directory. {e}")
    st.stop()

# --- Sidebar Navigation ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/188/188333.png", width=100) # Optional agriculture icon
    st.title("🌾 Navigation")
    page = st.radio("Select a Module:", [
        "🏠 Home", 
        "📊 Data Exploration", 
        "📈 Model Performance", 
        "🔮 Yield Prediction",
        "🧠 Model Explainability (SHAP)"
    ])
    st.markdown("---")
    st.markdown("**About:**\nThis app uses machine learning to estimate crop yields in Nepal based on climatic and soil data.")

# --- PAGE 1: Home ---
if page == "🏠 Home":
    st.title("🌾 Cereal Crop Yield Prediction in Nepal")
    st.markdown("""
    ### Welcome to the Agricultural Intelligence Platform!
    This application leverages advanced machine learning (Stacking Regressor) to estimate agricultural yields across various districts of Nepal based on historical data, weather parameters, and soil factors.
    
    **What you can do here:**
    - 📊 **Explore Data**: Uncover historical trends and distributions of crop yields.
    - 📈 **Review Models**: See how different algorithms stacked up against each other.
    - 🔮 **Predict Yields**: Use our interactive calculator to estimate crop yields based on your specific conditions.
    - 🧠 **Understand the AI**: Look under the hood to see exactly *why* the model makes its predictions using SHAP analysis.
    """)
    if os.path.exists("crop_yield.jpg"):
        st.image("crop_yield.jpg", use_container_width=True)

# --- PAGE 2: Data Exploration ---
elif page == "📊 Data Exploration":
    st.title("📊 Data Exploration & Insights")
    
    st.markdown("Dive into the cleaned dataset used to train our models.")
    st.dataframe(df_ml.head(), use_container_width=True)
    
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Yield Distribution by Crop Type")
        fig1 = px.box(df_ml, x="crop_type", y="yield_kg/ha", color="crop_type", template="plotly_white")
        fig1.update_layout(showlegend=False)
        st.plotly_chart(fig1, use_container_width=True)
        
    with col2:
        st.subheader("Average Temperature vs Yield")
        fig2 = px.scatter(df_ml, x="avg_temp_C", y="yield_kg/ha", color="crop_type", opacity=0.6, template="plotly_white")
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    st.subheader("Key Data Insights")
    st.dataframe(df_insights, use_container_width=True)

# --- PAGE 3: Model Performance ---
elif page == "📈 Model Performance":
    st.title("📈 Machine Learning Model Performance")
    
    st.markdown("We trained and evaluated several machine learning models to determine the best predictor for crop yields. A **Stacking Regressor** (combining XGBoost, LightGBM, Random Forest, and Ridge) provided the highest accuracy.")
    
    # Display results table
    st.dataframe(df_res.style.highlight_min(subset=['RMSE', 'MAE'], color='#C8E6C9').highlight_max(subset=['R2 Score'], color='#C8E6C9'), use_container_width=True)
    
    st.markdown("---")
    st.subheader("Accuracy Comparison (R² Score)")
    if 'Model' in df_res.columns and 'R2 Score' in df_res.columns:
        fig_r2 = px.bar(df_res, x='Model', y='R2 Score', color='Model', title="Higher is better", template="plotly_white")
        st.plotly_chart(fig_r2, use_container_width=True)
    else:
        st.warning("Could not find 'Model' and 'R2 Score' columns in model_results.csv")

# --- PAGE 4: Yield Prediction ---
elif page == "🔮 Yield Prediction":
    st.title("🔮 Interactive Yield Predictor")
    st.markdown("Adjust the geographic, climatic, and agricultural inputs below to dynamically estimate the expected crop yield in **kg/ha**.")
    
    # Load model and preprocessors
    try:
        model = joblib.load("best_model.pkl")
        encoder = joblib.load("encoder.pkl")
        scaler = joblib.load("scaler.pkl")
        selector = joblib.load("selector.pkl")
        high_vif_cols = joblib.load("high_vif_cols.pkl")
        min_year = joblib.load("min_year.pkl")
        numerical_cols = joblib.load("numerical_cols.pkl")
    except Exception as e:
        st.error(f"Error loading model or preprocessors: {e}")
        st.stop()

    # Input Form
    with st.form("prediction_form"):
        # Group 1: Geography & Crop
        st.markdown("### 📍 Geographic & Crop Selection")
        col1, col2, col3 = st.columns(3)
        with col1:
            districts = df_ml['Districts'].unique().tolist()
            district = st.selectbox("District", sorted(districts))
        with col2:
            crop_types = df_ml['crop_type'].unique().tolist()
            crop_type = st.selectbox("Crop Type", sorted(crop_types))
        with col3:
            year = st.slider("Year (B.S.)", min_value=2060, max_value=2100, value=2080)
            
        st.markdown("---")
            
        # Group 2: Farm Specifics
        st.markdown("### 🚜 Farm Details")
        col4, col5, col6 = st.columns(3)
        with col4:
            area = st.number_input("Cultivated Area (Hectares)", min_value=0.1, value=100.0, step=10.0)
        with col5:
            fertilizer = st.number_input("Fertilizer Applied (MT)", min_value=0.0, value=50.0, step=5.0)
        with col6:
            ph_value = st.slider("Soil pH Value", min_value=3.0, max_value=10.0, value=6.5, step=0.1)

        st.markdown("---")

        # Group 3: Climatic Conditions
        st.markdown("### 🌤️ Climatic Conditions")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            avg_temp = st.slider("Avg Temp (°C)", 0.0, 45.0, 22.0)
            max_temp = st.slider("Max Temp (°C)", 0.0, 50.0, 30.0)
        with c2:
            min_temp = st.slider("Min Temp (°C)", -10.0, 40.0, 15.0)
            humidity = st.slider("Avg Humidity (%)", 0.0, 100.0, 65.0)
        with c3:
            rainfall = st.number_input("Avg Rainfall (mm/yr)", value=1500.0, step=100.0)
            wind_speed = st.slider("Wind Speed (m/s)", 0.0, 15.0, 2.5)
        with c4:
            solar_rad = st.number_input("Solar Rad. (kWh/m2)", value=1800.0, step=100.0)
            par = st.number_input("Total PAR (MJ/m2)", value=2800.0, step=100.0)
            
        st.markdown("<br>", unsafe_allow_html=True)
        submit = st.form_submit_button("🚀 Calculate Yield", use_container_width=True)
        
    if submit:
        try:
            with st.spinner('Analyzing inputs and computing prediction...'):
                # 1. Create Input DataFrame
                input_data = pd.DataFrame({
                    'Year': [year], 'Districts': [district], 'Area': [area], 'crop_type': [crop_type],
                    'avg_temp_C': [avg_temp], 'max_temp_C': [max_temp], 'min_temp_C': [min_temp],
                    'avg_relative_humidity': [humidity], 'avg_rainfall_mm_per_year': [rainfall],
                    'total_solar_radiation_kWh/m2': [solar_rad], 'total_PAR_MJ/m2': [par],
                    'avg_wind_speed_m/s': [wind_speed], 'avg_pH_value': [ph_value], 'fertilizer_in_MT': [fertilizer]
                })

                # 2. Feature Engineering
                epsilon = 1e-9
                input_data['temp_range'] = input_data['max_temp_C'] - input_data['min_temp_C']
                input_data['rainfall_per_area'] = input_data['avg_rainfall_mm_per_year'] / (input_data['Area'] + epsilon)
                input_data['fertilizer_per_area'] = input_data['fertilizer_in_MT'] / (input_data['Area'] + epsilon)
                input_data['solar_radiation_per_area'] = input_data['total_solar_radiation_kWh/m2'] / (input_data['Area'] + epsilon)
                input_data['par_per_area'] = input_data['total_PAR_MJ/m2'] / (input_data['Area'] + epsilon)
                input_data['temp_rainfall_interaction'] = input_data['avg_temp_C'] * input_data['avg_rainfall_mm_per_year']
                input_data['log_Area'] = np.log1p(input_data['Area'])
                input_data['log_fertilizer_in_MT'] = np.log1p(input_data['fertilizer_in_MT'])
                input_data['years_since_start'] = input_data['Year'] - min_year

                # 3. Categorical Encoding
                categorical_cols = ['Districts', 'crop_type']
                encoded_cats = encoder.transform(input_data[categorical_cols])
                encoded_cat_cols = encoder.get_feature_names_out(categorical_cols)
                encoded_df = pd.DataFrame(encoded_cats, columns=encoded_cat_cols)

                # 4. Combine and Drop columns
                input_num = input_data.drop(columns=categorical_cols)
                final_df = pd.concat([input_num, encoded_df], axis=1)
                final_df.drop(columns=high_vif_cols, inplace=True, errors='ignore')

                # 5. Scaling
                final_df[numerical_cols] = scaler.transform(final_df[numerical_cols])

                # 6. Feature Selection
                final_X = selector.transform(final_df)

                # 7. Prediction
                prediction = model.predict(final_X)[0]

            # Display Result nicely
            st.markdown(f"""
                <div class="result-card">
                    <h3>Estimated Crop Yield</h3>
                    <div class="result-value">{prediction:,.2f}</div>
                    <p>Kilograms per Hectare (kg/ha)</p>
                </div>
            """, unsafe_allow_html=True)
            
            st.success("✅ Prediction calculated successfully using the AI Model!")

        except Exception as e:
            st.error(f"Prediction error: {e}")
            st.info("Ensure all inputs are valid and match the model requirements.")

# --- PAGE 5: Model Explainability ---
elif page == "🧠 Model Explainability (SHAP)":
    st.title("🧠 Understanding the AI Decision Process")
    st.markdown("""
    Machine Learning models are often considered "black boxes." **SHAP (SHapley Additive exPlanations)** is a game-theoretic approach to explain the output of any machine learning model. 
    Here, we break down exactly *why* our Stacking Regressor predicts what it does.
    """)
    
    st.markdown("---")
    
    col1, col2 = st.columns([1.5, 1])
    
    with col1:
        st.subheader("Global Feature Importance")
        try:
            shap_imp = pd.read_csv("plots/shap_importance.csv")
            # Create a nice horizontal bar chart
            fig_shap = px.bar(
                shap_imp.head(10).sort_values('importance', ascending=True), 
                x="importance", 
                y="feature", 
                orientation='h',
                title="Top 10 Most Influential Features",
                labels={"importance": "Mean Absolute SHAP Value (Impact on Yield)", "feature": "Feature"},
                color="importance",
                color_continuous_scale="Viridis",
                template="plotly_white"
            )
            st.plotly_chart(fig_shap, use_container_width=True)
        except Exception as e:
            st.warning("Could not load SHAP importance data. Please run the SHAP analysis script first.")
            
    with col2:
        st.subheader("Key Takeaways")
        st.info("""
        **1. Land Area Dominates:** The raw area of cultivation heavily dictates the scale of the yield outcome.
        
        **2. Crop Type Impact:** Distinct crops have distinct baseline yields. Paddy is highly water-intensive and produces high yields, pushing predictions up. Millet pulls predictions down.
        
        **3. Critical Environmentals:** Soil pH and Wind Speed are massive predictors, penalizing yields if they deviate from optimal levels.
        
        **4. Fertilizer Context:** The model intelligently weighs both the raw fertilizer amount and its intensity (fertilizer per area).
        """)
        
    st.markdown("---")
    st.subheader("Detailed SHAP Summary Plot")
    st.markdown("""
    This plot shows the distribution of the impacts each feature has on the model output. 
    * **Color** represents the feature value (Red = High, Blue = Low).
    * **X-axis** shows whether that value pushed the predicted yield higher or lower.
    """)
    try:
        st.image("plots/shap_summary.png", use_container_width=True)
    except:
        st.warning("SHAP summary plot image not found.")
