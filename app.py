import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib

# --- App Configuration ---
st.set_page_config(page_title="Nepal Crop Yield Prediction", page_icon="🌾", layout="wide")

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
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Data Exploration", "Model Performance", "Yield Prediction"])

# --- PAGE 1: Home ---
if page == "Home":
    st.title("🌾 Cereal Crop Yield Prediction in Nepal")
    st.markdown("""
    Welcome to the Cereal Crop Yield Prediction App! 
    This application leverages machine learning to estimate agricultural yields across various districts of Nepal based on historical data, weather parameters, and soil factors.
    
    ### Key Features:
    - **Data Exploration**: Investigate the historical dataset and crop distributions.
    - **Model Performance**: Compare how different machine learning models performed.
    - **Yield Prediction**: Input custom climatic and geographic parameters to predict potential crop yield (kg/ha).
    """)
    st.image("crop_yield.jpg", width="stretch")

# --- PAGE 2: Data Exploration ---

elif page == "Data Exploration":
    st.title("📊 Data Exploration & Insights")
    
    st.subheader("Cleaned Dataset Overview")
    st.dataframe(df_ml.head())
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Yield Distribution by Crop Type")
        fig1 = px.box(df_ml, x="crop_type", y="yield_kg/ha", color="crop_type")
        st.plotly_chart(fig1, use_container_width=True)
        
    with col2:
        st.subheader("Average Temperature vs Yield")
        fig2 = px.scatter(df_ml, x="avg_temp_C", y="yield_kg/ha", color="crop_type", opacity=0.6)
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Key Data Insights")
    st.dataframe(df_insights)

# --- PAGE 3: Model Performance ---
elif page == "Model Performance":
    st.title("📈 Machine Learning Model Performance")
    
    st.markdown("We trained and evaluated several machine learning models to determine the best predictor for crop yields.")
    
    # Display results table
    st.dataframe(df_res.style.highlight_min(subset=['RMSE', 'MAE'], color='lightgreen').highlight_max(subset=['R2 Score'], color='lightgreen'))
    
    st.subheader("R2 Score Comparison")
    if 'Model' in df_res.columns and 'R2 Score' in df_res.columns:
        fig_r2 = px.bar(df_res, x='Model', y='R2 Score', color='Model', title="Model Accuracy (R2 Score)")
        st.plotly_chart(fig_r2, use_container_width=True)
    else:
        st.warning("Could not find 'Model' and 'R2 Score' columns in model_results.csv")

# --- PAGE 4: Yield Prediction ---
elif page == "Yield Prediction":
    st.title("🔮 Predict Crop Yield")
    st.markdown("Enter the agricultural and climatic parameters below to estimate the expected crop yield (kg/ha).")
    
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
        col1, col2, col3 = st.columns(3)
        
        with col1:
            districts = df_ml['Districts'].unique().tolist()
            district = st.selectbox("District", sorted(districts))
            crop_types = df_ml['crop_type'].unique().tolist()
            crop_type = st.selectbox("Crop Type", sorted(crop_types))
            year = st.number_input("Year (B.S.)", min_value=2060, max_value=2100, value=2080)
            area = st.number_input("Cultivated Area (Hectares)", min_value=0.1, value=100.0)
            fertilizer = st.number_input("Fertilizer (MT)", min_value=0.0, value=50.0)
            
        with col2:
            avg_temp = st.number_input("Avg Temperature (°C)", value=22.0)
            max_temp = st.number_input("Max Temperature (°C)", value=30.0)
            min_temp = st.number_input("Min Temperature (°C)", value=15.0)
            humidity = st.number_input("Avg Relative Humidity (%)", value=65.0)
            
        with col3:
            rainfall = st.number_input("Avg Rainfall (mm/year)", value=1500.0)
            solar_rad = st.number_input("Solar Radiation (kWh/m2)", value=1800.0)
            par = st.number_input("Total PAR (MJ/m2)", value=2800.0)
            wind_speed = st.number_input("Wind Speed (m/s)", value=2.5)
            ph_value = st.number_input("Soil pH Value", value=6.5)
            
        submit = st.form_submit_button("Predict Yield")
        
    if submit:
        try:
            # 1. Create Input DataFrame
            input_data = pd.DataFrame({
                'Year': [year],
                'Districts': [district],
                'Area': [area],
                'crop_type': [crop_type],
                'avg_temp_C': [avg_temp],
                'max_temp_C': [max_temp],
                'min_temp_C': [min_temp],
                'avg_relative_humidity': [humidity],
                'avg_rainfall_mm_per_year': [rainfall],
                'total_solar_radiation_kWh/m2': [solar_rad],
                'total_PAR_MJ/m2': [par],
                'avg_wind_speed_m/s': [wind_speed],
                'avg_pH_value': [ph_value],
                'fertilizer_in_MT': [fertilizer]
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
            
            # Drop high VIF columns
            final_df.drop(columns=high_vif_cols, inplace=True, errors='ignore')

            # 5. Scaling (only numerical columns that survived VIF)
            final_df[numerical_cols] = scaler.transform(final_df[numerical_cols])

            # 6. Feature Selection
            final_X = selector.transform(final_df)

            # 7. Prediction
            prediction = model.predict(final_X)[0]

            # Display Result
            st.success(f"### 🌱 Estimated Crop Yield: **{prediction:.2f} kg/ha**")
            
            # Additional UI feedback
            col1, col2 = st.columns(2)
            col1.metric("Predicted Yield", f"{prediction:.2f} kg/ha")
            col2.metric("Target Accuracy (R2)", f"{df_res.iloc[0]['R2 Score']:.4f}" if not df_res.empty else "N/A")

        except Exception as e:
            st.error(f"Prediction error: {e}")
            st.info("Ensure all inputs are valid and match the model requirements.")
