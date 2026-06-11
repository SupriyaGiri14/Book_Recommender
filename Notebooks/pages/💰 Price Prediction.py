import streamlit as st
import pandas as pd
import joblib 
import os

#Displaying image on sidebar
with st.sidebar:
    st.image("https://raw.githubusercontent.com/SupriyaGiri14/Electric_Vehicles_EDA_And_Predictions/refs/heads/main/images/car_steamlit.avif", width=254)

# Page Configuration
st.set_page_config(
    page_title="Electric Car Market  Analytics",
    page_icon="🚗",
    layout="centered"
)

# Get the directory where this script resides
script_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the full path to the model file
model_path = os.path.join(script_dir, "ev_price_model.pkl")

# Load the model
model = joblib.load(model_path)

# main title
st.title("⚡EV Price Prediction⚡")

# ------------------------------
# CUSTOM CSS (FIXED SIDEBAR BUTTONS)
# ------------------------------
st.markdown("""
<style>
/* Sidebar background */
section[data-testid="stSidebar"] {
    background-color:#053827 !important;
}

/* Sidebar button wrapper */
section[data-testid="stSidebar"] div.stButton {
    width: 250px;
}

/* Sidebar buttons full width */
section[data-testid="stSidebar"] div.stButton > button {
    width: 100% !important;
    display: flex !important;
    justify-content: flex-start !important;
    border-radius: 10px;
    margin-bottom: 8px;
    padding-left: 12px;
}

/* Hover effect */
section[data-testid="stSidebar"] div.stButton > button:hover {
    background-color:  #fff9dc;
}

/* Main app background */
.stApp {
    background-color: #fff9dc;
}

/* Headings */
h1, h2, h3, p {
    color: #002366;
}

/* Section title */
.section-title {
    font-size: 24px;
    font-weight: 700;
    margin-top: 35px;
    margin-bottom: 15px;
    color: #002366;
}
            

/* Target the sidebar container background and text */
    [data-testid="stSidebar"] {
        color: white;
    }
    
/* Target text within the sidebar (headers, labels, etc.) */
    [data-testid="stSidebar"] div, [data-testid="stSidebar"] p, [data-testid="stSidebar"] span {
        color: white !important;
    }
            
/* Target links inside the sidebar */
    [data-testid="stSidebar"] a {
        color: white !important;
</style>
""", unsafe_allow_html=True)


st.sidebar.write("---")
csv_path = os.path.join(script_dir, '..', '..', 'Datasets', 'ev_market_2026.csv')

# 3. Read the CSV
df = pd.read_csv(csv_path)

# -------------------------
# USER INPUTS
# -------------------------
brands = sorted(df["brand"].dropna().unique())
brand = st.selectbox("Brand", brands)

market_segment = st.selectbox(
    "Market Segment",
    sorted(df["market_segment"].dropna().unique())
)

drive_type = st.selectbox(
    "Drive Type",
    sorted(df["drive_type"].dropna().unique())
)

country_of_origin = st.selectbox(
    "Country",
    sorted(df["country_of_origin"].dropna().unique())
)

# -------------------------
# AUTO-FILL BASE DATA
# -------------------------
match = df[
    (df["brand"] == brand) &
    (df["market_segment"] == market_segment)
]

if match.empty:
    st.warning("No exact match found. Using brand fallback.")
    match = df[df["brand"] == brand].head(1)

row = match.iloc[0]

# -------------------------
# AUTO FEATURES
# -------------------------
battery_capacity = row["battery_capacity_kwh"]
charging_speed = row["charging_speed_kw"]
range_miles = row["range_miles"]
horsepower = row["horsepower"]

# -------------------------
# DRIVE TYPE EFFECT (optional business logic)
# -------------------------
drive_multiplier = {
    "FWD": 0.95,
    "RWD": 1.00,
    "AWD": 1.10
}

mult = drive_multiplier.get(drive_type, 1.0)

range_miles *= mult
horsepower *= mult


# -------------------------
# DISPLAY
# -------------------------
st.subheader("Auto-filled Vehicle Specs")

col1, col2 = st.columns(2)

with col1:
    st.metric("Battery (kWh)", f"{battery_capacity:.0f}")
    st.metric("Charging Speed (kW)", f"{charging_speed:.0f}")

with col2:
    st.metric("Range (miles)", f"{range_miles:.0f}")
    st.metric("Horsepower", f"{horsepower:.0f}")

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict Price"):

    input_df = pd.DataFrame([{
        "brand": brand,
        "battery_capacity_kwh": battery_capacity,
        "range_miles": range_miles,
        "charging_speed_kw": charging_speed,
        "horsepower": horsepower,
        "drive_type": drive_type,
        "market_segment": market_segment
    }])

    prediction = model.predict(input_df)[0]

    st.success(f"Predicted EV Price for {brand} with above specifications: ${prediction:,.2f}")
    st.warning("Disclaimer: This tool is for educational purposes only. It provides estimated EV price predictions based on a ML model and available data, it does not represent real-world manufacturer pricing.")
    st.subheader("🤖 Machine Learning Model Information")

    st.markdown("""
**Machine Learning model** is used to predict 💰 **Electric Vehicle Prices**


### 🚀 Model Used
✅ **XGBoost Regressor (Extreme Gradient Boosting)**

XGBoost is a highly efficient and powerful machine learning algorithm based on gradient boosting. It builds multiple decision trees sequentially, where each new tree corrects the errors of the previous ones.

### 📌 Features Used for Prediction
The model analyzes various EV specifications such as:

- 🚗 Brand
- 🔋 Battery Capacity
- ⚡ Charging Speed
- 🛣️ Range
- 🏎️ Horsepower
- 🚘 Drive Type
- 🌍 Country of Origin   
- 🏷️ Market Segment

### ⚙️ Machine Learning Workflow
1. Data Preprocessing  
2. Feature Scaling/Encoding  
3. Model Training  
4. Prediction Generation  

### 📖 R2 Score : 0.9549
""")