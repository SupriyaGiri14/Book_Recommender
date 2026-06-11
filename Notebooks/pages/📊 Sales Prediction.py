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
model_path = os.path.join(script_dir, "ev_sales_model.pkl")

# Load the model
sales_model = joblib.load(model_path)

# main title
st.title("⚡EV Sales Prediction⚡")

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

# ------------------------------
# PAGE ROUTING
# ------------------------------

# -----------------------------
# Base vehicle profiles
# -----------------------------

# -----------------------------
# Prediction


csv_path = os.path.join(script_dir, '..', '..', 'Datasets', 'ev_market_2026.csv')

# 3. Read the CSV
df = pd.read_csv(csv_path)
# -------------------------
# BRAND (dynamic)
# -------------------------
brands = sorted(df["brand"].dropna().unique())

brand = st.selectbox("Brand", brands)

# -------------------------
# MODEL depends on brand
# -------------------------
filtered_models = df[df["brand"] == brand]["model"].dropna().unique()
models = sorted(filtered_models)

model = st.selectbox("Model", models)

# -------------------------
# COUNTRY
# -------------------------
country_of_origin = st.selectbox(
    "Country",
    sorted(df["country_of_origin"].dropna().unique())
)

# -------------------------
# DRIVE TYPE
# (independent like you wanted)
# -------------------------
drive_type = st.selectbox(
    "Drive Type",
    sorted(df["drive_type"].dropna().unique())
)

# -------------------------
# FILTER DATASET (core logic)
# -------------------------
match = df[
    (df["brand"] == brand) &
    (df["model"] == model)
]

# fallback safety
if match.empty:
    st.warning("No exact match found. Showing closest available model.")
    match = df[df["brand"] == brand].head(1)

row = match.iloc[0]

# -------------------------
# AUTO-FILL VALUES
# -------------------------
price_usd = row["price_usd"]
range_miles = row["range_miles"]
horsepower = row["horsepower"]

# optional: drive adjustment logic
drive_multiplier = {
    "FWD": 0.95,
    "RWD": 1.00,
    "AWD": 1.10
}

price_usd *= drive_multiplier.get(drive_type, 1.0)
range_miles *= drive_multiplier.get(drive_type, 1.0)
horsepower *= drive_multiplier.get(drive_type, 1.0)

# -------------------------
# DISPLAY
# -------------------------
st.subheader("Auto-filled Vehicle Specs")

col1, col2 = st.columns(2)

with col1:
    st.metric("Price USD", f"${price_usd:,.0f}")
    st.metric("Range (miles)", f"{range_miles:.0f}")

with col2:
    st.metric("Horsepower", f"{horsepower:.0f}")


# -----------------------------
# Predict
# -----------------------------
if st.button("Predict Sales"):

    input_df = pd.DataFrame([{
        "brand": brand,
        "model": model,
        "price_usd": price_usd,
        "range_miles": range_miles,
        "horsepower": horsepower,
        "drive_type": drive_type,
        "country_of_origin": country_of_origin
    }])

    prediction = sales_model.predict(input_df)[0]

    st.success(f"Predicted Annual Sales for {brand} for above specifications: {prediction:,.0f} units")
    st.warning("Disclaimer: This tool is for educational purposes only. It provides estimated EV Sales predictions based on a ML model and available data, it does not represent real-world manufacturer pricing.")
    st.subheader("🤖 Machine Learning Model Information")

    st.markdown("""
**Machine Learning model** is used to predict 💰 **Electric Vehicles Sales**


### 🚀 Model Used
✅ **Random Forest Regressor**

Random Forest is a supervised machine learning algorithm that combines multiple decision trees to make accurate and reliable predictions.

### 📌 Features Used for Prediction
The model analyzes various EV specifications such as:

- 🚗 Brand
- 🚘 Model 
- 🛣️ Range
- 🏎️ Horsepower
- 🚘 Drive Type
- 🌍 Country of Origin  
- 💰 Price (USD)

### ⚙️ Machine Learning Workflow
1. Data Preprocessing  
2. Feature Scaling/Encoding  
3. Model Training  
4. Prediction Generation  

### 📖 R2 Score : 0.8846
""")