import streamlit as st
import pandas as pd
import joblib

# --- Load trained model ---
MODEL_PATH = '/Users/krutikakatke/Documents/fashion-business-analyst-AI/notebooks/xgb_top_seller_model.pkl'
model = joblib.load(MODEL_PATH)

# --- Page setup ---
st.set_page_config(page_title="Fashion Business Sales Predictor", layout="centered")
st.title("🛍️ Fashion Product Top Seller Predictor")

st.markdown("Predict if your product will be a **Top Seller** based on business features.")

# --- Inputs ---
st.header("📊 Enter Product Details")

# Numeric
UnitCost = st.number_input("Unit Cost", min_value=0.0, step=0.1)
UnitPrice = st.number_input("Unit Price", min_value=0.0, step=0.1)
Discount = st.number_input("Discount (%)", min_value=0.0, max_value=100.0, step=0.1)
UnitsSold = st.number_input("Units Sold", min_value=0, step=1)
CustomerRating = st.slider("Customer Rating (1–5)", 1.0, 5.0, 3.0)
ProfitMargin = st.number_input("Profit Margin (%)", min_value=0.0, max_value=100.0, step=0.1)
Revenue = st.number_input("Revenue", min_value=0.0, step=0.1)

# Categorical
StoreType = st.selectbox("Store Type", ["Online", "Offline"])
ProductCategory = st.selectbox("Product Category", ["Kurti", "Lehenga", "Saree", "Top"])
Material = st.selectbox("Material", ["Cotton", "Georgette", "Linen", "Silk"])
Brand = st.selectbox("Brand", ["FabIndia", "GlobalDesi", "H&M", "Zara"])
Region = st.selectbox("Region", ["North", "South", "West"])
Season = st.selectbox("Season", ["Spring", "Summer", "Winter"])

# --- Prediction ---
if st.button("🔮 Predict Top Seller"):
    input_data = pd.DataFrame([{
        "UnitCost": UnitCost,
        "UnitPrice": UnitPrice,
        "Discount": Discount,
        "UnitsSold": UnitsSold,
        "StoreType": 1 if StoreType == "Online" else 0,  # Example if encoded that way
        "CustomerRating": CustomerRating,
        "ProfitMargin": ProfitMargin,
        "Revenue": Revenue,

        # One-hot encodings as per training data
        "ProductCategory_Kurti": 1 if ProductCategory == "Kurti" else 0,
        "ProductCategory_Lehenga": 1 if ProductCategory == "Lehenga" else 0,
        "ProductCategory_Saree": 1 if ProductCategory == "Saree" else 0,
        "ProductCategory_Top": 1 if ProductCategory == "Top" else 0,

        "Material_Cotton": 1 if Material == "Cotton" else 0,
        "Material_Georgette": 1 if Material == "Georgette" else 0,
        "Material_Linen": 1 if Material == "Linen" else 0,
        "Material_Silk": 1 if Material == "Silk" else 0,

        "Brand_FabIndia": 1 if Brand == "FabIndia" else 0,
        "Brand_GlobalDesi": 1 if Brand == "GlobalDesi" else 0,
        "Brand_H&M": 1 if Brand == "H&M" else 0,
        "Brand_Zara": 1 if Brand == "Zara" else 0,

        "Region_North": 1 if Region == "North" else 0,
        "Region_South": 1 if Region == "South" else 0,
        "Region_West": 1 if Region == "West" else 0,

        "Season_Spring": 1 if Season == "Spring" else 0,
        "Season_Summer": 1 if Season == "Summer" else 0,
        "Season_Winter": 1 if Season == "Winter" else 0
    }])

    # --- Predict ---
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.subheader("🧾 Prediction Result")
    if prediction == 1:
        st.success(f"🎉 Likely a **Top Seller!** (Confidence: {probability:.2%})")
    else:
        st.warning(f"📉 Might **not be a Top Seller.** (Confidence: {probability:.2%})")