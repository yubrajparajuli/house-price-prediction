import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import streamlit as st
import numpy as np
import pandas as pd
from predict import predict_all

st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="wide"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700&family=DM+Sans:wght@300;400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
    }

    .stApp {
        background-color: #0f0f0f;
        color: #f0ece4;
    }

    section[data-testid="stSidebar"] {
        background-color: #1a1a1a;
        border-right: 1px solid #2a2a2a;
    }

    section[data-testid="stSidebar"] * {
        color: #f0ece4 !important;
    }

    .hero {
        padding: 3rem 0 2rem 0;
        border-bottom: 1px solid #2a2a2a;
        margin-bottom: 2rem;
    }

    .hero h1 {
        font-family: 'Playfair Display', serif;
        font-size: 3.5rem;
        font-weight: 700;
        color: #f0ece4;
        line-height: 1.1;
        margin: 0;
    }

    .hero p {
        font-size: 1rem;
        color: #888;
        margin-top: 0.5rem;
        letter-spacing: 0.1em;
        text-transform: uppercase;
    }

    .price-card {
        background: #1a1a1a;
        border: 1px solid #2a2a2a;
        border-radius: 4px;
        padding: 2rem;
        text-align: center;
        transition: border-color 0.3s;
    }

    .price-card:hover {
        border-color: #c9a84c;
    }

    .price-card .label {
        font-size: 0.75rem;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        color: #888;
        margin-bottom: 0.75rem;
    }

    .price-card .value {
        font-family: 'Playfair Display', serif;
        font-size: 2.25rem;
        font-weight: 700;
        color: #c9a84c;
    }

    .price-card .subtext {
        font-size: 0.75rem;
        color: #555;
        margin-top: 0.5rem;
    }

    .info-box {
        background: #1a1a1a;
        border-left: 3px solid #c9a84c;
        padding: 1rem 1.5rem;
        border-radius: 0 4px 4px 0;
        color: #888;
        font-size: 0.9rem;
    }

    .section-title {
        font-family: 'Playfair Display', serif;
        font-size: 1.5rem;
        color: #f0ece4;
        margin-bottom: 1.5rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid #2a2a2a;
    }

    .stButton > button {
        background-color: #c9a84c !important;
        color: #0f0f0f !important;
        border: none !important;
        border-radius: 2px !important;
        font-family: 'DM Sans', sans-serif !important;
        font-weight: 500 !important;
        letter-spacing: 0.1em !important;
        text-transform: uppercase !important;
        padding: 0.6rem 2rem !important;
        width: 100% !important;
        transition: opacity 0.2s !important;
    }

    .stButton > button:hover {
        opacity: 0.85 !important;
    }

    .stNumberInput input, .stSelectbox select {
        background-color: #222 !important;
        color: #f0ece4 !important;
        border: 1px solid #2a2a2a !important;
        border-radius: 2px !important;
    }

    .footer {
        margin-top: 4rem;
        padding-top: 1.5rem;
        border-top: 1px solid #2a2a2a;
        color: #444;
        font-size: 0.8rem;
        letter-spacing: 0.05em;
    }

    .tag {
        display: inline-block;
        background: #1a1a1a;
        border: 1px solid #2a2a2a;
        color: #888;
        font-size: 0.7rem;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        padding: 0.2rem 0.6rem;
        border-radius: 2px;
        margin-right: 0.5rem;
    }

    .derived-box {
        background: #1a1a1a;
        border: 1px solid #2a2a2a;
        border-radius: 4px;
        padding: 1rem 1.5rem;
        margin-top: 1rem;
    }

    .derived-box p {
        color: #888;
        font-size: 0.85rem;
        margin: 0.2rem 0;
    }

    .derived-box span {
        color: #c9a84c;
        font-weight: 500;
    }

    hr {
        border-color: #2a2a2a !important;
    }
</style>
""", unsafe_allow_html=True)

# hero
st.markdown("""
<div class="hero">
    <h1>House Price<br>Predictor</h1>
    <p>King County, Seattle &nbsp;·&nbsp; Linear Regression from Scratch</p>
</div>
""", unsafe_allow_html=True)

# sidebar
st.sidebar.markdown("### Input Features")

bedrooms      = st.sidebar.number_input("Bedrooms", min_value=1, max_value=10, value=3)
bathrooms     = st.sidebar.number_input("Bathrooms", min_value=1.0, max_value=8.0, value=2.0, step=0.25)
sqft_living   = st.sidebar.number_input("Living Area (sqft)", min_value=300, max_value=13000, value=1800)
sqft_lot      = st.sidebar.number_input("Lot Size (sqft)", min_value=500, max_value=100000, value=7000)
floors        = st.sidebar.number_input("Floors", min_value=1.0, max_value=3.5, value=1.0, step=0.5)
waterfront    = st.sidebar.selectbox("Waterfront", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
view          = st.sidebar.number_input("View Quality (0-4)", min_value=0, max_value=4, value=0)
condition     = st.sidebar.number_input("Condition (1-5)", min_value=1, max_value=5, value=3)
grade         = st.sidebar.number_input("Grade (1-13)", min_value=1, max_value=13, value=7)
sqft_basement = st.sidebar.number_input("Basement (sqft)", min_value=0, max_value=5000, value=0)
lat           = st.sidebar.number_input("Latitude", min_value=47.1, max_value=47.8, value=47.5, step=0.01)
long          = st.sidebar.number_input("Longitude", min_value=-122.5, max_value=-121.3, value=-122.2, step=0.01)
sqft_living15 = st.sidebar.number_input("Neighbors Living Area (sqft)", min_value=400, max_value=6000, value=1800)
sale_year     = st.sidebar.selectbox("Sale Year", [2014, 2015])
sale_month    = st.sidebar.number_input("Sale Month", min_value=1, max_value=12, value=6)
house_age     = st.sidebar.number_input("House Age (years)", min_value=0, max_value=115, value=30)
is_renovated  = st.sidebar.selectbox("Renovated", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

total_rooms       = bedrooms + bathrooms
living_area_ratio = sqft_living / sqft_living15 if sqft_living15 > 0 else 1.0

st.sidebar.markdown(f"""
<div class="derived-box">
    <p>Total Rooms &nbsp;<span>{total_rooms}</span></p>
    <p>Living Area Ratio &nbsp;<span>{living_area_ratio:.2f}</span></p>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")
predict_btn = st.sidebar.button("Predict Price", type="primary")

input_dict = {
    'bedrooms': bedrooms, 'bathrooms': bathrooms,
    'sqft_living': sqft_living, 'sqft_lot': sqft_lot,
    'floors': floors, 'waterfront': waterfront,
    'view': view, 'condition': condition, 'grade': grade,
    'sqft_basement': sqft_basement, 'lat': lat, 'long': long,
    'sqft_living15': sqft_living15, 'sale_year': sale_year,
    'sale_month': sale_month, 'house_age': house_age,
    'is_renovated': is_renovated, 'total_rooms': total_rooms,
    'living_area_ratio': living_area_ratio
}

if predict_btn:
    with st.spinner(""):
        results = predict_all(input_dict)

    st.markdown('<p class="section-title">Predicted Price</p>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"""
        <div class="price-card">
            <div class="label">Normal Equation</div>
            <div class="value">${results['normal_equation']:,.0f}</div>
            <div class="subtext">Closed-form solution</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="price-card">
            <div class="label">Gradient Descent</div>
            <div class="value">${results['gradient_descent']:,.0f}</div>
            <div class="subtext">Iterative optimization</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="price-card">
            <div class="label">Sklearn</div>
            <div class="value">${results['sklearn']:,.0f}</div>
            <div class="subtext">Reference implementation</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<p class="section-title">Input Summary</p>', unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    items = list(input_dict.items())
    half  = len(items) // 2

    with col_a:
        df_a = pd.DataFrame(items[:half], columns=['Feature', 'Value'])
        st.dataframe(df_a, width='stretch', hide_index=True)

    with col_b:
        df_b = pd.DataFrame(items[half:], columns=['Feature', 'Value'])
        st.dataframe(df_b, width='stretch', hide_index=True)

else:
    st.markdown("""
    <div class="info-box">
        Configure house features in the sidebar and click <strong>Predict Price</strong> to get estimates from all three models.
    </div>
    """, unsafe_allow_html=True)

st.markdown("""
<div class="footer">
    <span class="tag">Linear Regression</span>
    <span class="tag">From Scratch</span>
    <span class="tag">King County</span>
    &nbsp; Built by <strong>Yubraj Parajuli</strong>
</div>
""", unsafe_allow_html=True)