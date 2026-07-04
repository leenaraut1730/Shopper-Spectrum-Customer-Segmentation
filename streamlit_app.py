import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px

from src.dashboard import show_dashboard


# ==================================================
# PAGE CONFIG
# ==================================================

st.set_page_config(
    page_title="🛒 Shopper Spectrum",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================================================
# PROFESSIONAL CSS
# ==================================================

st.markdown("""
<style>

[data-testid="stAppViewContainer"]{
    background-color:#0E1117;
}

section[data-testid="stSidebar"]{
    background:#161A23;
}

h1,h2,h3,h4,h5{
    color:white;
}

/* KPI CARD */

.kpi{
background:linear-gradient(135deg,#6C63FF,#00C2FF);
padding:20px;
border-radius:18px;
text-align:center;
color:white;
box-shadow:0px 6px 20px rgba(0,0,0,.35);
transition:0.3s;
}

.kpi:hover{
transform:translateY(-5px);
}

.kpi h2{
font-size:32px;
margin:0;
}

.kpi h3{
margin-bottom:10px;
}

/* Buttons */

.stButton>button{
width:100%;
background:#00C2FF;
color:white;
border-radius:10px;
height:45px;
font-size:17px;
border:none;
}

.stButton>button:hover{
background:#0099cc;
color:white;
}

/* Footer */

.footer{
text-align:center;
color:gray;
padding:20px;
}

</style>
""",unsafe_allow_html=True)

# ==================================================
# LOAD DATA
# ==================================================

@st.cache_data
def load_data():

    df = pd.read_csv("data/cleaned_online_retail.csv")

    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

    df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]

    return df


df = load_data()

# ==================================================
# LOAD MODELS
# ==================================================

kmeans = joblib.load("models/kmeans_model.pkl")

scaler = joblib.load("models/scaler.pkl")

similarity_df = joblib.load("models/similarity_matrix.pkl")

product_list = joblib.load("models/product_list.pkl")

# ==================================================
# SIDEBAR
# ==================================================

st.sidebar.image(
    "images/customer_clusters.png",
    use_container_width=True
)

st.sidebar.title("🛒 Shopper Spectrum")

menu = st.sidebar.radio(
    "📂 Navigation",
    [
        "🏠 Home",
        "📊 Dashboard",
        "👤 Customer Segmentation",
        "📦 Product Recommendation",
         "📋 Project Overview",
    ]
)

st.sidebar.markdown("---")

st.sidebar.markdown("---")

st.sidebar.success("Machine Learning Project")

st.sidebar.info("""

✔ RFM Analysis

✔ KMeans Clustering

✔ Recommendation System

✔ Streamlit Dashboard

""")
# ==================================================
# HOME PAGE
# ==================================================

if menu == "🏠 Home":

    st.markdown("""
    <h1 style='text-align:center;color:#00E5FF;'>
    🛒 Shopper Spectrum
    </h1>

    <h3 style='text-align:center;color:white;'>
    Customer Segmentation & Product Recommendation
    </h3>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # ==============================
    # KPI VALUES
    # ==============================

    revenue = df["TotalPrice"].sum()

    customers = df["CustomerID"].nunique()

    products = df["Description"].nunique()

    countries = df["Country"].nunique()

    orders = df["InvoiceNo"].nunique()

    avg_order = revenue / orders

    # ==============================
    # KPI CARDS
    # ==============================

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        st.markdown(f"""
        <div class='kpi'
        style="background:linear-gradient(135deg,#6C63FF,#8E2DE2);">

        <h3>💰 Revenue</h3>

        <h2>₹{revenue:,.0f}</h2>

        </div>
        """, unsafe_allow_html=True)

    with c2:

        st.markdown(f"""
        <div class='kpi'
        style="background:linear-gradient(135deg,#00C9FF,#92FE9D);">

        <h3>👥 Customers</h3>

        <h2>{customers}</h2>

        </div>
        """, unsafe_allow_html=True)

    with c3:

        st.markdown(f"""
        <div class='kpi'
        style="background:linear-gradient(135deg,#FF512F,#F09819);">

        <h3>📦 Products</h3>

        <h2>{products}</h2>

        </div>
        """, unsafe_allow_html=True)

    with c4:

        st.markdown(f"""
        <div class='kpi'
        style="background:linear-gradient(135deg,#11998E,#38EF7D);">

        <h3>🌍 Countries</h3>

        <h2>{countries}</h2>

        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ==================================
    # SECOND ROW KPI
    # ==================================

    c5, c6 = st.columns(2)

    with c5:

        st.metric(
            "🧾 Total Orders",
            f"{orders:,}"
        )

    with c6:

        st.metric(
            "🛒 Average Order Value",
            f"₹{avg_order:,.2f}"
        )

    st.markdown("---")

    # ==================================
    # PROJECT OVERVIEW
    # ==================================

    left, right = st.columns([2,1])

    with left:

        st.subheader("📌 Project Overview")

        st.write("""

This project analyzes customer purchasing behaviour using Machine Learning.

### Key Features

✅ Customer Segmentation using RFM

✅ KMeans Clustering

✅ Product Recommendation System

✅ Interactive Dashboard

✅ Business Insights

✅ Sales Analytics

        """)

    with right:

        st.image(
            "images/customer_clusters.png",
            use_container_width=True
        )

    st.markdown("---")

    # ==================================
    # FEATURE CARDS
    # ==================================

    f1, f2, f3 = st.columns(3)

    with f1:

        st.success("""
### 📊 Analytics

✔ Sales Dashboard

✔ Country Analysis

✔ Top Products

✔ Revenue Trend
""")

    with f2:

        st.info("""
### 🤖 Machine Learning

✔ RFM Analysis

✔ KMeans Clustering

✔ Prediction

✔ Customer Segments
""")

    with f3:

        st.warning("""
### 🛒 Recommendation

✔ Product Similarity

✔ Cosine Similarity

✔ Top 5 Products

✔ Personalized Suggestions
""")
        # ==================================================
# DASHBOARD
# ==================================================

elif menu == "📊 Dashboard":

    show_dashboard(df)


# ==================================================
# CUSTOMER SEGMENTATION
# ==================================================

elif menu == "👤 Customer Segmentation":

    st.title("👤 Customer Segmentation")

    recency = st.number_input("Recency", min_value=0, value=30)
    frequency = st.number_input("Frequency", min_value=1, value=5)
    monetary = st.number_input("Monetary", min_value=0.0, value=500.0)

    if st.button("Predict Segment"):

        user = np.array([[recency, frequency, monetary]])

        user_scaled = scaler.transform(user)

        cluster = kmeans.predict(user_scaled)[0]

        labels = {
            0: "🌟 High Value",
            1: "✅ Regular",
            2: "🛍️ Occasional",
            3: "⚠️ At Risk"
        }

        st.success(labels.get(cluster, "Unknown"))


# ==================================================
# PRODUCT RECOMMENDATION
# ==================================================

elif menu == "📦 Product Recommendation":

    st.title("📦 Product Recommendation")

    product = st.selectbox(
        "Select Product",
        sorted(product_list)
    )

    if st.button("Get Recommendations"):

        if product in similarity_df.index:

            rec = similarity_df.loc[product]\
                    .sort_values(ascending=False)\
                    .iloc[1:6]

            st.success("Top 5 Recommended Products")

            for item in rec.index:
                st.write("✅", item)

elif menu == "📋 Project Overview":

    st.title("📋 Project Overview & Business Case")

    st.markdown("""
## 📣 Problem Statement

The global e-commerce industry generates vast amounts of transaction data daily, offering valuable insights into customer purchasing behaviors.

Analyzing this data helps businesses identify valuable customers, improve customer experience, and increase overall sales through personalized recommendations.

This project analyzes an online retail dataset to uncover purchasing patterns, perform customer segmentation using **RFM Analysis** and **K-Means Clustering**, and build a **Collaborative Filtering Product Recommendation System**.

---

## 🎯 Business Objectives

- Understand customer buying behavior.
- Identify High-Value customers.
- Improve customer retention.
- Deliver personalized product recommendations.
- Support data-driven marketing decisions.
- Increase customer satisfaction and revenue.

---

## 📌 Real-Time Business Use Cases

🎯 **Targeted Marketing**

- Create marketing campaigns for different customer segments.

🛒 **Personalized Product Recommendations**

- Recommend similar products based on previous purchases.

👥 **Customer Retention**

- Identify At-Risk customers and re-engage them.

📦 **Inventory Management**

- Analyze product demand for better stock planning.

📈 **Business Decision Support**

- Generate insights to improve sales and customer experience.

---

## 📊 Dataset Details

The dataset contains transaction records from a UK-based online retail store.

| Column | Description |
|--------|-------------|
| InvoiceNo | Transaction bill number |
| StockCode | Product code |
| Description | Product name |
| Quantity | Quantity purchased |
| InvoiceDate | Purchase date and time |
| UnitPrice | Price of one unit |
| CustomerID | Unique customer ID |
| Country | Customer country |

---

## ⚙️ Project Workflow

✅ Load Dataset

⬇️

✅ Data Cleaning

- Remove missing CustomerID
- Remove cancelled invoices
- Remove invalid Quantity and UnitPrice
- Remove duplicates

⬇️

✅ Exploratory Data Analysis

- Country-wise Sales
- Top Selling Products
- Monthly Revenue Trend
- Revenue Distribution
- Customer Purchase Behaviour

⬇️

✅ RFM Feature Engineering

- Recency
- Frequency
- Monetary

⬇️

✅ Customer Segmentation

- StandardScaler
- K-Means Clustering
- High-Value Customers
- Regular Customers
- Occasional Customers
- At-Risk Customers

⬇️

✅ Product Recommendation

- Customer-Product Matrix
- Cosine Similarity
- Top-5 Product Recommendations

⬇️

✅ Interactive Streamlit Dashboard

---

## 🛠 Technologies Used

- 🐍 Python
- 📊 Pandas
- 🔢 NumPy
- 📈 Plotly Express
- 📉 Matplotlib
- 🤖 Scikit-Learn
- 📦 Joblib
- 🌐 Streamlit

---

## 🚀 Machine Learning Techniques

- ✔ Data Cleaning
- ✔ Feature Engineering
- ✔ RFM Analysis
- ✔ StandardScaler
- ✔ K-Means Clustering
- ✔ Elbow Method
- ✔ Silhouette Score
- ✔ Collaborative Filtering
- ✔ Cosine Similarity

---

## 💼 Business Impact

✔ Identify High-Value Customers

✔ Improve Customer Retention

✔ Personalized Shopping Experience

✔ Better Marketing Campaigns

✔ Inventory Optimization

✔ Increased Customer Satisfaction

✔ Data-Driven Decision Making

---

### 👩‍💻 Developed By

**Leena Raut**

**Shopper Spectrum**
Customer Segmentation & Product Recommendation in E-Commerce

Built using **Python • Streamlit • Scikit-Learn • Pandas**
""")
