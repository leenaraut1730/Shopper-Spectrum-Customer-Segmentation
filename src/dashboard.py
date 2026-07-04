import streamlit as st
import pandas as pd
import plotly.express as px

def show_dashboard(df):

    df = df.copy()

    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]

    # ============================
    # DASHBOARD TITLE
    # ============================

    st.markdown("""
    <h1 style='text-align:center;color:#00E5FF'>
    📊 Business Dashboard
    </h1>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # ============================
    # KPI VALUES
    # ============================

    revenue = df["TotalPrice"].sum()
    customers = df["CustomerID"].nunique()
    products = df["Description"].nunique()
    orders = df["InvoiceNo"].nunique()

    # ============================
    # KPI CARDS
    # ============================

    c1, c2, c3, c4 = st.columns(4)

    cards = [

        ("💰 Revenue", f"₹{revenue:,.0f}",
         "linear-gradient(135deg,#667eea,#764ba2)"),

        ("👥 Customers", customers,
         "linear-gradient(135deg,#11998e,#38ef7d)"),

        ("📦 Products", products,
         "linear-gradient(135deg,#f7971e,#ffd200)"),

        ("🛒 Orders", orders,
         "linear-gradient(135deg,#fc466b,#3f5efb)")
    ]

    for col, (title, value, color) in zip([c1, c2, c3, c4], cards):

        with col:

            st.markdown(f"""
            <div style="
            background:{color};
            padding:20px;
            border-radius:18px;
            color:white;
            text-align:center;
            box-shadow:0 6px 15px rgba(0,0,0,.35);">

            <h4>{title}</h4>

            <h2>{value}</h2>

            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ============================
    # ROW 1
    # ============================

    col1, col2 = st.columns(2)

    # ----------------------------
    # Monthly Sales
    # ----------------------------

    with col1:

        monthly = (
            df.groupby(
                df["InvoiceDate"].dt.strftime("%b-%Y")
            )["TotalPrice"]
            .sum()
            .reset_index()
        )

        fig = px.line(

            monthly,

            x="InvoiceDate",

            y="TotalPrice",

            markers=True,

            template="plotly_dark",

            title="📈 Monthly Revenue Trend"

        )

        fig.update_traces(
            line_color="#00E5FF",
            line_width=4
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # ----------------------------
    # Country Sales
    # ----------------------------

    with col2:

        country = (
            df.groupby("Country")["TotalPrice"]
            .sum()
            .sort_values(ascending=False)
            .head(10)
        )

        fig = px.bar(

            x=country.index,

            y=country.values,

            color=country.values,

            color_continuous_scale="Turbo",

            template="plotly_dark",

            title="🌍 Country-wise Sales"

        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # ============================
    # ROW 2
    # ============================

    col3, col4 = st.columns(2)

    # ----------------------------
    # Top Products
    # ----------------------------

    with col3:

        top = (
            df.groupby("Description")["Quantity"]
            .sum()
            .sort_values(ascending=False)
            .head(10)
        )

        fig = px.bar(

            x=top.values,

            y=top.index,

            orientation="h",

            color=top.values,

            color_continuous_scale="Viridis",

            template="plotly_dark",

            title="🏆 Top Selling Products"

        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # ----------------------------
    # Revenue Distribution
    # ----------------------------

    with col4:

        fig = px.histogram(

            df,

            x="TotalPrice",

            nbins=40,

            template="plotly_dark",

            color_discrete_sequence=["#00E5FF"],

            title="💰 Revenue Distribution"

        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # ============================
    # ROW 3
    # ============================

    col5, col6 = st.columns(2)

    # ----------------------------
    # Top Customers
    # ----------------------------

    with col5:

        customer = (
            df.groupby("CustomerID")["TotalPrice"]
            .sum()
            .sort_values(ascending=False)
            .head(10)
        )

        fig = px.bar(

            x=customer.index.astype(str),

            y=customer.values,

            color=customer.values,

            color_continuous_scale="Plasma",

            template="plotly_dark",

            title="👥 Top Customers"

        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    # ----------------------------
    # Customer Segments
    # ----------------------------

    with col6:

        rfm = pd.read_csv("data/rfm_data.csv")

        if "Segment" in rfm.columns:

            seg = rfm["Segment"].value_counts()

            fig = px.pie(

                values=seg.values,

                names=seg.index,

                hole=0.55,

                template="plotly_dark",

                title="🥧 Customer Segments"

            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        else:

            st.warning(
                "Segment column not found in rfm_data.csv"
            )