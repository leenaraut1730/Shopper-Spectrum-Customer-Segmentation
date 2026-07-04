import pandas as pd

# Load dataset
df = pd.read_csv("data/online_retail.csv")

# Remove missing CustomerID
df = df.dropna(subset=["CustomerID"])

# Create TotalPrice
df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]

# Latest date
latest_date = df["InvoiceDate"].max()

# RFM table
rfm = df.groupby("CustomerID").agg({
    "InvoiceDate": lambda x: (pd.to_datetime(latest_date) - pd.to_datetime(x).max()).days,
    "InvoiceNo": "count",
    "TotalPrice": "sum"
})

rfm.columns = ["Recency", "Frequency", "Monetary"]

# Save file
rfm.to_csv("data/rfm_data.csv")

print("RFM file created successfully!")