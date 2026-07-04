import os
import pandas as pd
import matplotlib.pyplot as plt

# Create folders (IMPORTANT)
os.makedirs("images", exist_ok=True)

# Load dataset
df = pd.read_csv("data/cleaned_online_retail.csv")

# Convert date
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

# ----------------------------
# Create TotalPrice (IMPORTANT FIX)
# ----------------------------
df["TotalPrice"] = df["Quantity"] * df["UnitPrice"]

# ====================================
# Total Revenue
# ====================================
revenue = df["TotalPrice"].sum()
print("Total Revenue :", revenue)

# ====================================
# Top Selling Products
# ====================================
top_products = (
    df.groupby("Description")["Quantity"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

plt.figure(figsize=(10,5))
top_products.plot(kind="bar")
plt.title("Top Selling Products")
plt.ylabel("Quantity Sold")
plt.tight_layout()
plt.savefig("images/top_products.png")
plt.show()

# ====================================
# Country Wise Sales
# ====================================
country_sales = (
    df.groupby("Country")["TotalPrice"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

plt.figure(figsize=(10,5))
country_sales.plot(kind="bar")
plt.title("Top Countries by Sales")
plt.ylabel("Revenue")
plt.tight_layout()
plt.savefig("images/country_sales.png")
plt.show()

# ====================================
# Monthly Sales
# ====================================
df["Month"] = df["InvoiceDate"].dt.to_period("M").astype(str)

monthly_sales = df.groupby("Month")["TotalPrice"].sum()

plt.figure(figsize=(10,5))
monthly_sales.plot(marker="o")
plt.title("Monthly Sales Trend")
plt.ylabel("Revenue")
plt.tight_layout()
plt.savefig("images/monthly_sales.png")
plt.show()

# ====================================
# Top Customers
# ====================================
top_customers = (
    df.groupby("CustomerID")["TotalPrice"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

plt.figure(figsize=(10,5))
top_customers.plot(kind="bar")
plt.title("Top Customers")
plt.ylabel("Total Spend")
plt.tight_layout()
plt.savefig("images/top_customers.png")
plt.show()

# ====================================
# Revenue Distribution
# ====================================
plt.figure(figsize=(8,5))
plt.hist(df["TotalPrice"], bins=50)
plt.title("Revenue Distribution")
plt.xlabel("Transaction Value")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig("images/revenue_distribution.png")
plt.show()

print("\nEDA Completed Successfully 🚀")