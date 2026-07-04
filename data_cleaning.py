import pandas as pd

# Load Dataset
df = pd.read_csv("online_retail.csv", encoding='ISO-8859-1')

# Display Dataset Information
print("Original Dataset Shape:", df.shape)
print(df.info())

# -----------------------------
# Missing Values
# -----------------------------
print("\nMissing Values:")
print(df.isnull().sum())

# Remove Missing CustomerID
df = df.dropna(subset=['CustomerID'])

# Remove Missing Description
df = df.dropna(subset=['Description'])

# -----------------------------
# Remove Duplicate Rows
# -----------------------------
print("\nDuplicate Rows:", df.duplicated().sum())

df = df.drop_duplicates()

# -----------------------------
# Remove Cancelled Invoices
# -----------------------------
df = df[~df['InvoiceNo'].astype(str).str.startswith('C')]

# -----------------------------
# Remove Negative Quantity
# -----------------------------
df = df[df['Quantity'] > 0]

# -----------------------------
# Remove Zero or Negative Price
# -----------------------------
df = df[df['UnitPrice'] > 0]

# -----------------------------
# Convert InvoiceDate
# -----------------------------
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

# -----------------------------
# Create TotalAmount Column
# -----------------------------
df['TotalAmount'] = df['Quantity'] * df['UnitPrice']

# -----------------------------
# Final Dataset
# -----------------------------
print("\nCleaned Dataset Shape:", df.shape)

print(df.head())

# Save Clean Dataset
df.to_csv("cleaned_online_retail.csv", index=False)

print("\nDataset Cleaned Successfully!")