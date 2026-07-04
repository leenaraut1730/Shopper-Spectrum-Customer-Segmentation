import pandas as pd

rfm = pd.read_csv("data/rfm_data.csv")

print(rfm.columns)

print("\nFirst 5 Rows:\n")
print(rfm.head())