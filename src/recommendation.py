import os
import pandas as pd
import joblib
from sklearn.metrics.pairwise import cosine_similarity

# Create folders
os.makedirs("models", exist_ok=True)

# ==========================
# Load Dataset
# ==========================
df = pd.read_csv("data/cleaned_online_retail.csv")

# Remove missing products
df = df.dropna(subset=["Description"])

# ==========================
# Customer-Product Matrix
# ==========================
customer_product_matrix = df.pivot_table(
    index="CustomerID",
    columns="Description",
    values="Quantity",
    aggfunc="sum",
    fill_value=0
)

print("Matrix Shape:", customer_product_matrix.shape)

# ==========================
# Similarity Matrix
# ==========================
similarity = cosine_similarity(customer_product_matrix.T)

similarity_df = pd.DataFrame(
    similarity,
    index=customer_product_matrix.columns,
    columns=customer_product_matrix.columns
)

print("Similarity matrix created!")

# ==========================
# Save Models
# ==========================
joblib.dump(similarity_df, "models/similarity_matrix.pkl")
joblib.dump(list(customer_product_matrix.columns), "models/product_list.pkl")

print("Model saved successfully!")

# ==========================
# Recommendation Function
# ==========================
def recommend(product_name):
    if product_name not in similarity_df.index:
        print("❌ Product Not Found")
        return

    recommendations = similarity_df[product_name] \
        .sort_values(ascending=False) \
        .iloc[1:6]

    print("\nTop 5 Recommended Products:")
    for product in recommendations.index:
        print("-", product)

# ==========================
# TEST (use real product name)
# ==========================
sample_product = customer_product_matrix.columns[0]
print("\nTesting with:", sample_product)

recommend(sample_product)