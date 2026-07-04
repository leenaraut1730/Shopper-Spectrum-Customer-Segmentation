import os
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# =====================================
# Create folders
# =====================================
os.makedirs("images", exist_ok=True)
os.makedirs("models", exist_ok=True)

# =====================================
# Load RFM Data
# =====================================
print("Loading RFM data...")

rfm = pd.read_csv("data/customer_segments.csv")

print("Dataset Shape:", rfm.shape)

# =====================================
# Features
# =====================================
X = rfm[["Recency", "Frequency", "Monetary"]]

# =====================================
# Scaling
# =====================================
print("Scaling data...")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# =====================================
# Elbow Method
# =====================================
print("Running Elbow Method...")

inertia = []

for i in range(2, 6):      # 2 to 5 (fast)
    print(f"Training KMeans with {i} clusters...")
    km = KMeans(
        n_clusters=i,
        random_state=42,
        n_init=10
    )

    km.fit(X_scaled)
    inertia.append(km.inertia_)

print("Elbow Method Completed")

plt.figure(figsize=(8,5))
plt.plot(range(2,6), inertia, marker="o")
plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")
plt.grid(True)

plt.savefig("images/elbow_curve.png")
plt.close()

# =====================================
# Final Model
# =====================================
print("Training Final KMeans Model...")

kmeans = KMeans(
    n_clusters=4,
    random_state=42,
    n_init=10
)

clusters = kmeans.fit_predict(X_scaled)

rfm["Cluster"] = clusters

# =====================================
# Silhouette Score
# =====================================
score = silhouette_score(X_scaled, clusters)

print("Silhouette Score:", score)

# =====================================
# Segment Names
# =====================================
cluster_names = {
    0: "High Value",
    1: "Regular",
    2: "Occasional",
    3: "At Risk"
}

rfm["Segment"] = rfm["Cluster"].map(cluster_names)

# =====================================
# Save Customer Segments
# =====================================
rfm.to_csv(
    "data/rfm_data.csv",
    index=False
)

print("✅ rfm_data.csv updated successfully")

print("customer_segments.csv saved")

# =====================================
# Scatter Plot
# =====================================
plt.figure(figsize=(8,6))

plt.scatter(
    rfm["Frequency"],
    rfm["Monetary"],
    c=rfm["Cluster"]
)

plt.xlabel("Frequency")
plt.ylabel("Monetary")
plt.title("Customer Segments")

plt.savefig("images/customer_clusters.png")
plt.close()

print("Scatter plot saved")

# =====================================
# Save Models
# =====================================
print("Saving models...")

joblib.dump(
    kmeans,
    "models/kmeans_model.pkl"
)

joblib.dump(
    scaler,
    "models/scaler.pkl"
)

print("Models Saved Successfully!")

print("\nCustomer Segments:")
print(rfm["Segment"].value_counts())