import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load the data
train_data = pd.read_excel('Dataset/train.xlsx')
test_data = pd.read_excel('Dataset/test.xlsx')

# Extract features from the training data
X_train = train_data.drop(columns=['target'])

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Extract features from the test data and standardize
X_test = test_data
X_test_scaled = scaler.transform(X_test)

# Determine the optimal number of clusters using the Elbow Method
inertia = []
K = range(1, 11)

for k in K:
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(X_train_scaled)
    inertia.append(kmeans.inertia_)

# Plot the elbow curve
plt.figure(figsize=(10, 6))
plt.plot(K, inertia, 'bx-')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal k')
plt.show()

# Fit the K-Means model with the optimal number of clusters (manually set optimal_k based on elbow curve)
optimal_k = 3
kmeans = KMeans(n_clusters=optimal_k, random_state=0)
kmeans.fit(X_train_scaled)

# Predict clusters for the training data
train_clusters = kmeans.predict(X_train_scaled)
train_data['Cluster'] = train_clusters

# Predict clusters for the test data
test_clusters = kmeans.predict(X_test_scaled)
test_data['Cluster'] = test_clusters

# Get the cluster centroids
centroids = kmeans.cluster_centers_

# Function to predict cluster for new data point
def predict_cluster(new_data_point):
    new_data_point_df = pd.DataFrame([new_data_point], columns=X_train.columns)
    new_data_point_scaled = scaler.transform(new_data_point_df)
    cluster_label = kmeans.predict(new_data_point_scaled)[0]
    centroid = centroids[cluster_label]

    # Find the most significant feature contributing to the cluster
    differences = new_data_point - centroid
    most_significant_feature = X_train.columns[differences.argmax()]

    print(f"Cluster Label: {cluster_label}")
    print(f"Most significant feature: {most_significant_feature}, value: {new_data_point[differences.argmax()]:.2f}, centroid value: {centroid[differences.argmax()]:.2f}")
    input("")

# Allow user to input new data points
new_data_point = list(map(float, input("Enter new data point: ").split(' ')))
predict_cluster(new_data_point)
