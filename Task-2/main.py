import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Load the datasets
train_df = pd.read_excel('Dataset/train.xlsx')
test_df = pd.read_excel('Dataset/test.xlsx')

# Split the train dataset into features (X) and target (y)
X = train_df.drop('target', axis=1)
y = train_df['target']

# Scale the features using StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train a Random Forest Classifier
rfc = RandomForestClassifier(n_estimators=100, random_state=42)
rfc.fit(X_train, y_train)

# Evaluate the model on the validation set
print("Validation Accuracy:", rfc.score(X_val, y_val))

# Evaluate the model on the training set
print("Train Accuracy:", rfc.score(X_train, y_train))

# Predict the target values for the test dataset
test_X = test_df
test_X_scaled = scaler.transform(test_X)
test_pred = rfc.predict(test_X_scaled)

# Save the predicted target values to a file
pd.DataFrame(test_pred, columns=['target']).to_csv('test_pred.csv', index=False)

print("Predicted target values for test dataset:")
print(test_pred)
