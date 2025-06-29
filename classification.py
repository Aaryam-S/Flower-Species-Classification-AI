from sklearn.datasets import load_iris
import pandas as pd

iris = load_iris()
# Optional: Convert to DataFrame
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target
df['species'] = iris.target_names[iris.target] # Add species names for clarity
import matplotlib.pyplot as plt
import seaborn as sns

# If using DataFrame:
print(df.head())
print(df.info())
print(df.describe())
print(df['species'].value_counts()) # Check class distribution

# Visualization example: Pair plot
sns.pairplot(df, hue='species', markers=["o", "s", "D"])
plt.show()
from sklearn.model_selection import train_test_split

X = iris.data # Features
y = iris.target # Target labels

# Split data: 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training set shape:", X_train.shape)
print("Testing set shape:", X_test.shape)
# Optional Scaling
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train) # Fit on training, transform training
X_test_scaled = scaler.transform(X_test)       # ONLY transform testing
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Use scaled data if you applied scaling, otherwise use original X_test
X_test_used = X_test_scaled if 'X_test_scaled' in locals() else X_test

y_pred = model.predict(X_test_used)

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Example: Predict a new flower with specific measurements
# Suppose a new flower has measurements: sepal_length=5.1, sepal_width=3.5, petal_length=1.4, petal_width=0.2
new_flower_measurements = [[5.1, 3.5, 1.4, 0.2]] # Must be a 2D array/list of lists

# Apply the same scaling if you used it for training
if 'scaler' in locals():
    new_flower_scaled = scaler.transform(new_flower_measurements)
    prediction = model.predict(new_flower_scaled)
else:
    prediction = model.predict(new_flower_measurements)


predicted_species_index = prediction[0] # Get the single prediction value
predicted_species_name = iris.target_names[predicted_species_index]

print(f"\nMeasurements: {new_flower_measurements[0]}")
print(f"Predicted species index: {predicted_species_index}")
print(f"Predicted species name: {predicted_species_name}")

from sklearn.linear_model import LogisticRegression

# Create and train the Logistic Regression model
model = LogisticRegression(max_iter=200)
model.fit(X_train_scaled, y_train)