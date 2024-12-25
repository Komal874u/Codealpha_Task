# Import necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Step 1: Load the dataset
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
data = pd.read_csv(url)

# Step 2: Data Preprocessing
# Drop columns that are not useful for prediction
data.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

# Convert 'Sex' column to numeric (0 for male, 1 for female)
data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})

# Convert 'Embarked' to numeric (C = 0, Q = 1, S = 2)
data['Embarked'] = data['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})

# Convert 'Pclass' to numeric (ensure it's an integer)
data['Pclass'] = pd.to_numeric(data['Pclass'], errors='coerce')

# Convert 'Fare' and 'Age' to numeric (handle non-numeric values)
data['Fare'] = pd.to_numeric(data['Fare'], errors='coerce')
data['Age'] = pd.to_numeric(data['Age'], errors='coerce')

# Fill missing values with median or mode
data['Age'].fillna(data['Age'].median(), inplace=True)
data['Fare'].fillna(data['Fare'].median(), inplace=True)
data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)

# Check for missing values
print("Missing values after preprocessing:")
print(data.isnull().sum())  # Should show 0 for all columns now

# Check for data types
print("Data types after preprocessing:")
print(data.dtypes)

# Visualization 1: Pie Chart for Passenger Class Distribution
plt.figure(figsize=(12, 6))  # Set the figure size for both charts

# Create a subplot grid with 1 row and 2 columns
plt.subplot(1, 2, 1)  # (rows, columns, position)
data['Pclass'].value_counts().plot.pie(
    autopct='%1.1f%%', labels=['3rd Class', '1st Class', '2nd Class'], startangle=90
)
plt.title('Passenger Class Distribution')
plt.ylabel('')  # Remove the y-label for a cleaner look

# Visualization 2: Horizontal Bar Chart for Gender Distribution
plt.subplot(1, 2, 2)  # (rows, columns, position)
data['Sex'].value_counts().plot(kind='barh', color=['skyblue', 'salmon'])
plt.title('Gender Distribution')
plt.xlabel('Count')

# Adjust layout to prevent overlap
plt.tight_layout()

# Show the plots
plt.show()

# Step 3: Split the dataset into features (X) and target (y)
X = data.drop('Survived', axis=1)  # Features
y = data['Survived']  # Target variable

# Step 4: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Standardize the features (optional but recommended)
scaler = StandardScaler()

# Scale numeric features only
numeric_features = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
X_train[numeric_features] = scaler.fit_transform(X_train[numeric_features])
X_test[numeric_features] = scaler.transform(X_test[numeric_features])

# Step 6: Train the model (Random Forest Classifier)
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Step 7: Make predictions on the test set
y_pred = model.predict(X_test)

# Step 8: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)