# Student Dropout Risk Prediction
# Train Machine Learning Model
# "Import Libraries"

import pandas as pd
import joblib

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

# "Load Dataset" 

df = pd.read_csv("data/student_burnout_dropout_dataset_2.csv")

print("First 5 Rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns)

print("\nData Types:")
print(df.dtypes)

print("\nDataset Information:")
df.info()

# "Check Missing Values"

print("\nMissing Values:")
print(df.isnull().sum())

# "Handle Missing Values"

numerical_columns = df.select_dtypes(include=["int64", "float64"]).columns
categorical_columns = df.select_dtypes(include=["object", "string"]).columns

# Fill numerical columns with median
for column in numerical_columns:
    df[column] = df[column].fillna(df[column].median())

# Fill categorical columns with mode
for column in categorical_columns:
    df[column] = df[column].fillna(df[column].mode()[0])

print("\nMissing Values After Cleaning:")
print(df.isnull().sum())

# "Feature Selection"

df = df.drop("Student_ID", axis=1)

# Features and Target
X = df.drop("Dropout_Risk", axis=1)
y = df["Dropout_Risk"]

# "Encode Categorical Variables"

feature_encoders = {}

for column in X.select_dtypes(include=["object", "string"]).columns:
    encoder = LabelEncoder()
    X[column] = encoder.fit_transform(X[column])
    feature_encoders[column] = encoder

target_encoder = LabelEncoder()
y = target_encoder.fit_transform(y)

print("\nFeature Matrix Shape:")
print(X.shape)

print("\nTarget Shape:")
print(y.shape)

print("\nFirst 5 Rows of Features:")
print(X.head())

print("\nFirst 5 Target Values:")
print(y[:5])

# "Train-Test Split"

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining Features Shape:", X_train.shape)
print("Testing Features Shape:", X_test.shape)
print("Training Target Shape:", y_train.shape)
print("Testing Target Shape:", y_test.shape)

# "Train Model"

model = RandomForestClassifier(
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

print("\nModel Trained Successfully!")

# "Predictions"

y_pred = model.predict(X_test)

print("\nFirst 10 Predictions:")
print(y_pred[:10])

print("\nFirst 10 Actual Values:")
print(y_test[:10])

# "Model Evaluation"

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\nModel Evaluation")
print("-" * 30)
print(f"Accuracy : {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall   : {recall:.2f}")
print(f"F1-Score : {f1:.2f}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# "Save Model"

joblib.dump(model, "models/student_dropout_model.pkl")

print("\nModel saved successfully!")