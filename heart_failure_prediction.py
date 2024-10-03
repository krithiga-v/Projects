import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv("C:\\Users\\KRITHIGA V\\Downloads\\heart\\heart.csv")

# Inspect the first few rows of the dataset
print(df.head())

# Get basic information about the dataset
print(df.info())

# Descriptive statistics of numerical features
print(df.describe())

# Distribution of numerical features
numerical_cols = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
df[numerical_cols].hist(bins=20, figsize=(14, 10), layout=(2, 3))
plt.tight_layout()
plt.show()

# Count plots of categorical features
categorical_cols = ['Sex', 'ChestPainType', 'FastingBS', 'RestingECG', 'ExerciseAngina', 'ST_Slope']
for col in categorical_cols:
    sns.countplot(x=col, data=df)
    plt.title(f'Distribution of {col}')
    plt.show()

# Correlation matrix
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Correlation matrix
corr_matrix = df_encoded.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Check for missing values
print(df.isnull().sum())


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
df = pd.read_csv('C:\\Users\\KRITHIGA V\\Downloads\\heart\\heart.csv')

# List of numerical and categorical columns
numerical_cols = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
categorical_cols = ['Sex', 'ChestPainType', 'FastingBS', 'RestingECG', 'ExerciseAngina', 'ST_Slope']

# Preprocessing pipeline for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(), categorical_cols)
    ])

# Initialize individual models
log_clf = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(random_state=42))
])

rf_clf = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

gb_clf = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', GradientBoostingClassifier(n_estimators=100, random_state=42))
])

# Create the voting classifier
voting_clf = VotingClassifier(
    estimators=[
        ('lr', log_clf),
        ('rf', rf_clf),
        ('gb', gb_clf)
    ],
    voting='soft'  # 'soft' for weighted average probabilities
)

# Split the data into training and testing sets
X = df.drop('HeartDisease', axis=1)
y = df['HeartDisease']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the individual classifiers
log_clf.fit(X_train, y_train)
rf_clf.fit(X_train, y_train)
gb_clf.fit(X_train, y_train)

# Train the voting classifier
voting_clf.fit(X_train, y_train)

# Make predictions
log_pred = log_clf.predict(X_test)
rf_pred = rf_clf.predict(X_test)
gb_pred = gb_clf.predict(X_test)
voting_pred = voting_clf.predict(X_test)

# Make probability predictions
log_proba = log_clf.predict_proba(X_test)
rf_proba = rf_clf.predict_proba(X_test)
gb_proba = gb_clf.predict_proba(X_test)
voting_proba = voting_clf.predict_proba(X_test)

# Evaluate the models
log_accuracy = accuracy_score(y_test, log_pred)
rf_accuracy = accuracy_score(y_test, rf_pred)
gb_accuracy = accuracy_score(y_test, gb_pred)
voting_accuracy = accuracy_score(y_test, voting_pred)

log_report = classification_report(y_test, log_pred)
rf_report = classification_report(y_test, rf_pred)
gb_report = classification_report(y_test, gb_pred)
voting_report = classification_report(y_test, voting_pred)

print(f"Logistic Regression Accuracy: {log_accuracy}")
print("Logistic Regression Classification Report:")
print(log_report)

print(f"Random Forest Accuracy: {rf_accuracy}")
print("Random Forest Classification Report:")
print(rf_report)

print(f"Gradient Boosting Accuracy: {gb_accuracy}")
print("Gradient Boosting Classification Report:")
print(gb_report)

print(f"Voting Classifier Accuracy: {voting_accuracy}")
print("Voting Classifier Classification Report:")
print(voting_report)

# Function to predict heart disease based on user input
def predict_heart_disease():
    print("Enter patient details for prediction:")

    # Get user input
    age = float(input("Age (years): "))
    sex = input("Sex (M/F): ")
    chest_pain_type = input("ChestPainType (TA/ATA/NAP/ASY): ")
    resting_bp = float(input("RestingBP (mm Hg): "))
    cholesterol = float(input("Cholesterol (mm/dl): "))
    fasting_bs = int(input("FastingBS (1 if FastingBS > 120 mg/dl, 0 otherwise): "))
    resting_ecg = input("RestingECG (Normal/ST/LVH): ")
    max_hr = float(input("MaxHR (numeric value between 60 and 202): "))
    exercise_angina = input("ExerciseAngina (Y/N): ")
    oldpeak = float(input("Oldpeak (numeric value measured in depression): "))
    st_slope = input("ST_Slope (Up/Flat/Down): ")

    # Create a DataFrame for the input
    input_data = pd.DataFrame({
        'Age': [age],
        'Sex': [sex],
        'ChestPainType': [chest_pain_type],
        'RestingBP': [resting_bp],
        'Cholesterol': [cholesterol],
        'FastingBS': [fasting_bs],
        'RestingECG': [resting_ecg],
        'MaxHR': [max_hr],
        'ExerciseAngina': [exercise_angina],
        'Oldpeak': [oldpeak],
        'ST_Slope': [st_slope]
    })

    # Make predictions with individual models and ensemble model
    log_proba = log_clf.predict_proba(input_data)
    rf_proba = rf_clf.predict_proba(input_data)
    gb_proba = gb_clf.predict_proba(input_data)
    voting_proba = voting_clf.predict_proba(input_data)

    # Probabilities for each class
    log_heart_disease = log_proba[0][1]
    rf_heart_disease = rf_proba[0][1]
    gb_heart_disease = gb_proba[0][1]
    voting_heart_disease = voting_proba[0][1]

    print(f"Logistic Regression Probability of Heart Disease: {log_heart_disease:.4f}")
    print(f"Random Forest Probability of Heart Disease: {rf_heart_disease:.4f}")
    print(f"Gradient Boosting Probability of Heart Disease: {gb_heart_disease:.4f}")
    print(f"Voting Classifier Probability of Heart Disease: {voting_heart_disease:.4f}")

# Call the prediction function
predict_heart_disease()

