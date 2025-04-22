import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import numpy as np

# Load the dataset
df = pd.read_csv('Heart_Dataset.csv')

# Preprocess the data
X = df.drop('target', axis=1)  # Features (assuming 'target' is the label)
y = df['target']  # Labels

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize base models with regularization
logistic = LogisticRegression(random_state=42)
svc = SVC(kernel='linear', probability=True, random_state=42)
rf = RandomForestClassifier(random_state=42)
gb = GradientBoostingClassifier(random_state=42)
xgb = XGBClassifier(random_state=42)

# Define parameter grids
param_grid_logistic = {
    'C': np.logspace(-3, 3, 7)
}
param_grid_svc = {
    'C': np.logspace(-3, 3, 7),
    'gamma': ['scale', 'auto']
}
param_grid_rf = {
    'n_estimators': [50, 100],
    'max_depth': [10, 20],
    'min_samples_split': [10, 20],
    'min_samples_leaf': [5, 10]
}
param_grid_gb = {
    'n_estimators': [50, 100],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 4, 5]
}
param_grid_xgb = {
    'n_estimators': [50, 100],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 4, 5],
    'subsample': [0.8, 0.9],
    'colsample_bytree': [0.8, 0.9]
}

# Perform GridSearchCV for each model
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid_search_logistic = GridSearchCV(logistic, param_grid_logistic, cv=cv, n_jobs=-1)
grid_search_svc = GridSearchCV(svc, param_grid_svc, cv=cv, n_jobs=-1)
grid_search_rf = GridSearchCV(rf, param_grid_rf, cv=cv, n_jobs=-1)
grid_search_gb = GridSearchCV(gb, param_grid_gb, cv=cv, n_jobs=-1)
grid_search_xgb = GridSearchCV(xgb, param_grid_xgb, cv=cv, n_jobs=-1)

# Fit the models
grid_search_logistic.fit(X_train, y_train)
grid_search_svc.fit(X_train, y_train)
grid_search_rf.fit(X_train, y_train)
grid_search_gb.fit(X_train, y_train)
grid_search_xgb.fit(X_train, y_train)

# Get the best models
best_logistic = grid_search_logistic.best_estimator_
best_svc = grid_search_svc.best_estimator_
best_rf = grid_search_rf.best_estimator_
best_gb = grid_search_gb.best_estimator_
best_xgb = grid_search_xgb.best_estimator_

# Define the hybrid model using VotingClassifier with the best models
voting_clf = VotingClassifier(estimators=[
    ('logistic', best_logistic),
    ('svc', best_svc),
    ('rf', best_rf),
    ('gb', best_gb),
    ('xgb', best_xgb)
], voting='soft')

# Train the voting ensemble
voting_clf.fit(X_train, y_train)

# Cross-validation
cross_val_scores = cross_val_score(voting_clf, X_train, y_train, cv=5)
mean_cv_score = cross_val_scores.mean()

# Predict and evaluate on the training set
y_train_pred = voting_clf.predict(X_train)
train_accuracy = accuracy_score(y_train, y_train_pred)

# Predict and evaluate on the test set
y_pred = voting_clf.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# Print performance metrics
print(f'Cross-validation scores: {cross_val_scores}')
print(f'Mean cross-validation score: {mean_cv_score:.2f}')
print(f'Train Accuracy: {train_accuracy:.2f}')
print(f'Test Accuracy: {test_accuracy:.2f}')
print('Confusion Matrix:')
print(conf_matrix)
print('Classification Report:')
print(class_report)

# Function to predict user input
def predict_heart_attack(input_data):
    input_data = pd.DataFrame([input_data], columns=X.columns)
    input_data = scaler.transform(input_data)
    prediction = voting_clf.predict(input_data)
    return 'Heart Attack' if prediction[0] == 1 else 'No Heart Attack'
# Interactive user input
def get_user_input():
    input_data = {}
    input_data['age'] = int(input("Enter age: "))
    input_data['sex'] = int(input("Enter sex (1 = male, 0 = female): "))
    input_data['cp'] = int(input("Enter chest pain type (0-3): "))
    input_data['trestbps'] = int(input("Enter resting blood pressure: "))
    input_data['chol'] = int(input("Enter serum cholesterol: "))
    input_data['fbs'] = int(input("Enter fasting blood sugar (1 if > 120 mg/dl, 0 otherwise): "))
    input_data['restecg'] = int(input("Enter resting ECG results (0-2): "))
    input_data['thalach'] = int(input("Enter maximum heart rate achieved: "))
    input_data['exang'] = int(input("Enter exercise induced angina (1 = yes, 0 = no): "))
    input_data['oldpeak'] = float(input("Enter ST depression induced by exercise: "))
    input_data['slope'] = int(input("Enter the slope of the peak exercise ST segment (0-2): "))
    input_data['ca'] = int(input("Enter number of major vessels (0-3): "))
    input_data['thal'] = int(input("Enter thal (1 = normal; 2 = fixed defect; 3 = reversible defect): "))
    return input_data

# Example usage
if __name__ == "__main__":
    user_input = get_user_input()
    prediction = predict_heart_attack(user_input)
    print(f'User Input Prediction: {prediction}')
