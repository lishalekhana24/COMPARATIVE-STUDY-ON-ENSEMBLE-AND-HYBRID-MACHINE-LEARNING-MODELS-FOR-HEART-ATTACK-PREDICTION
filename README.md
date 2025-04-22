# COMPARATIVE-STUDY-ON-ENSEMBLE-AND-HYBRID-MACHINE-LEARNING-MODELS-FOR-HEART-ATTACK-PREDICTION
Prediction of heart diseases is considered a major tool in the prevention and treatment at an early stage, and it will have a prospect of saving millions of lives each year. The paper draws several kinds of machine learning models with a focus on hybrid models and ensemble techniques for precise prediction. Among the hybrid methods are taken LR-Bagging, OHE2LM, and DT-SVM, where all are ensemble methods. Another method applied here is Stacking, Boosting, Bagging, and Voting.
The results indicate that hybrid models like LR-Bagging and OHE2LM, though they achieve very high testing accuracies at 88.52%, have the ensemble models, particularly the Voting classifier, which can achieve up to 90% accuracy in both training and testing phases. This analysis will depict the great potential of having a variety of algorithms combined to improve the predictability of heart diseases with the help of more efficient hybrid and ensemble models in healthcare.
<code for voting classifier>
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

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

# Define base learners with simpler models and increased regularization
estimators = [
    ('rf', RandomForestClassifier(n_estimators=50, max_depth=10, min_samples_split=10, random_state=42)),  # Simplified RF
    ('gb', GradientBoostingClassifier(n_estimators=50, learning_rate=0.1, max_depth=3, subsample=0.8, random_state=42)),  # Simplified GB
    ('svc', SVC(kernel='linear', C=0.1, probability=True, random_state=42)),  # Increased regularization for SVC
    ('nb', GaussianNB())
]

# Initialize the Voting classifier with soft voting
voting_clf = VotingClassifier(estimators=estimators, voting='soft')

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
