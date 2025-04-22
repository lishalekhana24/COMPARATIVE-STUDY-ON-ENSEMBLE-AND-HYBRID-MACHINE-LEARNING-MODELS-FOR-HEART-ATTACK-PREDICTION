# COMPARATIVE-STUDY-ON-ENSEMBLE-AND-HYBRID-MACHINE-LEARNING-MODELS-FOR-HEART-ATTACK-PREDICTION
Prediction of heart diseases is considered a major tool in the prevention and treatment at an early stage, and it will have a prospect of saving millions of lives each year. The paper draws several kinds of machine learning models with a focus on hybrid models and ensemble techniques for precise prediction. Among the hybrid methods are taken LR-Bagging, OHE2LM, and DT-SVM, where all are ensemble methods. Another method applied here is Stacking, Boosting, Bagging, and Voting.
The results indicate that hybrid models like LR-Bagging and OHE2LM, though they achieve very high testing accuracies at 88.52%, have the ensemble models, particularly the Voting classifier, which can achieve up to 90% accuracy in both training and testing phases. This analysis will depict the great potential of having a variety of algorithms combined to improve the predictability of heart diseases with the help of more efficient hybrid and ensemble models in healthcare.
4.1 DATASET
The dataset used in this paper is taken by GitHub, titled Heart Dataset.csv, is a comprehensive collection of medical and demographic information relevant to predicting heart disease.
The dataset is well-suited for binary classification tasks, where the objective is to predict the presence or absence of heart disease based on the provided features. It serves as an excellent resource for building and evaluating machine learning models aimed at early detection and management of heart disease, contributing to improved patient outcomes and healthcare decisions.
(Table 1) provides all the information of data used for this project it contains features and description where feature represents the attributes of the data and the last attribute which is “target” target variable of the dataset.
Age       Age in years
Sex       Sex (1 = male, 0 = female)
Cp        Chest pain type (0: typical angina,1:atypical angina,2:non-anginal pain,3:asymptomatic)
Trestbps  Resting blood pressure (in mm Hg on admission to the hospital) Serum cholesterol in mg/dl
chol       Serum cholesterol in mg/dl
Fbs        Fasting blood sugar > 120 mg/dl (1 = true; 0 = false)
Restecg    Resting electrocardiographic results (0: normal, 1: having ST-T wave abnormality, 2: showing probable or definite 
               left ventricular hypertrophy)
Exang      Maximum heart rate achieved
Oldpeak    Exercise-induced angina (1 = yes, 0 = no)
Slope      ST depression induced by exercise relative to rest
                  The slope of the peak exercise ST segment (0: upsloping, 1: flat, 2: downsloping)
Thal       ca: Number of major vessels (0-3) colored by fluoroscopy
target     Thalassemia (0 = error, 1 = fixed defect, 2 = normal, 3 = reversible defect)
                 (the label): 0 = no disease, 1 = disease






