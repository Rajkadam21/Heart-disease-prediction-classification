# Heart-disease-prediction-classification
Overview
This project aims to predict the likelihood of heart disease in individuals based on various health parameters. It utilizes machine learning classification techniques to analyze a dataset containing a combination of demographic information, lifestyle factors, and medical test results. The predictive model helps identify individuals at risk of heart disease, enabling early intervention and personalized healthcare.

Key Features
Data Exploration: Comprehensive exploration of the dataset to understand patterns, correlations, and potential factors influencing heart disease.

Data Preprocessing: Cleaning, handling missing values, and transforming raw data into a format suitable for machine learning algorithms.

Feature Selection: Identification and selection of relevant features that contribute significantly to the predictive model.

Model Selection: Implementation of multiple classification algorithms such as Logistic Regression, Random Forest, and Support Vector Machines. Comparison of their performances to choose the most suitable model.

Model Evaluation: Rigorous evaluation of the chosen model using metrics such as accuracy, precision, recall, and F1-score. Utilization of cross-validation to ensure robustness.

Hyperparameter Tuning: Fine-tuning of model parameters to optimize performance and generalization.

Visualization: Clear and informative visualizations of key insights, model performance, and feature importance.

Deployment: Optionally, a simple deployment script or API for making predictions based on the trained model.

Technologies Used
Python
Jupyter Notebooks
scikit-learn
Pandas
Matplotlib
Seaborn
Machine Learning Algorithms

Dataset
The dataset used in this project is sourced from [Heart.csv]. It includes 
[Data contains;

Age - age in years

Sex - (1 = male; 0 = female)

ChestPain - chest pain type

RestBP - resting blood pressure (in mm Hg on admission to the hospital)

Chol - serum cholestoral in mg/dl

Fbs - (fasting blood sugar > 120 mg/dl) (1 = true; 0 = false)

RestECG - resting electrocardiographic results

MaxHR - maximum heart rate achieved

ExAng - exercise induced angina (1 = yes; 0 = no)

Oldpeak - ST depression induced by exercise relative to rest

Slope - the slope of the peak exercise ST segment

Ca - number of major vessels (0-3) colored by flourosopy

Thal - 3 = normal; 6 = fixed defect; 7 = reversable defect

AHD - have disease or not (1=yes, 0=no)
].



