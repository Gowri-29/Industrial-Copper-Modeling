# Industrial-Copper-Modeling
## Import necessary libraries
   - import matplotlib.pyplot as plt
  - import warnings
  - warnings.filterwarnings("ignore")
  - import streamlit as st
  - import missingno as msno
  - import dtale
  - from sklearn.model_selection import cross_val_score
  - from sklearn.metrics import mean_squared_error, r2_score
  - from sklearn.datasets import make_regression
  - from sklearn.tree import DecisionTreeRegressor
  - from sklearn.ensemble import ExtraTreesRegressor
  - from sklearn.ensemble import RandomForestRegressor
  - from sklearn.ensemble import AdaBoostRegressor
  - from sklearn.ensemble import GradientBoostingRegressor
  - from xgboost import XGBRegressor
  - from sklearn.linear_model import LinearRegression
  - from sklearn.linear_model import Ridge
  - from sklearn.linear_model import Lasso
  - from sklearn.preprocessing import PolynomialFeatures
  - from sklearn.model_selection import cross_val_predict
  - from sklearn.metrics import confusion_matrix 
  - from sklearn.model_selection import cross_val_score
  - from sklearn.datasets import make_classification
  - from sklearn.tree import DecisionTreeClassifier
  - from sklearn.neighbors import KNeighborsClassifier
  - from sklearn.ensemble import RandomForestClassifier
  - from sklearn.ensemble import GradientBoostingClassifier
  - from sklearn.linear_model import LogisticRegression
  - from sklearn.ensemble import AdaBoostClassifier
  - from sklearn.svm import SVC
  - from sklearn.naive_bayes import GaussianNB

# Data Collection:
This stage involves gathering the necessary data for analysis. It may include obtaining data from various sources such as files.

# Data Analysis:
During this phase, the collected data is explored and analyzed to gain insights and understand patterns, trends, and relationships within the dataset.

# Data Preprocessing:
Data preprocessing involves preparing the data for analysis by cleaning and transforming it. This may include tasks such as removing duplicate entries, addressing skewness in the data distribution, and identifying and handling outliers.

# Feature and Target Selection:
In this step, relevant features (independent variables) and the target variable (dependent variable) are identified from the dataset. This is crucial for building a predictive model that can accurately predict the target variable based on the selected features.

# Model Treatment:
Once the features and target variable are selected, appropriate machine learning models are chosen and trained using the preprocessed data. This involves tuning model hyperparameters, selecting the right algorithm, and optimizing the model's performance.

# Predictive Analysis:
After training the model, it is used to make predictions on new, unseen data. Predictive analysis focuses on forecasting future outcomes or making informed decisions based on the insights gained from the model's predictions.

# Prescriptive Analysis:
Prescriptive analysis goes beyond predictive analysis by recommending actions or strategies to achieve desired outcomes. It involves using insights from predictive models to guide decision-making and optimize processes for better outcomes.
