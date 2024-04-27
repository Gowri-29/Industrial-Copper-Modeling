#import necessary libraries
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import streamlit as st
import missingno as msno
import dtale
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import make_regression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix 
from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB



df = pd.read_csv("Copper_Set.xlsx - Result 1.csv")
# dealing with data in wrong format,for categorical variables, this step is ignored

df['item_date'] = pd.to_datetime(df['item_date'], format='%Y%m%d', errors='coerce').dt.date
df['quantity tons'] = pd.to_numeric(df['quantity tons'], errors='coerce')
df['customer'] = pd.to_numeric(df['customer'], errors='coerce')
df['country'] = pd.to_numeric(df['country'], errors='coerce')
df['application'] = pd.to_numeric(df['application'], errors='coerce')
df['thickness'] = pd.to_numeric(df['thickness'], errors='coerce')
df['width'] = pd.to_numeric(df['width'], errors='coerce')
df['product_ref'] = pd.to_numeric(df['product_ref'], errors='coerce')
df['delivery date'] = pd.to_datetime(df['delivery date'], format='%Y%m%d', errors='coerce').dt.date
df['selling_price'] = pd.to_numeric(df['selling_price'], errors='coerce')
df.drop(columns=['id','material_ref'], inplace=True)
df['item_date'] = pd.to_datetime(df['item_date'])
df['delivery date'] = pd.to_datetime(df['delivery date'])

view = df.head(5)

# Data preprocessing:Visualize missing values
def missing_values(df):
    msno.matrix(df)
    st.pyplot()
    sns.heatmap(df_num.corr(), annot=True)
    st.pyplot()

def EDA(df):
    # Launch d-tale for the DataFrame
    d = dtale.show(df)

    # To view the d-tale instance in a Jupyter notebook, you can use:
    d.open_browser()
# removing negative values:
a = df['selling_price'] <= 0
print(a.sum())
df.loc[a, 'selling_price'] = np.nan

a = df['quantity tons'] <= 0
print(a.sum())
df.loc[a, 'quantity tons'] = np.nan

a = df['thickness'] <= 0
print(a.sum())
df.loc[a, 'thickness'] = np.nan
# STATUS variable with WON being considered as Success and LOST being considered as Failure and remove data points other than WON, LOST STATUS values.
selected_statuses = ["Won", "Lost"]
df = df[df["status"].isin(selected_statuses)]
# before removing sknew:
df_num = df.select_dtypes(["float64","int64"])
def plot(df, column):
    # Create subplots
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))

    # Box plot
    sns.boxplot(data=df, x=column, ax=axes[0])
    axes[0].set_title(f'Box Plot for {column}')

    # Distribution plot
    sns.histplot(data=df, x=column, kde=True, bins=50, ax=axes[1])
    axes[1].set_title(f'Distribution Plot for {column}')

    # Violin plot
    sns.violinplot(data=df, x=column, ax=axes[2])
    axes[2].set_title(f'Violin Plot for {column}')

    # Show plots
    st.pyplot(fig)

#Skew removing:
df_skewed = df.copy()
df_skewed['quantity tons'] = np.log(df_skewed['quantity tons'])
df_skewed['thickness'] = np.log(df_skewed['thickness'])
df_skewed = df_skewed.dropna()

# Outlier:
df_outiler = df_skewed.copy()
def outlier(df, column):
    iqr = df[column].quantile(0.75) - df[column].quantile(0.25)
    upper_threshold = df[column].quantile(0.75) + (1.5*iqr)
    lower_threshold = df[column].quantile(0.25) - (1.5*iqr)
    df[column] = df[column].clip(lower_threshold, upper_threshold)
outlier(df_outiler, 'quantity tons')
outlier(df_outiler, 'thickness')
outlier(df_outiler, 'width')        

# encoded data:
df_encode = df_outiler.copy()
from sklearn.preprocessing import LabelEncoder

# Instantiate the LabelEncoder object
le = LabelEncoder()

# Fit and transform the "status" column
encoded_status = le.fit_transform(df_encode["status"])
df_encode["status"] = encoded_status  # Add a new column for the encoded status

# Fit and transform the "item type" column
encoded_item_type = le.fit_transform(df_encode["item type"])
df_encode["item type"] = encoded_item_type  # Add a new column for the encoded item type

# Prediction 1:
def selling_price_prediction():
    Features = df_encode[["item type","quantity tons","status","customer","country","application","thickness","width","product_ref"]]
    Target = df_encode["selling_price"]
    X = Features
    y = Target
    from sklearn.preprocessing import StandardScaler
    SS = StandardScaler()
    SS.fit_transform(X)
    return X,y

# Define the function
def machine_learning_selling_price(X, y, algorithm, cv=5):
    # Generate some synthetic data for demonstration
    X, y = make_regression(n_samples=1000, n_features=10, noise=0.1, random_state=42)
    scores = cross_val_score(algorithm(), X, y, cv=cv, scoring='neg_mean_squared_error')
    rmse_scores = np.sqrt(-scores)  # Calculate root mean squared error from negative MSE scores
    r2_scores = cross_val_score(algorithm(), X, y, cv=cv, scoring='r2')  # Calculate R-squared scores

    metrics = {
        'Algorithm': str(algorithm).split("'")[1].split(".")[-1],
        'Cross-Validation RMSE Mean': rmse_scores.mean(),
        'Cross-Validation RMSE Std': rmse_scores.std(),
        'Cross-Validation R2 Mean': r2_scores.mean(),
        'Cross-Validation R2 Std': r2_scores.std()
    }

    return metrics


def prediction_model1(X, y, algorithm, cv=5):
    # Perform cross-validated prediction
    X, y = make_regression(n_samples=1000, n_features=10, noise=0.1, random_state=42)
    from sklearn.model_selection import cross_val_predict

    # Define RandomForestRegressor
    algorithm = algorithm()

    # Perform cross-validation to obtain predicted values
    y_pred_rf_cv = cross_val_predict(algorithm, X, y, cv=5)
    
    # Plot true vs predicted values
    plt.figure(figsize=(8, 6))
    plt.scatter(y, y_pred_rf_cv, color='blue', alpha=0.5)
    plt.plot([min(y), max(y)], [min(y), max(y)], color='red', linestyle='--')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title('True vs Predicted Values')
    
    # Get the current figure handle and display it using st.pyplot()
    fig = plt.gcf()
    st.pyplot(fig)


df_class = df_outiler.copy()

# Prediction 2:
def Status_prediction():
    Features = df_encode[["item type","quantity tons","status","customer","country","application","thickness","width","product_ref"]]
    Target = df_class["status"]
    X_classification = Features
    y_classification = Target
    return(X_classification,y_classification)
    

# Define the function 2
def machine_learning_Status(X, y, algorithm, cv=5):
    # Generate some synthetic data for demonstration
    X, y = make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=42)

    if algorithm.__name__ == "LogisticRegression":
        scoring = 'accuracy'
    else:
        scoring = 'roc_auc'

    scores = cross_val_score(algorithm(), X, y, cv=cv, scoring=scoring)
    mean_score = np.mean(scores)
    std_score = np.std(scores)

    metrics = {
        'Algorithm': algorithm.__name__,
        'Cross-Validation Mean Score': mean_score,
        'Cross-Validation Std Score': std_score
    }

    return metrics

import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression

import streamlit as st
import pandas as pd

def prediction_model2(X, y, algorithm, cv=5):
    from sklearn.model_selection import cross_val_predict
    from sklearn.metrics import confusion_matrix

    # Example usage with Logistic Regression
    model = algorithm()

    # Obtain predicted labels using cross-validation
    y_pred = cross_val_predict(model, X, y, cv=5)

    # Plot true vs predicted values
    plt.figure(figsize=(8, 6))
    plt.scatter(y, y_pred, color='blue', alpha=0.5)
    plt.plot([min(y), max(y)], [min(y), max(y)], color='red', linestyle='--')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title('True vs Predicted Values')
    
    # Get the current figure handle and display it using st.pyplot()
    fig = plt.gcf()
    st.pyplot(fig)


# Example usage
# Replace X, y, and algorithm with your actual data and algorithm
# prediction_model2(X, y, algorithm)

# streamlit:
st.set_option('deprecation.showPyplotGlobalUse', False)
st.header(':blue[_Industrial Copper Modeling Application_]')
tab1, tab2, tab3, tab4 = st.tabs([":briefcase: Data collection", ":clipboard: Data Modelling_Reg", ":clipboard: Data Modelling_Reg",":chart_with_upwards_trend: Data Prediction"])

with tab1:
    if st.button("Data View"):
        st.table(view)
    if st.button("Preprocessed Data"):
        missing_values(df)
    if st.button("EDA"):
        EDA(df)
    if st.button("Untreated_Skew_Data"):
        for i in df_num:
            plot(df_num, i)
    if st.button("Treated_Skew_Data"):
        for i in ['quantity tons', 'thickness', 'width']:
            plot(df_skewed, i)
    if st.button("Outlier_Treatment"):
        for i in ['quantity tons', 'thickness', 'width']:
            plot(df_outiler, i)
        view_data =  df_outiler.head(5)
        st.table(view_data)   
Data = []

with tab2:
   st.header("Data Modelling")

   Regression_algorithms = {
        "Extra Trees Regressor": ExtraTreesRegressor,
        "Random Forest Regressor": RandomForestRegressor,
        "AdaBoost Regressor": AdaBoostRegressor,
        "Gradient Boosting Regressor": GradientBoostingRegressor,
        "XGBoost Regressor": XGBRegressor,
        "Decision Tree Regressor": DecisionTreeRegressor,
        "Linear Regression": LinearRegression,
        "Ridge Regression": Ridge,
        "Lasso Regression": Lasso,
    }
   Regression_Selection = st.selectbox("Algorithm selection", list(Regression_algorithms.keys()))
   
   if st.button("Run Model_Reg"):
        X, y = selling_price_prediction()
        Metrics = machine_learning_selling_price(X, y, Regression_algorithms[Regression_Selection])
        st.write("Results:", Metrics)
        Data.append(Metrics)
        
   if Data:
        st.header("Evaluation Metrics_Reg")
        st.table(Data)
   
# confusion matrix

  # Dropdown to select the regression algorithm for confusion matrix
   Regression_Selection_Confusion_Matrix = st.selectbox("Algorithm selection for Confusion Matrix", list(Regression_algorithms.keys()))

   if st.button("Confusion Matrix_Reg"):
        X, y = selling_price_prediction()
        prediction_model1(X, y, Regression_algorithms[Regression_Selection_Confusion_Matrix])
   
       
with tab3:
   Classification_algorithms = {
        "DecisionTreeClassifier": DecisionTreeClassifier,
        "KNeighborsClassifier": KNeighborsClassifier,
        "RandomForestClassifier": RandomForestClassifier,
        "GradientBoostingClassifier": GradientBoostingClassifier,
        "LogisticRegression": LogisticRegression,
        "AdaBoostClassifier": AdaBoostClassifier,
        "SVC": SVC,
        "Ridge GaussianNB": GaussianNB
    }
   Classification_Selection = st.selectbox("Algorithm selection", list(Classification_algorithms.keys()))
   
   if st.button("Run Model_Class"):
        X_class, y_class = Status_prediction()
        Metrics = machine_learning_Status(X_class, y_class, Classification_algorithms[Classification_Selection])
        st.write("Results:", Metrics)
        Data.append(Metrics)
        
   if Data:
        st.header("Evaluation Metrics_Class")
        st.table(Data)
   
# confusion matrix

  # Dropdown to select the regression algorithm for confusion matrix
   Classification_Selection_Confusion_Matrix = st.selectbox("Algorithm selection for Confusion Matrix", list(Classification_algorithms.keys()))

   if st.button("Confusion Matrix_Class"):
        X_class, y_class = Status_prediction()
        prediction_model2(X_class, y_class, Classification_algorithms[Classification_Selection])
        
with tab4:                
    import streamlit as st
    from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
    from xgboost import XGBRegressor
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.linear_model import LinearRegression, Ridge, Lasso

    # Define Regression algorithms
    Regression_algorithms = {
        "Extra Trees Regressor": ExtraTreesRegressor,
        "Random Forest Regressor": RandomForestRegressor,
        "AdaBoost Regressor": AdaBoostRegressor,
        "Gradient Boosting Regressor": GradientBoostingRegressor,
        "XGBoost Regressor": XGBRegressor,
        "Decision Tree Regressor": DecisionTreeRegressor,
        "Linear Regression": LinearRegression,
        "Ridge Regression": Ridge,
        "Lasso Regression": Lasso,
    }

    Regression_Selec = st.selectbox("Algorithm_selection", list(Regression_algorithms.keys()))

    # Function to predict probability
    def predict_probability(input_features, model):
        # Make prediction
        predicted_value = model.predict([input_features])
        return predicted_value[0]

    # Example usage:
    text_input = st.text_input(
    "Enter valid Data - item type (int), quantity tons (float), status (int), customer (float), country (float), application (float), thickness (float), width, product_ref (int)) 👇")

    if text_input:
        st.write("You entered: ", text_input)
        # Split the input string into individual values
        input_values = text_input.split(',')
        # Convert each value to the appropriate data type
        try:
            input_features = [int(input_values[0]), float(input_values[1]), int(input_values[2]), float(input_values[3]), float(input_values[4]), float(input_values[5]), float(input_values[6]), float(input_values[7]), int(input_values[8])]
            st.write("Input features:", input_features)
        except ValueError:
            st.error("Please enter valid numeric values for all features.")


        # Select the model
        selected_model = Regression_algorithms[Regression_Selec]()
        Features = df_encode[["item type","quantity tons","status","customer","country","application","thickness","width","product_ref"]]
        Target = df_encode["selling_price"]
        X_classification = Features
        y_classification = Target
        selected_model.fit(X_classification, y_classification)

        # Predict
        predicted_probability = predict_probability(input_features, selected_model)
        st.write("Predicted value:", predicted_probability)
