import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.linear_model import Lasso

def load_data(filepath):
    data = pd.read_csv(filepath, encoding = "latin1", low_memory = False)
    print(data.head())
    print(data.info())
    return data

def assigned_meal_category(data):
    data.loc[0:17, 'Type'] = 'McValue'
    data.loc[17:46, 'Type'] = 'Breakfast'
    data.loc[46:55, 'Type'] = 'Burgers'
    data.loc[55:61, 'Type'] = 'Chicken & Fish Sandwiches'
    data.loc[61:68, 'Type'] = 'McNuggets & McCrispy Strips'
    data.loc[68:83, 'Type'] = 'Fries & Sides'
    data.loc[83:86, 'Type'] = 'Happy Meals'
    data.loc[86:106, 'Type'] = 'Sweets & Treats'
    data.loc[106:166, 'Type'] = 'McCafe Coffees'
    data.loc[166:224, 'Type'] = 'Beverages'
    return data

def check_duplicates(data):
    print(data.duplicated().sum())

def check_nulls(data):
    print(data.isnull().sum())

def replace_values(data):
    data['Calories'] = data['Calories'].replace('<5', 4.9)
    data['Cholesterol'] = data['Cholesterol'].replace('<5', 4.9)
    data['Sodium'] = data['Sodium'].replace('<5', 4.9)
    return data

def convert_data_types(data):
    data['Calories'] = pd.to_numeric(data['Calories'].astype(str).str.replace(',', '', regex = True), errors = 'coerce')
    data['Cholesterol'] = pd.to_numeric(data['Cholesterol'].astype(str).str.replace(',', '', regex = True), errors = 'coerce')
    data['Sodium'] = pd.to_numeric(data['Sodium'].astype(str).str.replace(',', '', regex = True), errors = 'coerce')
    return data

def numerical_distribution(feature):
    plt.figure(figsize = (12, 5))

    plt.subplot(1, 2, 1)
    sns.histplot(feature)
    plt.title("Histogram")

    plt.subplot(1, 2, 2)
    sns.boxplot(feature)
    plt.title("Boxplot")
    plt.show()

def categorical_distribution(feature):
    sns.barplot(feature)
    plt.show()

def numerical_correlation_matrix(feature):
    corr_df = feature.corr(method = "pearson")

    plt.figure(figsize = (12, 5))
    sns.heatmap(corr_df, annot = True, fmt = '.2f', cmap = 'coolwarm')
    plt.show()

def label_encoder(data, feature):
    le = LabelEncoder()
    data[feature] = le.fit_transform(data[feature])
    return data

def split_data(data, target_column):
    X = data.drop(columns = [target_column], axis = 1)
    y = data[target_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
    return X_train, X_test, y_train, y_test

def feature_scaling(X_train, X_test, features):
    scaler = PowerTransformer(method = "yeo-johnson")
    X_train[features] = scaler.fit_transform(X_train)
    X_test[features] = scaler.transform(X_test)
    return X_train, X_test

def feature_selection(model, param_grid, X_train, y_train):
    cv = KFold(n_splits = 5, shuffle = True, random_state = 42)
    model = GridSearchCV(model, 
                         param_grid, 
                         cv = cv, 
                         scoring = {"neg_mse": "neg_mean_squared_error", 
                                    "r2": "r2"}, 
                         refit = "neg_mse",
                         verbose = 0)
    model.fit(X_train, y_train)

    coef = pd.Series(model.best_estimator_.coef_, index = X_train.columns)

    plt.figure(figsize = (8, 5))
    coef.plot(kind = "bar", color = "navy")
    plt.title("Lasso Feature Selection")
    plt.xlabel("Features")
    plt.ylabel("Coefficient Values")
    plt.show()

def save_preprocessed_data(X_train, X_test, y_train, y_test, file_path):
    joblib.dump((X_train, X_test, y_train, y_test), file_path)


if __name__ == "__main__":
    data = load_data("data/mcdonalds.csv")
    data = assigned_meal_category(data)
    check_duplicates(data)
    check_nulls(data)
    data = replace_values(data)
    data = convert_data_types(data)

    # visualize the categorical data distribution
    categorical_distribution(data['Type'])

    # visualized the numerical data distribution 
    numerical_distribution(data['Calories'])
    numerical_distribution(data['Total Fat'])
    numerical_distribution(data['Saturated Fat'])
    numerical_distribution(data['Trans Fat'])
    numerical_distribution(data['Cholesterol'])
    numerical_distribution(data['Sodium'])
    numerical_distribution(data['Total Carbohydrates'])
    numerical_distribution(data['Dietary Fiber'])
    numerical_distribution(data['Sugars'])
    numerical_distribution(data['Added Sugars'])
    numerical_distribution(data['Protein'])

    # drop food menu - irrelevant feature 
    data = data.drop(['Food Menu'], axis = 1)

    # visualize the numerical correlation 
    numerical_correlation_matrix(data.drop(['Type'], axis = 1))

    # convert categorical features into labels 
    data = label_encoder(data, "Type")
    print(data.head())

    data['Type'] = data['Type'].astype('category')

    # split the data
    X_train, X_test, y_train, y_test = split_data(data, "Calories")

    # for lasso regression 
    param_grid_lasso = {'alpha': [0.001, 0.01, 0.1, 1, 10, 100], 
                        'fit_intercept': [True, False]}
    feature_selection(Lasso(), param_grid_lasso, X_train, y_train)
   
    # feature scaling on regressors
    features = ['Total Fat', 'Saturated Fat', 'Trans Fat', 'Total Carbohydrates', 'Dietary Fiber', 'Protein']
    X_train = X_train[features]
    X_test = X_test[features]
    X_train, X_test = feature_scaling(X_train, X_test, features)
    
    # save preprocessed files
    save_preprocessed_data(X_train, X_test, y_train, y_test, "preprocessed_data.pkl")
    
    
    

    
    
