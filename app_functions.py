import streamlit as st
import pandas as pd
import pickle

from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso

from sklearn.metrics import f1_score, mean_absolute_error, mean_squared_error, r2_score

import plotly.express as px
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm

def getDataTypes(df):
    data_types = df.dtypes
    return data_types

def changeDataType(df,selected_column,desired_dtype):
    df[selected_column] = df[selected_column].astype(desired_dtype)
    #st.write('Successfully changed '+str(selected_column)+' to type '+str(desired_dtype))
    data_types = df.dtypes
    return data_types #We return the data types to show the user the new, updated datatypes now that they have been changed.

def getCorrelation(df,corr_columns):
    correlation = df[corr_columns[0]].corr(df[corr_columns[1]])
    return correlation

def createGraph(df, selected_columns, categorical_variable, chosen_graph, web_column):
    #scatter
    if chosen_graph == 'Scatter Plot':
        regline = web_column.checkbox('Regression Line')
        if regline:
            web_column.plotly_chart(px.scatter(data_frame= df, x=df[selected_columns[0]], y=df[selected_columns[1]], color = categorical_variable, trendline = 'ols', height = 800), use_container_width=True)
        else:
            web_column.plotly_chart(px.scatter(data_frame= df, x=df[selected_columns[0]], y=df[selected_columns[1]], color = categorical_variable, height = 800), use_container_width=True)
    #line
    elif chosen_graph == 'Line Plot':
        web_column.plotly_chart(px.line(data_frame=df, x=df[selected_columns[0]], y= df[selected_columns[1]], color = categorical_variable, height = 800), use_container_width=True)
    #bar
    elif chosen_graph == 'Bar Chart':
        web_column.plotly_chart(px.bar(data_frame=df, x=df[selected_columns[0]], y= df[selected_columns[1]], color = categorical_variable, height = 800), use_container_width=True)
    #pca
    elif chosen_graph == 'PCA':
        pca_data, categorical_cols, pca_cols = CreatePCA(df)
        catgorical_variable = web_column.selectbox('Variable Selector', options= categorical_cols)
        categorical_variable2 = web_column.selectbox('Second Variable Selector', options= categorical_cols)
        pca_1 = web_column.selectbox('First Principle Component', options= pca_cols)
        pca_cols.remove(pca_1)
        pca_2 = web_column.selectbox('Second Principle Component', options= pca_cols)
        web_column.plotly_chart(px.scatter(data_frame= pca_data, x=pca_1, y=pca_2, color = catgorical_variable, height = 800, hover_data= [categorical_variable2]), use_container_width=True)

def labelEncode(df, selected_columns):
    labelencoder = LabelEncoder()
    for i in range(len(selected_columns)):
        df[selected_columns[i]] = labelencoder.fit_transform(df[selected_columns[i]])
    
def heatmap(df, web_column):
    total_correlation = df.corr()
    fig = px.imshow(total_correlation, text_auto = True)
    web_column.plotly_chart(fig, use_container_width=True)

def dropColumn(df, selected_column, web_column):
    for i in range(len(selected_column)):
        df = df.drop(columns = selected_column[i])
    return df

def splitData(df, dependent_var, independent_vars, standardize, normalize):
    Y = df[dependent_var].to_numpy()
    X = df[independent_vars].to_numpy()
    if normalize:
        normalizer = Normalizer()
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.2,random_state=2)
        X_train_scaled = normalizer.fit_transform(X_train)
        X_test_scaled = normalizer.fit_transform(X_test)
        return X_train_scaled, X_test_scaled, Y_train, Y_test

    elif standardize:
        scaler = preprocessing.StandardScaler()
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.2,random_state=2)
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.fit_transform(X_test)
        return X_train_scaled, X_test_scaled, Y_train, Y_test
    else:
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.2,random_state=2 )
        return X_train, X_test, Y_train, Y_test

def createModel(selected_model, web_column):
        if selected_model == 'KNN':
            cv_slider = web_column.slider('CV Amount', min_value = 1, max_value = 15, step = 1)
            neighbor_input = int(web_column.text_input('Number of Neighbors You Would Like To Test'))
            neighbors = list(range(1, neighbor_input+1))
            algorithms = ['auto', 'ball_tree', 'kd_tree', 'brute']
            parameters = {'n_neighbors': neighbors,'algorithm': algorithms, 'p': [1, 2]}
            button = web_column.button('Run')
            if button:
                KNN = KNeighborsClassifier()
                KNN_cv = GridSearchCV(KNN, parameters, cv= cv_slider, refit=True)
                return KNN_cv
        elif selected_model == 'LogReg':
            cv_slider = web_column.slider('CV Amount', min_value = 1, max_value = 15, step = 1)
            parameters = {'C': np.logspace(-3, 3, 5), 'penalty': ['l1', 'l2', 'elasticnet'], 
            'solver': ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga']}
            button = web_column.button('Run')
            if button:
                logreg = LogisticRegression(random_state = 2)
                logreg_cv = GridSearchCV(logreg, parameters, cv=cv_slider, refit=True)
                return logreg_cv
        elif selected_model == 'SVM (Experimental)':
            cv_slider = web_column.slider('CV Amount', min_value = 1, max_value = 15, step = 1)
            parameters = {'kernel':('linear','rbf','poly','sigmoid'), 'C': np.logspace(-3, 3, 5), 'gamma': np.logspace(-3, 3, 5)}
            button = web_column.button('Run')
            if button:
                svm = SVC(probability=True, random_state=3)
                svm_cv = GridSearchCV(svm, parameters, cv = cv_slider, refit=True)
                return svm_cv
        elif selected_model == 'Decision Tree':
            cv_slider = web_column.slider('CV Amount', min_value = 1, max_value = 15, step = 1)
            parameters = {'criterion': ['gini', 'entropy'],
                'splitter': ['best', 'random'],
                'max_depth': [2*n for n in range(1,10)],
                'max_features': ['sqrt'],
                'min_samples_leaf': [1,2,4],
                'min_samples_split' : [2,5,10]}
            button = web_column.button('Run')
            if button:
                tree = DecisionTreeClassifier(random_state=4)
                tree_cv = GridSearchCV(tree, parameters, cv=cv_slider, refit=True)
                return tree_cv
        elif selected_model == 'Linear':
            button = web_column.button('Run')
            if button:
                linreg = LinearRegression()
                return linreg
        elif selected_model == 'Ridge':
            cv_slider = web_column.slider('CV Amount', min_value = 1, max_value = 15, step = 1)
            parameters = {'alpha': [0.01, 0.1, 1, 10, 100],
                        'fit_intercept': [True, False],
                        'copy_X': [True, False],
                        'max_iter': [1000, 2000, 3000, 4000, 5000],
                        'tol': [1e-4, 1e-3, 1e-2, 1e-1],
                        'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga', 'lbfgs'],
                        'random_state': [None, 42]}
            button = web_column.button('Run')
            if button:
                ridge = Ridge()
                ridge_cv = GridSearchCV(ridge, parameters, cv=cv_slider)
                return ridge_cv
        elif selected_model == 'Lasso':
            cv_slider = web_column.slider('CV Amount', min_value = 1, max_value = 15, step = 1)
            lasso  = Lasso()
            parameters = {'alpha': [0.01, 0.1, 1, 10, 100],
                        'fit_intercept': [True, False],
                        'precompute': [True, False],
                        'copy_X': [True, False],
                        'max_iter': [1000, 2000, 3000, 4000, 5000],
                        'tol': [1e-4, 1e-3, 1e-2, 1e-1],
                        'warm_start': [True, False],
                        'positive': [True, False],
                        'selection': ['cyclic', 'random'],
                        'random_state': [None, 42]}
            button = web_column.button('Run')
            if button:
                lasso_cv = GridSearchCV(lasso, parameters, cv=cv_slider)
                return lasso_cv

def train(model, x_train, x_test, y_train, y_test, problem_type, web_column):
    trained_model = model.fit(x_train, y_train)
    model_yhat = model.predict(x_test)

    if problem_type == 'Classification':
        model_micro_f1 = f1_score(y_test, model_yhat, pos_label='positive', average = 'micro')
        model_macro_f1 = f1_score(y_test, model_yhat, pos_label='positive', average = 'macro')
        model_weighted_f1 = f1_score(y_test, model_yhat, pos_label='positive', average = 'weighted')   
        model_eval_data = {'Micro F1-Score / Accuracy': [model_micro_f1], 
        'Macro F1-Score': [model_macro_f1], 'Weighted F1-Score': [model_weighted_f1]}
        model_eval_df = pd.DataFrame(model_eval_data, index=['Your Model']).sort_values(by=['Micro F1-Score / Accuracy'], ascending=False)
        model_eval_df.round(3)
        web_column.write(model_eval_df)
        web_column.plotly_chart(px.bar(data_frame=model_eval_df, x=model_eval_df.index, y= model_eval_df.columns, height = 500))
    elif problem_type == 'Regression':
        model_mae = mean_absolute_error(y_test,model_yhat)
        model_mse = mean_squared_error(y_test,model_yhat)
        model_rmse = np.sqrt(model_mse)
        model_rmsle= np.log(model_rmse)
        model_r2= r2_score(y_test, model_yhat)
        model_acc=model.score(x_test, y_test)

        model_eval_data = {'Accuracy': [model_acc],'R^2': [model_r2], 
        'RMSE': [model_rmse], 'RMSLE': [model_rmsle], 'MAE': [model_mae]}
        model_eval_df = pd.DataFrame(model_eval_data, index=['Your Model']).sort_values(by=['RMSE'], ascending=False)
        model_eval_df.round(3)
        web_column.write(model_eval_df)
        web_column.plotly_chart(px.bar(data_frame=model_eval_df, x=model_eval_df.index, y= model_eval_df.columns, height = 500))
    return trained_model

def save_model(model, web_column):
    web_column.download_button(
    "Download Model",
    data=pickle.dumps(model),
    file_name="model.pkl")

def CreatePCA(df):
    numerical_column_list = []
    categorical_column_list = []

    for i in df.columns:
        if df[i].dtype == np.dtype('float64') or df[i].dtype == np.dtype('int64'):
            numerical_column_list.append(df[i])
        else:
            categorical_column_list.append(df[i])

    numerical_df = pd.concat(numerical_column_list, axis=1)
    categorical_df = pd.concat(categorical_column_list, axis=1)

    numerical_df.apply(lambda x: x.fillna(np.mean(x)))

    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(numerical_df)
    pca = PCA()
    pca_data = pca.fit_transform(scaled_values)
    pca_data = pd.DataFrame(pca_data)

    new_columns_names = ['PCA_'+str(i) for i in range(1, len(pca_data.columns)+1)]

    column_mapper = dict(zip(list(pca_data.columns), new_columns_names))

    pca_data = pca_data.rename(columns=column_mapper)

    output_df = pd.concat([df, pca_data], axis = 1)
    return output_df, list(categorical_df.columns), new_columns_names



#add some more models and a confusion matrix.


# def plot_confusion_matrix(y,y_predict, web_column):
#     "this function plots the confusion matrix"

#     cm = confusion_matrix(y, y_predict)
#     ax= plt.subplot()
#     sns.heatmap(cm, annot=True, ax = ax); #annot=True to annotate cells
#     ax.set_xlabel('Predicted labels')
#     ax.set_ylabel('True labels')
#     ax.set_title('Confusion Matrix'); 
#     ax.xaxis.set_ticklabels(['0', '1']); ax.yaxis.set_ticklabels(['0', '1'])
#     plt.show()


# def show_values(axs, orient="v", space=.01):
#     def _single(ax):
#         if orient == "v":
#             for p in ax.patches:
#                 _x = p.get_x() + p.get_width() / 2
#                 _y = p.get_y() + p.get_height() + (p.get_height()*0.01)
#                 value = '{:.3f}'.format(p.get_height())
#                 ax.text(_x, _y, value, ha="center") 
#         elif orient == "h":
#             for p in ax.patches:
#                 _x = p.get_x() + p.get_width() + float(space)
#                 _y = p.get_y() + p.get_height() - (p.get_height()*0.5)
#                 value = '{:.3f}'.format(p.get_width())
#                 ax.text(_x, _y, value, ha="left")

#     if isinstance(axs, np.ndarray):
#         for idx, ax in np.ndenumerate(axs):
#             _single(ax)
#     else:
#         _single(axs)