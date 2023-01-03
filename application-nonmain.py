#from matplotlib.pyplot import scatter
import streamlit as st
import pandas as pd
import seaborn as sns
#import plotly.express as px
from app_functions import getDataTypes, changeDataType, heatmap, save_model
from app_functions import createModel, dropColumn, createGraph, labelEncode, splitData, train
#import statsmodels.api as sm

st.set_page_config(layout='wide')
EDA_column, settings_column = st.columns((4,1))

EDA_column.title('Data Science Tool')

settings_column.title('Settings')

uploaded_file = settings_column.file_uploader('Choose a File')

def main(df):
    #Assigning data_type to the datatypes of the uploaded dataframe
    data_type = getDataTypes(df)
    #Presenting the user their uploaded dataframe / datatypes of its columns
    EDA_column.write('Below displays the datatype of your dataframe')
    EDA_column.write(data_type)

    options = ['Choose an option']
    for i in df.columns:
        options.append(i)
    #Allowing user to select columns they would like to change the data type of and desired datatype
    selected_column = EDA_column.selectbox('Select the column you would like to change the data type of', options = options, index = 0)
    dtypes = ['object', 'int64', 'float64', 'bool', 'category'] #try to integrate timedelta and datetime datatypes at some point
    desired_dtype = EDA_column.selectbox('Select desired data type', options = dtypes)
    
    #Changed_datatype is assigned to a returned df from the chageDataType function that represents the dataframe with new datatypes
    try:
        changed_datatype = changeDataType(df,selected_column,desired_dtype)
        EDA_column.write(changed_datatype)
        EDA_column.write('Successfully changed '+str(selected_column)+' to type '+str(desired_dtype))
    except:
        pass
    #Drop Columns section
    drop_choice = EDA_column.multiselect('Select the columns you would like to drop', options = df.columns)
    df = dropColumn(df,drop_choice,EDA_column)
    EDA_column.write('Updated Dataframe')
    EDA_column.write(df)

    #LABEL ENCODING
    label_encoding_columns = EDA_column.multiselect('Select the categorical colums you would like label encoded', options = df.columns)
    labelEncode(df, label_encoding_columns)
    EDA_column.write(df)


    #Correlation Heatmap
    #CHI SQUARE IS CORRELATION FOR CATEGORICAL VARIABLES. INCLUDE THIS.
    EDA_column.write('Heatmap of Correlation')
    heatmap(df, EDA_column)

    #Further Visualization / GET INPUT, SHOULD USERS BE ABLE TO DISPLAY MORE THAN ONE CHART SIMULTANEOUSLY??
    chosen_columns =  EDA_column.multiselect('Select at least 2 columns you would like to see visualized', options = df.columns)
    color_choice = EDA_column.selectbox('Choose Categorical Variable', options = df.columns)
    chosen_graphtype = EDA_column.selectbox('Select the type of graph', options = ['Scatter Plot', 'Line Plot', 'Bar Chart', 'PCA'])
    try:
        createGraph(df, chosen_columns, color_choice, chosen_graphtype, EDA_column)
    except:
        pass
    problem_type = EDA_column.selectbox('Type of Problem', options = ['None', 'Classification', 'Regression'])
    #st.write(df)
    if problem_type == 'None':
        EDA_column.write('Please select a problem type')
    if problem_type == 'Classification':
        selected_model = EDA_column.selectbox('Classification Models', options = ['None','KNN','LogReg', 'SVM', 'Decision Tree'])
    elif problem_type == 'Regression':
        selected_model = EDA_column.selectbox('Regression Models', options = ['None','Linear', 'Ridge', 'Lasso'])
    x_train, x_test, y_train, y_test = splitData(df, EDA_column)
    model = createModel(df, problem_type, selected_model, EDA_column)
    trained_model, model_yhat = train(model,x_train, x_test, y_train, y_test, problem_type, EDA_column)
    save_model(trained_model, EDA_column)


if uploaded_file is not None:
    #Reading uploaded file
    df  = pd.read_csv(uploaded_file)
    main(df)
    # #Assigning data_type to the datatypes of the uploaded dataframe
    # data_type = getDataTypes(df)
    # #Presenting the user their uploaded dataframe / datatypes of its columns
    # EDA_column.write('Below displays the datatype of your dataframe')
    # EDA_column.write(data_type)

    # options = ['Choose an option']
    # for i in df.columns:
    #     options.append(i)
    # #Allowing user to select columns they would like to change the data type of and desired datatype
    # selected_column = EDA_column.selectbox('Select the column you would like to change the data type of', options = options, index = 0)
    # dtypes = ['object', 'int64', 'float64', 'bool', 'category'] #try to integrate timedelta and datetime datatypes at some point
    # desired_dtype = EDA_column.selectbox('Select desired data type', options = dtypes)
    
    # #Changed_datatype is assigned to a returned df from the chageDataType function that represents the dataframe with new datatypes
    # try:
    #     changed_datatype = changeDataType(df,selected_column,desired_dtype)
    #     EDA_column.write(changed_datatype)
    #     EDA_column.write('Successfully changed '+str(selected_column)+' to type '+str(desired_dtype))
    # except:
    #     pass
    # #Drop Columns section
    # drop_choice = EDA_column.multiselect('Select the columns you would like to drop', options = df.columns)
    # df = dropColumn(df,drop_choice,EDA_column)
    # EDA_column.write('Updated Dataframe')
    # EDA_column.write(df)

    # #LABEL ENCODING
    # label_encoding_columns = EDA_column.multiselect('Select the categorical colums you would like label encoded', options = df.columns)
    # labelEncode(df, label_encoding_columns)
    # EDA_column.write(df)


    # #Correlation Heatmap
    # #CHI SQUARE IS CORRELATION FOR CATEGORICAL VARIABLES. INCLUDE THIS.
    # EDA_column.write('Heatmap of Correlation')
    # heatmap(df, EDA_column)

    # #Further Visualization / GET INPUT, SHOULD USERS BE ABLE TO DISPLAY MORE THAN ONE CHART SIMULTANEOUSLY??
    # chosen_columns =  EDA_column.multiselect('Select at least 2 columns you would like to see visualized', options = df.columns)
    # color_choice = EDA_column.selectbox('Choose Categorical Variable', options = df.columns)
    # chosen_graphtype = EDA_column.selectbox('Select the type of graph', options = ['Scatter Plot', 'Line Plot', 'Bar Chart', 'PCA'])
    # try:
    #     createGraph(df, chosen_columns, color_choice, chosen_graphtype, EDA_column)
    # except:
    #     pass
    # problem_type = EDA_column.selectbox('Type of Problem', options = ['None', 'Classification', 'Regression'])
    # #st.write(df)
    # if problem_type == 'None':
    #     EDA_column.write('Please select a problem type')
    # if problem_type == 'Classification':
    #     selected_model = EDA_column.selectbox('Classification Models', options = ['None','KNN','LogReg', 'SVM', 'Decision Tree'])
    # elif problem_type == 'Regression':
    #     selected_model = EDA_column.selectbox('Regression Models', options = ['None','Linear', 'Ridge', 'Lasso'])
    # x_train, x_test, y_train, y_test = splitData(df, EDA_column)
    # model = createModel(df, problem_type, selected_model, EDA_column)
    # trained_model, model_yhat = train(model,x_train, x_test, y_train, y_test, problem_type, EDA_column)
    # save_model(trained_model, EDA_column)

    #TODO
    #ADD HISTOGRAMS 
    #COULD MAYBE DO SOME SORT OF NUTS FOR LOOP THAT LOOPS THROUGH ALL COLUMNS IN THE DF, TRAINS AND RUNS THE MODEL, AND RETURNS THE MOST IMPORTANT PREDICTOR COLUMNS.
    # IN-PROGRESS ALLOW USERS TO CREATE VARIOUS ML MODELS AND VISUALIZE PERFORMANCE.
    # allow users to Compare the performance of the various models

else:
    EDA_column.header('Please Upload A .csv File Or Choose From The Sample Data Below To Begin')
    dataset_names = sns.get_dataset_names()
    chosen_dataset = EDA_column.selectbox('Sample Datasets:', options = dataset_names)
    df = sns.load_dataset(chosen_dataset)
    main(df)
