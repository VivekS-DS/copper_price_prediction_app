# Streamlit app for copper price prediction

# Importing necessary libraries for data manipulation and analysis
import pandas as pd
import numpy as np

# Importing tools for splitting data into training and testing sets
from sklearn.model_selection import train_test_split

# Importing SMOTE-Tomek for handling imbalanced datasets
from imblearn.combine import SMOTETomek

# Importing machine learning models for classification
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor

from datetime import date
import streamlit as st

st.set_page_config(layout='wide')

st.title('Copper Price Prediction')


col1, col2, col3 = st.columns(3)

with col1:

    item_date = st.date_input(label='Item Date', min_value=date(2020,1,1), 
                                            max_value=date(2021,12,31), value=date(2020,6,15))

    customer = st.slider('Customer',12458, 30408185)
    country = st.selectbox('Country',('25', '26', '27', '28', '30', '32', '38', '39', '40', '77', '78', '79', '80', '84', '89', '107', '113'))
    item_type = st.selectbox('item_type',('3', '2', '0', '1', '4'))

    application = st.selectbox('application',('2', '3', '4', '5', '10', '15', '19', '20', '22', '25', '26', '27', '28', '29', 
                                                  '38', '39', '40', '41', '42', '56', '58', '59', '65', '66', '67', '68', '69', '70', '79', '99'))
        
    product_ref = st.selectbox('Product Reference',('611728', '611733','611993', '628112', '628117', '628377',
                                            '640400', '640405','640665','164141591','164336407', '164337175','1282007633',
                                            '1332077137','1665572032', '1665572374','1665584642','1668701376','1668701698',
                                            '1668701718', '1668701725', '1670798778', '1671863738','1671876026','1690738206',
                                            '1690738219','1693867550','1693867563','1721130331','1722207579'))
    
with col3:
    

    delivery_date = st.date_input(label='Item Date', min_value=date(2020,1,1), 
                                            max_value=date(2021,7,31), value=date(2020,7,15))

    selling_price = st.number_input('selling_price (Min:₹250  to Max:₹82,0000)')   
    quantity_tons = st.number_input(label='quantity_tons (Min:1 ton to Max:80,0000)')
    thickness = st.number_input('thickness (Min: 0.18 mm to Max:400 mm)')
    width = st.number_input('width (Min:1 mm to Max:3000 mm)')
    status = st.selectbox('Select Status to predict Selling Price',(0,1))



def user_input_class():

    user_data_class ={'customer': customer, 
                    'country':country, 
                    'item_type':item_type, 
                    'application': application, 
                    'width':width,
                    'product_ref': product_ref, 
                    'quantity_tons_log_tr':np.log(quantity_tons), 
                    'thickness_log_tr':np.log(thickness+ 1e-10) ,
                    'selling_price_log_tr':(np.log(selling_price+ 1e-10)), 
                    'item_date_day':item_date.day , 
                    'item_date_month':item_date.month,
                    'item_date_year':item_date.year , 
                    'delivery_date_day':delivery_date.day , 
                    'delivery_date_month':delivery_date.month,
                    'delivery_date_year':delivery_date.year,
                    }
    
    features_class = pd.DataFrame(user_data_class,index=[0])
    return features_class



def ml_class_model():
    # Reading a CSV file ('copper_cleaned.csv') into a pandas DataFrame
    data = pd.read_csv('copper_clean_dataset.csv')
    
    # Separating features (X) and target variable (y) from the DataFrame 'copper4'
    x = data.drop('status', axis=1)
    y = data['status']

    # Applying SMOTE-Tomek resampling technique to handle imbalanced classes
    x_ovf, y_ovf = SMOTETomek().fit_resample(x, y)

    # Splitting the resampled data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x_ovf,y_ovf,test_size=0.2,random_state=42)
    # Creating a RandomForestClassifier with specified hyperparameters
    model_rfc = RandomForestClassifier(criterion='entropy',
                                        max_features='sqrt',
                                        max_samples=None,
                                        min_samples_split=2).fit(x_train, y_train)
    return model_rfc

with col2:
    df_input_class = user_input_class()
    user_submit = st.button('Predict Status')

    # applying the model to make prediction
    if user_submit == True:
        
        model_class = ml_class_model()
        predict_status = model_class.predict(df_input_class)
        

        if predict_status == 1: 
            st.success('Status: Won 👍')
        else: st.write('Status: Lost 👎')

        
def user_input_reg():

    user_data_reg ={'customer': customer, 
                    'country':country,
                    'status':status, 
                    'item_type':item_type, 
                    'application': application, 
                    'width':width,
                    'product_ref': product_ref, 
                    'quantity_tons_log_tr':np.log(quantity_tons), 
                    'thickness_log_tr':np.log(thickness+ 1e-10) ,
                    'item_date_day':item_date.day , 
                    'item_date_month':item_date.month,
                    'item_date_year':item_date.year , 
                    'delivery_date_day':delivery_date.day , 
                    'delivery_date_month':delivery_date.month,
                    'delivery_date_year':delivery_date.year,
                    }
    
    features_reg = pd.DataFrame(user_data_reg,index=[0])
    return features_reg

def ml_reg_model():
    # Reading a CSV file ('copper_cleaned.csv') into a pandas DataFrame
    datas = pd.read_csv('copper_clean_dataset.csv')
    
    # Separating features (X) and target variable (y) from the DataFrame 'copper4'
    x_reg = datas.drop('selling_price_log_tr', axis=1)
    y_reg = datas['selling_price_log_tr']

    # Splitting the resampled data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x_reg,y_reg,test_size=0.2,random_state=42)
    
    # Creating a RandomForestRegressor model
    model_rfr = RandomForestRegressor(criterion='squared_error',
                             max_features=None,
                             max_samples=None,
                             min_samples_split=4,
                             min_samples_leaf=1).fit(x_train, y_train)


    return model_rfr

with col2:
    df_input_reg = user_input_reg()
    user_submit_reg = st.button('Predict Selling Price')

    # applying the model to make prediction
    if user_submit_reg == True:
        
        model_reg = ml_reg_model()
        predict_price = np.exp(model_reg.predict(df_input_reg))
        st.write('Selling Price ₹',predict_price)

        