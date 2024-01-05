
# Importing necessary libraries for the Streamlit app
from datetime import date
import numpy as np 
import gdown
import pickle
import streamlit as st
import os

# Setting the title for the Streamlit app
st.title('Copper Price Prediction APP')

    
# Function to load the pre-trained classification model
def load_data_class():
    # URL for the pre-trained classification model
    url_class = "https://onedrive.live.com/download?resid=F275F26477A8CF4E%21345968&authkey=!ACykorlc012lPIs"
    # Output file for saving the downloaded model
    output_class = "copper_class.pkl"
    # Downloading the model only if it doesn't exist locally
    if not os.path.exists(output_class):
        gdown.download(url_class, output_class, quiet=False)
    return

# Function to load the pre-trained regression model
def load_data_reg():
    # URL for the pre-trained regression model
    url_reg = "https://onedrive.live.com/download?resid=F275F26477A8CF4E%21362322&authkey=!AO_39icVBpNOq44"
    # Output file for saving the downloaded model
    output_reg = "copper_reg.pkl"
    # Downloading the model only if it doesn't exist locally
    if not os.path.exists(output_reg):
        gdown.download(url_reg, output_reg, quiet=False)
    return


class prediction:
    # Method for regression prediction
    def regression():
        # Creating a form for user input in the Streamlit app
        with st.form('regression'):
            col1, col2, col3 = st.columns(3)
        # First column of the form
        with col1:
            # User input 
            item_date = st.date_input(label='Item Date', min_value=date(2020,1,1), 
                                            max_value=date(2021,12,31), value=date(2020,6,15))

            customer = st.number_input('Customer',12458, 30408185)
            country = st.selectbox('Country',('25', '26', '27', '28', '30', '32', '38', '39', '40', '77', '78', '79', '80', '84', '89', '107', '113'))
            item_type = st.selectbox('item_type',('3', '2', '0', '1', '4'))

            application = st.selectbox('application',('2', '3', '4', '5', '10', '15', '19', '20', '22', '25', '26', '27', '28', '29', 
                                                        '38', '39', '40', '41', '42', '56', '58', '59', '65', '66', '67', '68', '69', '70', '79', '99'))
                
            product_ref = st.selectbox('Product Reference',('611728', '611733','611993', '628112', '628117', '628377',
                                            '640400', '640405','640665','164141591','164336407', '164337175','1282007633',
                                            '1332077137','1665572032', '1665572374','1665584642','1668701376','1668701698',
                                            '1668701718', '1668701725', '1670798778', '1671863738','1671876026','1690738206',
                                            '1690738219','1693867550','1693867563','1721130331','1722207579'))

        # Third column of the form
        with col3:
            # User input
            delivery_date = st.date_input(label='Delivery Date', min_value=date(2020,1,1), 
                                                    max_value=date(2021,7,31), value=date(2020,7,15))
            quantity_tons = st.number_input(label='quantity_tons (Min:1 ton to Max:80,0000)',value=1)
            thickness = st.number_input('thickness (Min: 0.18 mm to Max:400 mm)',value=0.18)
            width = st.number_input('width (Min:1 mm to Max:3000 mm)',value=1)
            cust_status = st.selectbox('Customer Status',(0,1))

        # Second column of the form
        with col2:
            # Button to submit the form
            button = st.form_submit_button('Regression')
            # Executed when the form is submitted
            if button:
                # Load the pre-trained regression model
                load_data_reg()

                # Read the model from the pickle file
                with open('copper_reg.pkl', 'rb') as f:
                    model_rfr = pickle.load(f)

                # Create a numpy array with user input data
                user_data_reg = np.array([[customer, country, item_type, application, width, product_ref,
                                np.log(float(quantity_tons + 1e-10)), np.log(float(thickness + 1e-10)),
                                item_date.day, item_date.month,item_date.year, delivery_date.day, 
                                delivery_date.month, delivery_date.year,cust_status]])
                
                # Predict the selling price using the regression model
                predict_reg = model_rfr.predict(user_data_reg)
                # Convert the log-transformed prediction back to the original price
                price = np.exp(predict_reg)

                return price

    

    def classification():
        # Creating a form for user input in the Streamlit app
        with st.form('classification'):
            col1, col2, col3 = st.columns(3)
        
        # First column of the form
        with col1:
            # User input 
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
    
        # Third column of the form
        with col3:
            # User input
            delivery_date = st.date_input(label='Item Date', min_value=date(2020,1,1), 
                                                    max_value=date(2021,7,31), value=date(2020,7,15))

            selling_price = st.number_input('selling_price (Min:₹250  to Max:₹82,0000)', value=250)   
            quantity_tons = st.number_input(label='quantity_tons (Min:1 ton to Max:80,0000)',value=1)
            thickness = st.number_input('thickness (Min: 0.18 mm to Max:400 mm)',value=0.18)
            width = st.number_input('width (Min:1 mm to Max:3000 mm)',value=1)
            

        # Second column of the form
        with col2:
            # Button to submit the form
            button = st.form_submit_button('Classification')

            # Executed when the form is submitted
            if button:
                # Load the pre-trained classification model
                load_data_class()
                with open('copper_class.pkl', 'rb') as f:
                    model_rfc = pickle.load(f)

                # Create a numpy array with user input data
                user_data_class = np.array([[customer, country, item_type, application, width, product_ref,
                                np.log(float(quantity_tons + 1e-10)), np.log(float(thickness + 1e-10)),
                                np.log(float(selling_price + 1e-10)), item_date.day, item_date.month,
                                item_date.year, delivery_date.day, delivery_date.month, delivery_date.year]])
                
                # Predict the status using the classification model
                predict_y = model_rfc.predict(user_data_class)
                status = predict_y[0]

                return status



# Creating two tabs for Regression and Classification
tab1,tab2 = st.tabs(['Regression','Classification'])  

# Code for the Regression tab
with tab1:
    # Calling the regression method and displaying the predicted selling price
    price = prediction.regression()
    st.write('Selling Price ₹',price)

# Code for the Classification tab
with tab2:
    # Calling the classification method and displaying the predicted customer status
    status = prediction.classification()
    if status == 1:
        st.write('Customer Status Won 👍')
    else: st.write('Customer Status Lost 👎')

#-------------------------------------End of Code -------------------------------------------------------