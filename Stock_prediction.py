import Training_part as tp
from math import ceil
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import pandas_datareader as data
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
import streamlit as st
from plotly import express as  px
import datetime
from os.path import exists as file_exists

# selecting stock ticker
def select_stock():
    st.title('Stock price prediction')
    first, last = st.columns(2)
    start = first.date_input("Enter starting date",datetime.date(2015,1,1))
    end = last.date_input("Enter end date",datetime.date(2019,12,31))

    user_input = st.selectbox("Enter stock ticker",["AAPL","SBIN.NS","MSFT","AMZN","GOOGL"])
    try:
        df = data.DataReader(user_input,'yahoo', start, end)
        df.reset_index(inplace=True)
        return start,end,df,user_input
    except:
        st.error('Unable to fetch data from yahoo finance')

# showing graph of stock price
def show_graph(df):
    st.subheader('Closing price VS Time')
    fig = px.line(df,x="Date",y="Close")
    st.plotly_chart(fig)
        
# streamlit navigation slidebar
rad = st.sidebar.radio("Navigation", ["View stock data", "Predict", "Trained model"])

if rad == "View stock data":
    start, end, df, user_input = select_stock()
    # converting datetime to string
    d1 = start.strftime('%Y-%m-%d')
    d2 = end.strftime('%Y-%m-%d')

    # button for processing data
    if st.button("Process"):
        st.subheader('Data from ('+d1+') to ('+d2+')')
        st.write(df.describe())

        show_graph(df)


if rad == "Predict":
    start, end, df, user_input =select_stock()
    # converting datetime to string
    d1 = start.strftime('%Y-%m-%d')
    d2 = end.strftime('%Y-%m-%d')

    
    st.subheader('Graph from ('+d1+') to ('+d2+')')
    show_graph(df)

    if(len(df)<100):
        st.error('Diffrence between selected dates should be greater than 100 to predict the data')
    
    else:
        data = pd.DataFrame(df['Close'][:])

        scaler = MinMaxScaler(feature_range=(0,1))

        # Load model
        if file_exists(str(user_input)+'.h5'):
            model = load_model(str(user_input)+'.h5')
            st.subheader('Stock Prediction')
            val = st.number_input("Enter days",min_value=10, max_value=60)

            # taking past 100 days data for time series prediction
            past_100 = data.tail(100)
            predict_df = past_100.reset_index(drop = True)
            x_input = scaler.fit_transform(predict_df)
            x_input = x_input.reshape(1,-1)

            temp_input=x_input
            temp_input=temp_input[0].tolist()

            # predict
            lst_output=[]
            n_steps=100
            i=0
            while(i<val):
                
                if(len(temp_input)>100):
                    
                    x_input=np.array(temp_input[1:])
                    x_input=x_input.reshape(1,-1)
                    x_input = x_input.reshape((1, n_steps, 1))
                    # output for input data
                    yhat = model.predict(x_input, verbose=0)
                    temp_input.extend(yhat[0].tolist())
                    temp_input=temp_input[1:]

                    lst_output.extend(yhat.tolist())
                    i=i+1
                else:
                    x_input = x_input.reshape((1, n_steps,1))
                    yhat = model.predict(x_input, verbose=0)
                    temp_input.extend(yhat[0].tolist())
                    lst_output.extend(yhat.tolist())
                    i=i+1
            
            # predicted data
            predicted = scaler.inverse_transform(lst_output)
            day_new=np.arange(1,101)
            day_pred=np.arange(101,101+val)


            # output
            st.header("Output")
            df1=df.Close.tolist()
            fig = plt.figure(figsize=(15,9))
            plt.plot(day_new, df1[len(df1)-100:])
            plt.plot(day_pred, predicted)
            plt.xlabel('Time')
            plt.ylabel('Price')
            st.pyplot(fig)


            # Final output
            st.header("Final Output")
            fig = plt.figure(figsize = (15,9))
            df2 = df1[:]
            df1.extend(predicted)
            plt.plot(df1, 'r', label="Predicted trend")
            plt.plot(df2,'g', label="Stock price")
            plt.xlabel('Time')
            plt.ylabel('Price')
            plt.legend()
            st.pyplot(fig)

        else:
            st.error('Trained model is not avialable for this stock.....   Goto trained model section in navbar to train model')

if rad == "Trained model":
    start, end, df, user_input =select_stock()

    d1 = start.strftime('%Y-%m-%d')
    d2 = end.strftime('%Y-%m-%d')

    if st.button("Process"):
        st.subheader('Data from ('+d1+') to ('+d2+')')
        st.write(df.describe())

        show_graph(df)

        if(len(df)<100):
            st.error('Diffrence between selected dates should be greater than 150 to predict the data')
        else:
            # splitting data into training and testing part
            data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.65)])
            data_testing = pd.DataFrame(df['Close'][int(len(df)*0.65):int(len(df))])

            scaler = MinMaxScaler(feature_range=(0,1))

            # Load model
            if file_exists(str(user_input)+'.h5'):
                model = load_model(str(user_input)+'.h5')

                # Testing part
                past_100 = data_training.tail(100)
                final_df = pd.concat([past_100, data_testing]) 
                final_df = final_df.reset_index(drop = True)
                input_data = scaler.fit_transform(final_df)

                x_test = []

                for i in range(100, input_data.shape[0]):
                    x_test.append(input_data[i-100: i])

                x_test = np.array(x_test)
                x_test = np.reshape(x_test, (x_test.shape[0],x_test.shape[1],1))


                # Model Prediction
                last=df.tail(ceil(len(df)*(0.35)))
                y_predict = model.predict(x_test)
                y_predict = scaler.inverse_transform(y_predict)
                y_predict = y_predict.tolist()

                # store nparray data to list
                predicted = []
                for i in y_predict:
                    predicted.append(i[0])

                #adding predicted data col with value
                last['Predict']=predicted

                # Testing output
                st.subheader('Prediction VS orignal')
                fig2 = px.line(last,x="Date",y=["Close","Predict"], title="Closing_price")
                st.plotly_chart(fig2)
            
            else:
                st.error("Trained model is not available for this Stock... Reload button will appear after model is trained")
                tp.train(df,str(user_input))
                st.button("Reload")
