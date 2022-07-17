import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential

def train(df, user_input):

    # splitting data for data training part
    data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.65)])

    # preprocessing the data
    scaler = MinMaxScaler(feature_range=(0,1))
    data_training_array = scaler.fit_transform(data_training)

    x_train = []
    y_train = []

    for i in range(100, data_training_array.shape[0]):
        x_train.append(data_training_array[i-100: i])
        y_train.append(data_training_array[i, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))

    # LSTM
    model=Sequential()
    model.add(LSTM(units=50,return_sequences=True,input_shape=(x_train.shape[1],1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50,return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    
    model.compile(optimizer= 'adam', loss = 'mean_squared_error')
    model.fit(x_train, y_train, epochs=50)

    # saving trained model
    model.save(user_input+'.h5')