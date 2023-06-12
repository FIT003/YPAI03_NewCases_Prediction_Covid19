"""
Assessment_1: predict new cases (cases_new) in Malaysia
URL: https://github.com/MoH-Malaysia/covid19-public
"""
#%%
# 1. Import packages
import pandas as pd
import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
import tensorflow 
from tensorflow.keras import Sequential,Input
from tensorflow.keras.layers import LSTM,Dropout,Dense
from tensorflow.keras.utils import plot_model

from tensorflow.keras.callbacks import TensorBoard
name = "tensorboard_assessment_1"

#%%
#2. Data loading
PATH = os.getcwd()
CSV_PATH_TRAIN = os.path.join(PATH,"cases_malaysia_train.csv")
CSV_PATH_TEST = os.path.join(PATH,"cases_malaysia_test.csv")

df_train = pd.read_csv(CSV_PATH_TRAIN)
df_test = pd.read_csv(CSV_PATH_TEST)

tensorboard = TensorBoard(log_dir='logdir\\{}'.format(name))

#%%
#3. Data inspection
df_train.head()

#%%
df_train.info()

#%%
df_test.info()

#%%
#find amount of NaN in df_train
df_train.isna().sum()

#%%
#find amount of NaN in df_test
df_test.isna().sum()

#%%
#4. Data cleaning
df_train["cases_new"].dtype

#%%
#convert data type of column to numeric by using coerce to replcae non-numeric values into NaN
df_train['cases_new'] = pd.to_numeric(df_train['cases_new'], errors='coerce')
df_train['cases_new'].dtype

#%%
df_train['cases_new'].isna().sum()

#%%
#use interpolate to fill in NaN elements
df_train['cases_new'] = df_train['cases_new'].interpolate()

#%%
print("datatype: ",df_train['cases_new'].dtype)
print("number of NaN: ", df_train['cases_new'].isna().sum())

#%%
df_test['cases_new'] = df_test["cases_new"].interpolate()

#%%
df_test['cases_new'].isna().sum()

#%%
#plot graph to see the cases_new
df_disp = df_train['cases_new']
plt.figure()
plt.plot(df_disp)
plt.show()

#%%
#5.feature selection
df_train_newcases = df_train['cases_new']
df_test_newcases = df_test['cases_new']

# %%
mms = MinMaxScaler()
df_train_newcases_scaled = mms.fit_transform(np.expand_dims(df_train_newcases,axis=1))
df_test_newcases_scaled = mms.fit_transform(np.expand_dims(df_test_newcases,axis=1))

#  7. Data Windowing
#window size set to 30 days
window_size = 30
X_train = []
y_train = []
for i in range(window_size, len(df_train_newcases_scaled)):
    X_train.append(df_train_newcases_scaled[i-window_size:i])
    y_train.append(df_train_newcases_scaled[i])

X_train = np.array(X_train)
y_train = np.array(y_train)
#%% 
df_newcases_stacked = np.concatenate((df_train_newcases_scaled, df_test_newcases_scaled))
length_days = window_size + len(df_test_newcases_scaled)
data_test = df_newcases_stacked[-length_days:]

X_test = []
y_test = []
for i in range(window_size, len(data_test)):
    X_test.append(data_test[i-window_size:i])
    y_test.append(data_test[i])

X_test = np.array(X_test)
y_test = np.array(y_test)
#%%
#  8. Model Development
model = Sequential()
model.add(Input(shape=X_train.shape[1:]))
model.add(LSTM(64, return_sequences=True))
model.add(Dropout(0.5))
model.add(LSTM(64, return_sequences=True))
model.add(Dropout(0.5))
model.add(LSTM(64, return_sequences=True))
model.add(Dropout(0.5))
model.add(LSTM(64))
model.add(Dropout(0.5))
model.add(Dense(1,activation='relu'))


model.summary()
plot_model(model, show_shapes=True, show_layer_names=True)
#%%
#  9. Model Compile
#MAPE is use as required in criteria
model.compile(optimizer='adam', loss='mse', metrics=['mae', 'mape'])
#%%
#  10. Model Training
history = model.fit(X_train, y_train, epochs=100, validation_split=0.2, callbacks=[tensorboard])
#%%
# 11. Model Evaluation
print(history.history.keys())

#%%
plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['Train loss', "Test loss"])
plt.show()

#%%
plt.figure()
plt.plot(history.history['mape'])
plt.plot(history.history['val_mape'])
plt.legend(['Train MAPE', "Test MAPE"])
plt.show()

#%%
# 12. Model Deployment
y_pred = model.predict(X_test)
#%%
# Perform inverse transform
actual_newcases = mms.inverse_transform(y_test)
predicted_newcases = mms.inverse_transform(y_pred)
#%%
#plot the graph of predicted and actual number of newcases
plt.figure()
plt.title("New Cases in Malaysia")
plt.plot(actual_newcases, color='red')
plt.plot(predicted_newcases, color='blue')
plt.xlabel("Days")
plt.ylabel("Number of New Cases")
plt.legend(['Actual','Predicted'])
# %%