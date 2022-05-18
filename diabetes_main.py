# -*- coding: utf-8 -*-
"""
Created on Tue May 17 20:45:52 2022

@author: End User
"""

import os
import pandas as pd
from sklearn.impute import SimpleImputer
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,OneHotEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout,Dense,BatchNormalization
from sklearn.metrics import classification_report, confusion_matrix
import pickle

DATA_PATH=os.path.join(os.getcwd(),'diabetes.csv')

#step 1)load data
df=pd.read_csv(DATA_PATH)

#step 2)intepret data

df.info()
df.describe().T

#step3)data cleaning
ii_imputer=SimpleImputer()
df=ii_imputer.fit_transform(df)

pd.DataFrame(df).describe().T


bool_series=pd.DataFrame(df).duplicated()
sum(bool_series==True)

df=pd.DataFrame(df).drop_duplicates()


X=df.iloc[:,0:8]
y=df.iloc[:,8]

scaler=MinMaxScaler()
X_scaled=scaler.fit_transform(X)
pickle.dump(scaler,open('mms_scaler.pkl','wb'))

ohe=OneHotEncoder()
y_one_hot=ohe.fit_transform(np.expand_dims(y,axis=-1))
pickle.dump(ohe,open('ohe_scaler.pkl','wb'))


X_train,X_test,y_train,y_test=train_test_split(X_scaled,y_one_hot,test_size=0.3)


#%% Model Building

model=Sequential()
model.add(Dense(128,activation=('relu'),input_shape=(8,)))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(128,activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(2,activation='softmax'))
          
model.summary()

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics='acc')

#%%model saving

MODEL_SAVE_PATH=os.path.join(os.getcwd(),'model.h5')
model.save(MODEL_SAVE_PATH)


#%% Evaluation
y_pred = model.predict(X_test)
y_true=np.argmax(y_test,axis=1)
y_pred=np.argmax(y_pred, axis=1)
cm=confusion_matrix(y_true, y_pred)

print(classification_report(y_true, y_pred))
