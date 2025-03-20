import numpy as np
import pandas as pd

df=pd.read_csv(r"C:\Users\Smile\Downloads\winequality-red.csv")
df.head()

#fixed acidity & citric acid ,free sulfur dioxide& total sulfur dioxide are same corrlation each other .So we can drop 2 columns fixed acidity,free sulfur dioxide.These are low correlation to Quality
df=df.drop(['fixed acidity','free sulfur dioxide'],axis=1)

x=df.drop('quality',axis=1)
y=df['quality']

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=42)
from sklearn import linear_model
lr=linear_model.LinearRegression()
lr_model=lr.fit(x_train,y_train)
pred=lr_model.predict(x_test)

import pickle
# Save train model
with  open('lr_model.pickle','wb')as f:
    pickle.dump(lr_model,f)