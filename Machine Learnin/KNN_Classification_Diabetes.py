#%%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split 
#support vector classifier (SVC)
from sklearn.svm import SVC
from sklearn import metrics

import warnings
warnings.filterwarnings('ignore')

#%%
df= pd.read_csv("../../Machine_Learning_Project_codes/diabetes.csv")
print(df)

#%%
print(df.columns)

#%%

print(df.info())

#%%
print(df.describe())

#%%
print(df.isnull().sum())

#%%

X = df.drop('Outcome',axis = 1)
y = df['Outcome']

from sklearn.preprocessing import scale
X = scale(X)
print(X.shape)
#%%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state =0 )

#%%
print(X_train.shape)
#%%
print(X_test.shape)

#%%
print(y_train.shape)

#%%
print(y_test.shape)

#%%
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print(y_pred)
#%%
print("Confusion matrix: ")
cs = metrics.confusion_matrix(y_test,y_pred)
print(cs)


#%%
print("Accuracy ",metrics.accuracy_score(y_test,y_pred))
#%%
total_misclassified = cs[0,1] + cs[1,0]
print(total_misclassified)
total_examples = cs[0,0]+cs[0,1]+cs[1,0]+cs[1,1]
print(total_examples)
print("Error rate",total_misclassified/total_examples)
print("Error rate ",1-metrics.accuracy_score(y_test,y_pred))
#%%

tp_fp = cs[0][0]+cs[0][1]
tp = cs[0][0]

precision = tp/(tp_fp)
print("Precision score",precision)
#%%

tp_fn = cs[0][0]+cs[1][0]
tp = cs[0][0]

recall = tp/(tp_fn)
print("Recall Score",recall)




