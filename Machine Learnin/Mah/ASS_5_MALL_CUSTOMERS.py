import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df=pd.read_csv("Datasets/Mall_Customers.csv")

df.head()

df.columns

df.shape

df.describe()

df.info()

df.isnull()

df.isnull().sum()

import seaborn as sns
sns.heatmap(df.corr(),annot=True)

X=df[['Annual Income (k$)','Spending Score (1-100)']]
X.head()

from sklearn.cluster import KMeans

wcss=[]
for i in range(1,11):
  km=KMeans(n_clusters=i)
  km.fit(X)
  wcss.append(km.inertia_)

plt.figure(figsize=(12,6))
plt.plot(range(1,11),wcss)
plt.plot(range(1,11),wcss,linewidth=2,color='red',marker=8)
plt.xlabel("k-value")
plt.ylabel("wcss")
plt.xticks(np.arange(1,11,1))
plt.show()

km1=KMeans(n_clusters=5)
km1.fit(X)

y=km1.predict(X)
y

df['label']=y

df.head()

"""**Now printing the customer ID according to the groups**"""

cust1=df[df["label"]==1]
print('Number of customer in 1st group=', len(cust1))
print('They are -', cust1["CustomerID"].values)

cust2=df[df['label']==2]
print('number of customer in 2nd group=',len(cust2))
print("they are ",cust2['CustomerID'].values)

cust3=df[df['label']==3]
print("no of customer in 3rd group",len(cust3))
print("they are",cust3['CustomerID'].values)

cust4=df[df['label']==4]
print("no of customer in 4th group",len(cust4))
print("they are",cust4['CustomerID'].values)

cust5=df[df['label']==0]
print("no of customer in 5th group",len(cust5))
print("They are ",cust5['CustomerID'].values)

