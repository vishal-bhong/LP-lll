#%%
import pandas as pd
import numpy as np
import seaborn as sns 
import matplotlib.pyplot as plt


#%%
df=pd.read_csv("uber.csv")
print(df)


#%%

print(df.info())

#%%

print(df.describe())

#%%

print(df.head())

print(df.tail())

print(df.columns)

#%%

df = df.drop(['Unnamed: 0', 'key'], axis= 1)
print(df)


#%%

print(df.head())

print(df.shape)

print(df.dtypes)

print(df.info())

print(df.describe())

print(df.isnull().sum())

#%%

df['dropoff_latitude'].fillna(value=df['dropoff_latitude'].mean(),inplace = True)

df['dropoff_longitude'].fillna(value=df['dropoff_longitude'].median(),inplace = True)

#%%
print(df.isnull().sum())
print("\n")

print(df.dtypes)  
print("\n")

#%%
#change the pickup_datetime datatype
df.pickup_datetime = pd.to_datetime(df.pickup_datetime, errors='coerce')
print(df.dtypes) 
print("\n")

#%%
df= df.assign(hour = df.pickup_datetime.dt.hour,
              day= df.pickup_datetime.dt.day,
              month = df.pickup_datetime.dt.month,
              year = df.pickup_datetime.dt.year,
              dayofweek = df.pickup_datetime.dt.dayofweek) 
print(df.head())
print("\n")
#%%


df = df.drop('pickup_datetime',axis=1)
print(df.head())
print("\n")

print(df.dtypes)
print("\n")

#%%

df.plot(kind = "box",subplots = True,layout = (7,2),figsize=(15,20))
def remove_outlier(df1 , col):
    Q1 = df1[col].quantile(0.25)
    Q3 = df1[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_whisker = Q1-1.5*IQR
    upper_whisker = Q3+1.5*IQR
    df[col] = np.clip(df1[col] , lower_whisker , upper_whisker)
    return df1  

def treat_outliers_all(df1 , col_list):
    for c in col_list:
        df1 = remove_outlier(df , c)
    return df1   

df = treat_outliers_all(df , df.iloc[: , 0::])

#%%

df.plot(kind = "box",subplots = True,layout = (7,2),figsize=(15,20))

import haversine as hs
#calculate dist bet two pt
travel_dist = []
for pos in range(len(df['pickup_longitude'])):
     long1,lati1,long2,lati2 = [df['pickup_longitude'][pos],df['pickup_latitude'][pos],df['dropoff_longitude'][pos],df['dropoff_latitude'][pos]]                               
     loc1=(lati1,long1)
     loc2=(lati2,long2)
     c = hs.haversine(loc1,loc2)
     travel_dist.append(c)
    
print(travel_dist)
df['dist_travel_km'] = travel_dist
df.head() 
#%%

