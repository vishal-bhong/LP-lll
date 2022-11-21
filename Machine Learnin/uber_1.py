import pandas as pd
import seaborn as sns

df=pd.read_csv('.\Datasets\uber.csv')
df.head()

df.info

df.shape

df.isnull().sum()

df.columns

df.describe()

df.corr()

sns.heatmap(df.corr(),annot=True)

df['trips'].mean()

x=df['trips']
y=df['active_vehicles']

print(x)

print(y)

x=df['trips'].values.reshape(-1,1)
print(x)

y=df['active_vehicles'].values.reshape(-1,1)
print(y)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
y_std = std.fit_transform(y)
print(y_std)

x_std=sc.fit_transform(x)
print(x_std)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(x_std,y_std,test_size=0.1,random_state=0)
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(X_train,y_train)
print("trainning score is",lr.score(X_train,y_train))
print("testing score is",lr.score(X_test,y_test))
y_predct=lr.predict(X_test)

from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
print(mean_squared_error(y_test,y_predct))
print(mean_absolute_error(y_test,y_predct))
#print(np.sqrt(mean_squared_error(y_test,y_predct))
print(r2_score(y_test,y_predct))