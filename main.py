# Step1 - Importing the libraries
import pandas as pd
import numpy as np
import pickle


# Step2 - Importing the dataset
daywise_df = pd.read_csv('daywise.csv')
print("The Dataset are : ")
print(daywise_df)

# Step3 - Removing unnecessary columns
df1=daywise_df.loc[:,['Date','Confirmed','Deaths','Recovered','Active','New cases','New deaths']]
print("The dataset After removing Unnecessary columns are :")
print(df1)

# Step4 - Analysing the Datatype of each columns 
print("Analysing the Datatype of each columns : ")
print( daywise_df.info())

# Step 5 - Typecasting  "Date column date type " from string to date-formate 


# Step 6 -  Grouping  date  and confirmed column
data=daywise_df.groupby(['Date'])[['Confirmed']].sum() # it made a cluster 
print("Grouping  date  and confirmed column :")
print(data)

# Step 7 - Making x (independent variable) and y (dependent variable), where  ( x is no of days vs  y is confirmed Cases )

x=np.arange(len(data))  #it arange based on array index assumend number of days as independent variable on x-axies
x=x.reshape(-1,1)       # reshaping x axies from 1 dimensional to 2 dimensional
print(x)

y=data.values  #assign values of confirmend cases that is dependent variable on y axies
print(y)


# Step 9 - a)  Applying polynomial regression
from sklearn.preprocessing import PolynomialFeatures #importing polynomial regression
poly = PolynomialFeatures(degree=4)  #Applying degree of 4
X= poly.fit_transform(x)
print(X)

# b) Seeing in DataFrame
print(pd.DataFrame(X)) #powers of  X till 4rth degree

# c) Applying linear regression
from sklearn.linear_model import LinearRegression #linaer regression 
regressor = LinearRegression() # storing linearregression object in regressor1
regressor.fit(X, y) # fitting X and y  Note :Not small x , its capital X of polynomial drgree values


#step 10 : dumping  model in pickle file
pickle.dump(data,open('model.pkl','wb'))

 #model=pickle.load(open('model.pkl','rb'))



# regressor.predict(model.transform([[187]])) 












