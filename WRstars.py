import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn as scikit_learn
from sklearn import preprocessing
from prettytable import PrettyTable
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv('./wrstars.csv')
df = df.drop(['ID', 'WR#','Reference','HD','Alias1', 'Alias3','Right Ascension J2000','Declination J2000','Galactic Longitude (deg)','Spectral Type', 'Spectral Type Reference', 'Binary Status', 'Binary Status Reference','Cluster','Association','Star Forming Region', 'Distance (kpc)','Distance Reference', 'Alias2','Galactic Latitude (deg)'], axis = 1)
df.dropna()
uwr = df['u (WR)'].tolist()
bwr = df['b (WR)'].tolist()
vwr = df['v (WR)'].tolist()
rwr = df['r (WR)'].tolist()
u = df['U'].tolist()
b = df['B'].tolist()
v = df['V'].tolist()
g = df['G'].tolist()
j = df['J'].tolist()
h = df['H'].tolist()
k = df['K'].tolist()
drop = []
df['type'] = 'WR'
i = 0
while i<675:
    try:
        uwr[i] = float((uwr[i]))
        bwr[i] = float((bwr[i]))
        vwr[i] = float((vwr[i]))
        rwr[i] = float((rwr[i]))
        u[i] = float((u[i]))
        b[i] = float((b[i]))
        v[i] = float((v[i]))
        g[i] = float((g[i]))
        j[i] = float((j[i]))
        h[i] = float((h[i]))
        k[i] = float((k[i]))
    except:
        drop.append(i)
    i = i+1
df.drop(drop, axis = 0, inplace = True)
df.dropna(inplace = True)
df['type'] = 1

# Initialize a list to store the MSE values
mse_list = []

# Create a list of lists of all unique predictor combinations
# For example, if you have 2 predictors,  A and B, you would 
# end up with [['A'],['B'],['A','B']]
cols = [['uwr'],['bwr'],['vwr'],['uwr','bwr'],['bwr','vwr'],['uwr','vwr'],['vwr','bwr','uwr']]

# Loop over all the predictor combinations 
for i in cols:

    # Set each of the predictors from the previous list as x
    x = df[i]
    
    # Set the "Sales" column as the reponse variable
    y = df[['type']]
   
    # Split the data into train-test sets with 80% training data and 20% testing data. 
    # Set random_state as 0
    x_train, x_test, y_train, y_test = train_test_split(x,y,train_size=0.8,random_state=0)

    # Initialize a Linear Regression model
    lreg = LinearRegression()

    # Fit the linear model on the train data
    lreg.fit(x_train,y_train)

    # Predict the response variable for the test set using the trained model
    y_pred = lreg.predict(x_test)
    
    # Compute the MSE for the test data
    MSE = mean_squared_error(y_pred,y_test)
    
    # Append the computed MSE to the initialized list
    mse_list.append(MSE)

t = PrettyTable(['Predictors', 'MSE'])

for i in range(len(mse_list)):
    t.add_row([cols[i],round(mse_list[i],3)])

print(t)