
# Importing Libraries
import pandas as pd            #Version: 1.5.3
import numpy as np             #Version: 1.24.2
import matplotlib.pyplot as plt #Version: 3.7.1
import seaborn as sns               # Version: 0.12.2
from sklearn.model_selection import train_test_split #Version: 1.2.2
from sklearn.linear_model import LinearRegression #Version: 1.2.2
from sklearn.tree import DecisionTreeRegressor #Version: 1.2.2
from sklearn.ensemble import RandomForestRegressor #Version: 1.2.2
from sklearn.svm import SVR #Version: 1.2.2
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score #Version: 1.2.2

# Loading Dataset
data = pd.read_csv(r'C:\Users\Swapnil Jyot\Downloads\50_Startups.csv')
data
df_sorted = data.sort_values('R&D Spend')
df_sorted

len(data.index)

# # ANALYSING THE DATA....

plt.subplot(3, 2, 1)
data["Profit"].plot.hist()
plt.title('Histogram of Profit')

# #Hence we have to get maximum profit of near about 100000 for 10 times acoording to the table
# 
# lets see the how all three factors(R&D Spend,Administration,Marketing Spend) affect the profit of the companies ...

# ### 1.R&D Spend vs Profit

df_sorted = data.sort_values('R&D Spend')
X = df_sorted["R&D Spend"].values
Y= df_sorted["Profit"].values
plt.subplot(3, 2, 4)
plt.plot(X,Y,color='#58b970', label='Regression Line')
plt.title('Line Graph of R&D Spend vs Profit')


# ### 2.Administration vs Profit

df_sorted = data.sort_values('Administration')
X = df_sorted["Administration"].values
Y= df_sorted["Profit"].values
plt.subplot(3, 2, 5)
plt.plot(X,Y,color='#58b970', label='Regression Line')
plt.title('Line Graph of Administration vs Profit')

# ### 3.Marketing Spend vs Profit

df_sorted = data.sort_values('Marketing Spend')
X = df_sorted["Marketing Spend"].values
Y= df_sorted["Profit"].values
plt.subplot(3, 2, 6)
plt.plot(X,Y,color='#58b970', label='Regression Line')
plt.title('Line Graph of Marketing Spend vs Profit')

# ### CONCLUSION

# #As profit change partial linearly with the R&D spend
# #As profit depend partially with the Marketing Spend
# #As profit is almost independent of the parameter Administration

# # DATA WRANGLING

# #finding the null values in the data before training and testing the data
data.isnull()

plt.subplot(3, 2, 2)
sns.heatmap(data.isnull())
plt.title('Heatmap of Correlation Matrix')

# ### Almost zero null values in the data now data is ready for testing and training

# # Training And Testing the data

x=data.drop("Profit",axis=1)
y=data['Profit']

X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.4,random_state=1)

print(X_train)

print(y_train)


# # BUILDING THE REGRESSION MODELS

# Building Linear Regression Model
lr = LinearRegression()
lr.fit(X_train, y_train)

# Building Decision Tree Regression Model
dt = DecisionTreeRegressor(random_state=42)
dt.fit(X_train, y_train)

# Building Random Forest Regression Model
rf = RandomForestRegressor(random_state=42)
rf.fit(X_train, y_train)

# Building  support vector Regression Model
svr = SVR()
svr.fit(X_train, y_train)

# Evaluating Linear Regression Model
y_pred_lr = lr.predict(X_test)
mae_lr = mean_absolute_error(y_test, y_pred_lr)
mse_lr = mean_squared_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)*100

# Evaluating Decision Tree Regression Model
y_pred_dt = dt.predict(X_test)
mae_dt = mean_absolute_error(y_test, y_pred_dt)
mse_dt = mean_squared_error(y_test, y_pred_dt)
r2_dt = r2_score(y_test, y_pred_dt)*100

# Evaluating Random Forest Regression Model
y_pred_rf = rf.predict(X_test)
mae_rf = mean_absolute_error(y_test, y_pred_rf)
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_dt)*100
#Evaluating support vector regression model(SVR)
y_pred_svr = svr.predict(X_test)
mae_svr = mean_absolute_error(y_test, y_pred_dt)
mse_svr = mean_squared_error(y_test, y_pred_dt)
r2_svr = r2_score(y_test, y_pred_dt)*100

print(mae_lr,mse_lr,r2_lr)
print(mae_dt,mse_dt,r2_dt )
print(mae_rf,mse_rf,r2_rf)
print(mae_svr,mse_svr,r2_svr)

reg_metrices=pd.DataFrame({'LRM':[mae_lr,mse_lr,r2_lr],'DTR':[mae_dt,mse_dt,r2_dt],
                    'RFR':[mae_rf,mse_rf,r2_rf],'SVR':[mae_svr,mse_svr,r2_svr]})

reg_metrices

r2_scores = pd.DataFrame({'Model': ['LRM', 'DTM', 'RFM','SVM'], 
                          'R-squared Score': [r2_lr,r2_dt,r2_rf,r2_svr]})
r2_scores['R-squared Score'] = [round(score, 2) for score in r2_scores['R-squared Score']]

plt.subplot(3, 2, 3)
colors = ['red', 'green', 'blue','orange']
plt.bar(r2_scores['Model'], r2_scores['R-squared Score'],color=colors)
for i in range(len(r2_scores['Model'])):
    plt.text(i, r2_scores['R-squared Score'][i], r2_scores['R-squared Score'][i], ha='center', va='bottom')
plt.title('Comparison of R-squared Scores for Different Regression Models')
plt.xlabel('Regression Model')
plt.ylabel('R-squared Score')
plt.ylim([0.0, 100.0])
plt.tight_layout()
plt.show()

# ### hence you can clearly see that the heighest accuracy levels of different regression models
