#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing essential libraries
import numpy as np
import pandas as pd
import matplotlib .pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings ("ignore")


# In[2]:


#importing dataset using pandas

concrete_df = pd.read_csv("Concrete Compressive Strength.csv")


# In[3]:


concrete_df


# In[4]:


# Separating dependent and independent variable.
X_raw = concrete_df.drop(columns=['strength '], axis=1)
Y = concrete_df['strength ']


# In[5]:


# Preparing the data to fit Linear regression
def Prepare_data(X):
    X.columns = ['X_1', 'X_2', 'X_3', 'X_4', 'X_5', 'X_6', 'X_7', 'X_8']
    X_0 = pd.DataFrame({'X_0' : [1]*1030})
    df = pd.concat([X_0,X],axis = 1, join = 'inner')
    arr=np.array(df)
    return df,arr


# In[6]:


df, X = Prepare_data(X_raw)
print("____________________________________The prepared data is_______________________________________________________")
df


# In[7]:


# function to estimate parameters
def Parameter_est( X ,y):
    Transpose = X.T
    mal = np.matmul(X.T,X)
    inv = np.linalg.inv(mal)
    b_hat = np.matmul(np.matmul(inv,X.T),y)
    return b_hat

# Assign the estimated parameters to b_hat
b_hat = Parameter_est( X ,Y)
print("_____________________________________The weight parameters are__________________________________________________________")
b_hat


# In[8]:


# Function the Linear Regression
def linearRegression(beta):
    print ( "___________________________________The fitted equation is____________________________________________________________")
    print (f"y_hat={b_hat[0]:.3f}X_0+{b_hat[1]:.3f}X_1+{b_hat[2]:.3f}X_2+{b_hat[3]:.3f}X_3+{b_hat[4]:.3f}X_4+{b_hat[5]:.3f}X_5+{b_hat[6]:.3f}X_6+{b_hat[7]:.3f}X_7+{b_hat[8]:.3f}X_8}}")
    print ( "______________________________________________________________________________________________________________________")

# Fitting the equation:
linearRegression(b_hat)


# predicting the dependent variable using the fitted equation
y_hat = X@b_hat
print("________________________________________The Predicted Values are__________________________________________________________")
print(y_hat)


# In[9]:


#Error calculation:
def Errors(y_true, y_pred):
    e = y_true - y_pred
    mse = ((e**2).sum())/X.shape[0]
    return mse
mse = Errors(Y,y_hat)

print("Mean squared error is: ", mse)
rmse = np.sqrt(mse)
print("Root mean squared error is: {}".format(rmse))


# In[10]:


# Function to create ANOVA Table
def Anova(y_true,y_pred):
    # Total variation
    SSt = ((y_true-y_true.mean())**2).sum()
    degree_t = X.shape[0] - 1
    # Residual variation
    SSres= ((y_true - y_pred)**2).sum()
    degree_res = X.shape[0] - X.shape[1]
    MSres = SSres/degree_res
    # variation due to regression
    SSreg = SSt-SSres
    degree_reg = X.shape[1]-1
    MSreg= SSreg/degree_reg
    F = MSreg/MSres
    return degree_res,degree_reg, SSres,SSreg,MSres,MSreg, F

degree_res, degree_reg, SSres,SSreg,MSres,MSreg, F = Anova(Y,y_hat)


# In[11]:


# Creating ANOVA Table
anova_dict = {'DF':[degree_reg, degree_res, degree_reg+degree_res], 'SS':[SSreg, SSres, SSreg+SSres], 'MS':[MSreg, MSres, MSreg+MSres], 'F':[F]}
Anova_df = pd.DataFrame(anova_dict,index=["Regression","Residual","Total"])
print("__________________________________________________ANOVA Table____________________________________________________________")
Anova_df


# In[12]:


# Testing Null hypothesis
print("________________________________________________________________________________________________________________________")
print(f"H0: b0=b1=............bk-1=0 against \nH1: bi!=0 for i=0 to k-1")
print("If F>F{alpha, k-1, n-k}, The H0 is rejested")
print("_________________________________________________________________________________________________________________________")
import scipy.stats
from scipy.stats import f
alpha = 0.05
q = 1 - alpha
f = f.ppf(q, degree_reg, degree_res)
print(f"The calcylated f value is: {f}" )
print(f"The observed f value is {F}")
if(abs(F)>f):
        print("The null hypothesis H0 is rejected")
else:
        print("The null hypothesis H0 is accepted")


# In[13]:


# Predict R^2 value and adjusted R^2 value:
R_sq = (SSreg/ (SSres+SSreg) ) * 100
AdjR_sq = (1- MSres/(MSres+MSreg)) * 100
print(f"The R square value is:{R_sq}")
print(f"Adjusted R square value is: {AdjR_sq}")


# In[14]:


# Test on individual regression coefficient (Partial test or Marginat Test)
Corr = np.linalg.inv(X.T@X)
from scipy.stats import t
def Marginal_test(beta, C, MSres,X):
    n = X.shape[1]
    for i in range(n):
        T = beta[i]/np.sqrt(MSres*C[i][i])
        if(abs(T)>t.ppf(1-0.05, degree_res)):
            print(f"H0:b{i}=0 is rejected")
        else:
            print(f"HØ:b{i}=0 is accepted")

Marginal_test(b_hat,Corr,MSres,X)


# In[15]:


# Plot the regression Line fitted by the function made
plt.figure(figsize=[19 , 9])
plt.scatter( y_hat, Y)
plt.plot([Y.min(), Y.max()], [Y.min(), Y.max()], color = 'green')
plt.xlabel('predicted')
plt.ylabel('orignal')
plt.show()


# In[16]:


# trying the inbuilt regression functionn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
lr = LinearRegression()
fit = lr.fit(X,Y)
print( '.......................................................................')
y_predict = lr.predict(X)
print( 'mean_ squred_error is ==',mean_squared_error(Y, y_predict))
rms = np.sqrt(mean_squared_error(Y,y_predict))
print( 'root mean squared error is == {}'.format(rms))
print("________________________________________________The predicted values are_______________________________________________")
y_predict


# In[17]:


# Plot the regression Line fitted by the inbuilt libraries in python
plt.figure(figsize=[19 , 9])
plt.scatter( y_predict, Y)
plt.plot([Y.min(), Y.max()], [Y.min(), Y.max()], color = 'green')
plt.xlabel('predicted')
plt.ylabel('orignal')
plt.show()


# In[18]:


# Plot heatmap to check multicollinearity
plt.figure(figsize = (14,7))
sns.heatmap(concrete_df.corr( ), annot=True, cmap='viridis')


# In[19]:


# check for any duplicate values in the data
duplicates = concrete_df.duplicated()
concrete_df[duplicates]
duplicates.value_counts( )


# In[20]:


# Dropping duplicate values.
concrete_df=concrete_df.drop_duplicates()
concrete_df


# In[21]:


fig, axes = plt.subplots(nrows=len(concrete_df.iloc[:,:-1].columns)//2, ncols=2, figsize=(20, 32))
axes = axes.flatten()

for i, column in enumerate(concrete_df.iloc[:,:-1].columns):
    sns.boxplot(concrete_df[column], ax=axes[i])
    axes[i].set_title(column, fontsize = 22)
    axes[i].tick_params(axis='x', labelsize=18)
    axes[i].set_xlabel(column, fontsize=22)

plt.tight_layout()
plt.show()


# In[ ]:





# In[22]:


def remove_outlier(col):
    col_sorted = sorted(col, reverse=True)
    Q1, Q3 = pd.Series(col_sorted).quantile([0.25, 0.75])
    IQR = Q3 - Q1
    lower_range = Q1 - (1.5 * IQR)
    upper_range = Q3 + (1.5 * IQR)
    return lower_range, upper_range

for i in concrete_df.columns:
    l_r, u_r = remove_outlier(concrete_df[i])
    concrete_df[i].loc[~concrete_df[i].between(l_r, u_r)] = pd.NA

concrete_df


# In[23]:


rows_with_nan = concrete_df[concrete_df.isna().any(axis=1)]

rows_with_nan


# In[24]:


concrete_df.dropna(inplace=True)
concrete_df


# In[25]:


# Separating dependent and independent variable.
X_raw = concrete_df.drop(columns=['strength '], axis=1)
Y = concrete_df['strength ']


# In[26]:


X_raw


# In[27]:


arr = np.array(X_raw)


# In[28]:


print("____________________________________The prepared data is_______________________________________________________")
X_raw


# In[29]:


X_raw = X_raw.astype(float)
X = X_raw
X.info()


# In[30]:


# function to estimate parameters
def Parameter_est( X ,y):
    Transpose = X.T
    mal = np.matmul(X.T,X)
    inv = np.linalg.inv(mal)
    b_hat = np.matmul(np.matmul(inv,X.T),y)
    return b_hat

# Assign the estimated parameters to b_hat
b_hat = Parameter_est( X ,Y)
print("_____________________________________The weight parameters are__________________________________________________________")
b_hat


# In[31]:


# Function the Linear Regression 
def linearRegression(beta):
    print ( "___________________________________The fitted equation is____________________________________________________________")
    print (f"y_hat={b_hat[0]:.3f}X_1+{b_hat[1]:.3f}X_2+{b_hat[2]:.3f}X_3+{b_hat[3]:.3f}X_4+{b_hat[4]:.3f}X_5+{b_hat[5]:.3f}X_6+{b_hat[6]:.3f}X_7+{b_hat[7]:.3f}X_8}}")
    print ( "______________________________________________________________________________________________________________________")

# Fitting the equation:
linearRegression(b_hat)


# predicting the dependent variable using the fitted equation
b_hat = b_hat.values
y_hat = X_raw@b_hat
print("________________________________________The Predicted Values are__________________________________________________________")
print(y_hat)


# In[32]:


#Error calculation: 
def Errors(y_true, y_pred):
    e = y_true - y_pred
    mse = ((e**2).sum())/X.shape[0]
    return mse
mse = Errors(Y,y_hat)

print("Mean squared error is: ", mse)
rmse = np.sqrt(mse)
print("Root mean squared error is: {}".format(rmse))


# In[33]:


from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
poly = PolynomialFeatures ( degree=3, interaction_only=False , include_bias=True, order= 'C' )
x = poly.fit_transform(X)
poly_clf = linear_model.LinearRegression()
poly_clf.fit (x, Y)
print(poly_clf.score(x,Y))


# In[34]:


print( '.........................................................................')
y_predict = poly_clf.predict(x)
print( 'mean_squared_error is ==' , mean_squared_error(Y,y_predict) )
rms = np.sqrt(mean_squared_error(Y,y_predict))
print( 'root mean squared error is == {} '.format(rms))


# In[35]:


# Function to create ANOVA Table 
def Anova(y_true,y_pred):
    # Total variation
    SSt = ((y_true-y_true.mean())**2).sum()
    degree_t = X.shape[0] - 1
    # Residual variation
    SSres= ((y_true - y_pred)**2).sum()
    degree_res = X.shape[0] - X.shape[1]
    MSres = SSres/degree_res
    # variation due to regression
    SSreg = SSt-SSres
    degree_reg = X.shape[1]-1
    MSreg= SSreg/degree_reg
    F = MSreg/MSres
    return degree_res,degree_reg, SSres,SSreg,MSres,MSreg, F

degree_res, degree_reg, SSres,SSreg,MSres,MSreg, F = Anova(Y,y_hat)


# In[36]:


# Creating ANOVA Table 
anova_dict = {'DF':[degree_reg, degree_res, degree_reg+degree_res], 'SS':[SSreg, SSres, SSreg+SSres], 'MS':[MSreg, MSres, MSreg+MSres], 'F':[F]}
Anova_df = pd.DataFrame(anova_dict,index=["Regression","Residual","Total"])
print("__________________________________________________ANOVA Table____________________________________________________________")
Anova_df


# In[37]:


# Predict R^2 value and adjusted R^2 value:  
R_sq = (SSreg /(SSreg + SSres)) * 100
AdjR_sq = (1-MSres/(MSres+MSreg)) * 100
print(f"The R square value is: {R_sq}")
print(f"Adjusted R square value is: {AdjR_sq}")


# In[38]:


# Plot the regression Line fitted by the function made  
plt.figure(figsize=[19 , 9])
plt.scatter( y_hat, Y, color='red')
plt.plot([Y.min(), Y.max()], [Y.min(), Y.max()], color = 'green')
plt.xlabel('predicted')
plt.ylabel('orignal')
plt.show()


# In[39]:


poly_clf.coef_ 


# In[ ]:





# In[ ]:




