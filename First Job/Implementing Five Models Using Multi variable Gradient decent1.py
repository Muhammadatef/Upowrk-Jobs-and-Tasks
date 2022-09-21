#!/usr/bin/env python
# coding: utf-8

# In[231]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error


# In[204]:


df = pd.read_csv('A:/Upwork/First Job/test1.csv')
df


# In[205]:


df.isnull().sum()


# In[206]:


df.drop('time', axis=1, inplace=True)
df


# In[207]:


bias = np.repeat(1,200)
df = {'x_1':df['x1'],'x_1^2':np.power(df['x1'],2),'x_1^3':np.power(df['x1'],3),'x_1^4':np.power(df['x1'],4),'x_1^5':np.power(df['x1'],5),'x_2':df['x2'],'bias':bias,'y':df['y']}
df = pd.DataFrame(data=df,columns=['x_1','x_1^2','x_1^3','x_1^4','x_1^5','x_2','bias','y'])
df


# In[208]:


df.isnull().sum()


# In[209]:


# Data for the first Model
x = df.loc[:,['x_1^3','x_1^5','x_2','bias']]
y = df.loc[:,'y']


# In[210]:


x


# In[211]:


## nom_variables = all the features including the bias in the model 
## example model1 = x_1^3 + x_1^5 _ x_2 + bias, 
## that means we have num_variables = 4

## Here i made a function to multi variable gradient decent to find 
## the best parameters to converge the data and to generalize on all 5 models
def multi_variable(x_point,y_points,learn_rate,nom_iteration,nom_variables):
    m = len(x_point)
    theta = np.repeat(0,nom_variables).reshape(nom_variables,1)
    cost_fun_all = []
    theta_all = []
    y_points = np.array(y_points).reshape(-1,1)
    
    for i in range(nom_iteration):
        theta_all.append(theta)
        
        y_pred = np.dot(x_point,theta) 
        #print(y_pred)
        
        error = y_pred - y_points
        
        cost_fun = (1/(2*m)) * np.sum(np.square(error))
        cost_fun_all.append(cost_fun)
        
        grad = (1/m) *np.sum(np.dot(error.T,x_point))
        
        theta = theta - learn_rate*grad
        #print(theta)
        if i > 1 :
            if np.abs((cost_fun_all[i] - cost_fun_all[i-1])) < 0.0001:
                break
                
            if np.linalg.norm(theta_all[i]-theta_all[i-1]) < 0.0001:
                break 
            if np.linalg.norm(grad) < 0.0001:
                break
        
    return y_pred,theta_all,cost_fun_all,i+1


# In[212]:


y_pred1,theta_all,cost,nom_iter = multi_variable(x,y,0.0001,10000,4)


# In[213]:


plt.figure(figsize=(8,6))
plt.plot(np.arange(nom_iter),cost);
plt.xlabel('Iteration',fontsize=10,color='b')
plt.ylabel('Cost Function',fontsize=10,color='b')
plt.title('Cost vs Iteration',fontsize=12,color='b');


# In[242]:


model_error1= mean_squared_error(y,y_pred1)
model_error1


# In[214]:


print(f"data converged after {nom_iter} iteration")


# ## For the second Model

# In[215]:


# Data for the second Model
x = df.loc[:,['x_1','x_2','bias']]
y = df.loc[:,'y']


# In[216]:


y_pred2,theta_all,cost,nom_iter = multi_variable(x,y,0.001,10000,3)


# In[217]:


plt.figure(figsize=(8,6))
plt.plot(np.arange(nom_iter),cost);
plt.xlabel('Iteration',fontsize=10,color='b')
plt.ylabel('Cost Function',fontsize=10,color='b')
plt.title('Cost vs Iteration',fontsize=12,color='b');


# In[244]:


model_error2 = mean_squared_error(y,y_pred2)
model_error2


# In[218]:


print(f"data converged after {nom_iter} iteration")


# ## For the Third Model

# In[219]:


# Data for the third model
x = df.loc[:,['x_1','x_1^2','x_1^4','x_2','bias']]
y = df.loc[:,'y']


# In[220]:


y_pred3,theta_all,cost,nom_iter = multi_variable(x,y,0.001,10000,5)


# In[221]:


plt.figure(figsize=(8,6))
plt.plot(np.arange(nom_iter),cost);
plt.xlabel('Iteration',fontsize=10,color='b')
plt.ylabel('Cost Function',fontsize=10,color='b')
plt.title('Cost vs Iteration',fontsize=12,color='b');


# In[245]:


model_error3= mean_squared_error(y,y_pred3)
model_error3


# In[246]:


print(f"data converged after {nom_iter} iteration")


# ## For the fourth Model

# In[223]:


#Data for the fourth model
x = df.loc[:,['x_1','x_1^2','x_1^3','x_1^5','x_2','bias']]
y = df.loc[:,'y']


# In[224]:


y_pred4,theta_all,cost,nom_iter = multi_variable(x,y,0.0001,100000,6)


# In[225]:


plt.figure(figsize=(8,6))
plt.plot(np.arange(nom_iter),cost);
plt.xlabel('Iteration',fontsize=10,color='b')
plt.ylabel('Cost Function',fontsize=10,color='b')
plt.title('Cost vs Iteration',fontsize=12,color='b');


# In[247]:


model_error4= mean_squared_error(y,y_pred4)
model_error4


# In[226]:


print(f"data converged (reached to the best parameters) after {nom_iter} iteration")


# ## For the fivth model 

# In[227]:


# Data for the fivth model 
x = df.loc[:,['x_1','x_1^3','x_1^4','x_2','bias']]
y = df.loc[:,'y']


# In[228]:


y_pred5,theta_all,cost,nom_iter = multi_variable(x,y,0.001,100000,5)


# In[229]:


plt.figure(figsize=(8,6))
plt.plot(np.arange(nom_iter),cost);
plt.xlabel('Iteration',fontsize=10,color='b')
plt.ylabel('Cost Function',fontsize=10,color='b')
plt.title('Cost vs Iteration',fontsize=12,color='b');


# In[241]:


model_error5= mean_squared_error(y,y_pred5)
model_error5


# In[238]:


#for the fivth model
#data will find the best parameters (converge) after (nom_iter) iteration
print(f"data converged after {nom_iter} iteration")


# In[249]:


print(f"The best model is the model which have the least mean squared error which is Model 2 which is have {model_error2} mse")

