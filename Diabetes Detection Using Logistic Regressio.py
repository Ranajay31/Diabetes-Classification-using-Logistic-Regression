#!/usr/bin/env python
# coding: utf-8

# # Diabetes Detection Using Logistic Regression

# In[1]:


import pandas as pd 
import numpy as np 
from sklearn.preprocessing import StandardScaler 
from sklearn.linear_model  import Ridge,Lasso,RidgeCV, LassoCV, ElasticNet, ElasticNetCV, LogisticRegression
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor 
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import scikitplot as skl
sns.set()


# In[2]:


data=pd.read_csv('E:\\Machine Learning\\Logistic-regression_final\\diabetes.csv')
data.head()


# In[3]:


data.describe()


# There are no missing values

# In[4]:


df=data.copy()
df.head()


# In[5]:


#Now let us look at how the data is distributed for every column

plt.figure(figsize=(20,25),facecolor='white')
plotnumber=1

for column in df:
    if plotnumber<=9:
        ax=plt.subplot(3,3,plotnumber)
        sns.histplot(df[column])
        plt.xlabel(column,fontsize=15)
    plotnumber+=1
plt.show()    


# We can see there is some skewness in the data, let's deal with data.
# 
# Also, we can see there few data for columns Glucose, Insulin, skin thickness, BMI and Blood Pressure which have value as 0. That's not possible. You can do a quick search to see that one cannot have 0 values for these. Let's deal with that. we can either remove such data or simply replace it with their respective mean values. Let's do the latter.

# In[8]:


df['Glucose']=df['Glucose'].replace(0,df['Glucose'].mean())
df['Insulin']=df['Insulin'].replace(0,df['Insulin'].mean())
df['SkinThickness']=df['SkinThickness'].replace(0,df['SkinThickness'].mean())
df['BMI']=df['BMI'].replace(0,df['BMI'].mean())
df['BloodPressure']=df['BloodPressure'].replace(0,df['BloodPressure'].mean())


# In[11]:


##Now let us again see how the distribution for different column looks

plt.figure(figsize=(15,20),facecolor='white')
plotnumber=1

for column in df:
    if plotnumber<=9:
        ax=plt.subplot(3,3,plotnumber)
        sns.histplot(df[column])
        plt.xlabel(column,fontsize=15)
    plotnumber+=1
plt.show()
    


# We notice that there are outliers.It is time to remove them

# In[15]:


fig, ax = plt.subplots(figsize=(15,10))
sns.boxplot(data=df, width= 0.5,ax=ax,  fliersize=3)


# In[16]:


q = df['Pregnancies'].quantile(0.98)
# we are removing the top 2% data from the Pregnancies column
data_cleaned = df[df['Pregnancies']<q]
q = data_cleaned['BMI'].quantile(0.99)
# we are removing the top 1% data from the BMI column
data_cleaned  = data_cleaned[data_cleaned['BMI']<q]
q = data_cleaned['SkinThickness'].quantile(0.99)
# we are removing the top 1% data from the SkinThickness column
data_cleaned  = data_cleaned[data_cleaned['SkinThickness']<q]
q = data_cleaned['Insulin'].quantile(0.95)
# we are removing the top 5% data from the Insulin column
data_cleaned  = data_cleaned[data_cleaned['Insulin']<q]
q = data_cleaned['DiabetesPedigreeFunction'].quantile(0.99)
# we are removing the top 1% data from the DiabetesPedigreeFunction column
data_cleaned  = data_cleaned[data_cleaned['DiabetesPedigreeFunction']<q]
q = data_cleaned['Age'].quantile(0.99)
# we are removing the top 1% data from the Age column
data_cleaned  = data_cleaned[data_cleaned['Age']<q]


# In[21]:


# let's see how data is distributed for every column
plt.figure(figsize=(20,25), facecolor='white')
plotnumber = 1

for column in data_cleaned:
    if plotnumber<=9 :
        ax = plt.subplot(3,3,plotnumber)
        sns.distplot(data_cleaned[column])
        plt.xlabel(column,fontsize=20)
        #plt.ylabel('Salary',fontsize=20)
    plotnumber+=1
plt.show()


# The data looks much better now than before. We will start our analysis with this data now as we don't want to lose important information. If our model doesn't work with accuracy, we will come back for more preprocessing.

# In[22]:


df=data_cleaned


# In[24]:


X= df.drop(columns=["Outcome"])
y=df["Outcome"]


# Now we should scale our data. Let's use the standard scaler for that.

# In[27]:


scalar=StandardScaler()
X_scaled=scalar.fit_transform(X)


# In[28]:


X_scaled


# Let us check for multicollinearity now

# In[33]:


vif = pd.DataFrame()
vif["vif"] = [variance_inflation_factor(X_scaled,i) for i in range(X_scaled.shape[1])]
vif["Features"] = X.columns

#let's check the values
vif


# All the VIF values are less than 5 and are very low. That means no multicollinearity. Now, we can go ahead with fitting our data to the model. Before that, let's split our data in test and training set.

# In[34]:


x_train,x_test,y_train,y_test= train_test_split(X_scaled,y,test_size= 0.25, random_state = 355)


# In[35]:


log_reg= LogisticRegression()
log_reg.fit(x_train,y_train)


# ## Model Evaluation
# ### Let's see how well our model performs on the test data set.

# In[37]:


y_pred=log_reg.predict(x_test)


# In[39]:


accuracy=accuracy_score(y_test,y_pred)
accuracy


# In[42]:


##Let us find the Confusion Matrix

conf_mat = confusion_matrix(y_test,y_pred)
conf_mat


# In[43]:


true_positive=conf_mat[0][0]
false_positive=conf_mat[0][1]
false_negative=conf_mat[1][0]
true_negative=conf_mat[1][1]


# In[44]:


# Breaking down the formula for Accuracy
Accuracy = (true_positive + true_negative) / (true_positive +false_positive + false_negative + true_negative)
Accuracy


# In[45]:


# Precison
Precision = true_positive/(true_positive+false_positive)
Precision


# In[46]:


# Recall
Recall = true_positive/(true_positive+false_negative)
Recall


# In[47]:


#F1 Score

F1_score=2*Precision*Recall/(Precision+Recall)
F1_score


# ##### AUC

# In[48]:


#Area Under Curve (AUC)

auc= roc_auc_score(y_test,y_pred)
auc


# ##### ROC

# In[49]:


fpr,tpr, thresholds= roc_curve(y_test,y_pred)


# In[50]:


plt.plot(fpr, tpr, color='orange', label='ROC')
plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--',label='ROC curve (area = %0.2f)' % auc)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
plt.show()


# In[ ]:




