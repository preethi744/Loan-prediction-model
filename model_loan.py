#!/usr/bin/env python
# coding: utf-8

# In[1]:

#Introcution
#This dataset is for loan prediction of an applicant.It has features like Loan_ID(ID of the applicant's loan),
#Gender(gender of applicant),
#Dependents(dependents of applicant),
#Education,(applicant is a graduate or not)
#Self_employment(it says self employment like business or not),
#Married(Maritial Status),
#ApplicantIncome(Income of the applicant),
#CoapplicantIncome(Income of co-applicant),
#LoanAmount(Loan amount requested),
#Loan-amount-term(term of loan amount),
#credit-history(history of applicant credit)
#Loan_Status(Status of loan that says loan approved or not)


#importing required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import warnings#to ignore warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)


# In[2]:


data=pd.read_csv('traindataset.csv')


# In[3]:


data.head()


# In[4]:


data.info()


# In[5]:


data.shape

Data Visualization
# In[7]:


sns.countplot('Loan_Status', data=data)
plt.show()


# From above plot,we can say that the loan status count is high for yes.

# In[8]:


sns.countplot(x ="Loan_Status", hue ='Gender', data = data)
plt.show()


# From above plot,we can say that if applicant is male then he has more chance of getting loan.

# In[9]:


sns.countplot(x ="Loan_Status", hue ='Married', data = data)
plt.show()


# From above plot,we can say that married has more chance of getting loan.

# In[10]:


sns.countplot(x ="Loan_Status", hue ='Property_Area', data = data)
plt.show()


# From above plot,we can say that semi urban applicants has more chance of getting loan followed by urban and then rural.

# In[11]:


sns.countplot(x ="Education", hue ='Loan_Status', data = data)
plt.show()


# From above plot,we can say that graduated applicants has more chance of getting loan.

# In[12]:


grid = sns.FacetGrid(data, col = 'Loan_Status')
grid.map(sns.countplot, 'Credit_History')
plt.show()


# From above plot,we can say that having credit history makes chances to get loan easier.

# In[13]:


grid = sns.FacetGrid(data, col = 'Loan_Status')
grid.map(sns.countplot, 'Dependents')
plt.show()


# From above plot,we can say that having depedent=1 has chance of getting loan.

# In[14]:


grid = sns.FacetGrid(data, col = 'Loan_Status')
grid.map(sns.countplot, 'Self_Employed')
plt.show()


# From above plot,we can say that not self employed applicants have more chance of getting loan. 

# In[15]:


data.drop('Loan_ID', axis=1, inplace=True)#dropping unrequired column:Loan_ID


# In[16]:


data.duplicated().any()#finding whether there are duplicated rows


# In[17]:


data.groupby('Loan_Status').median()


# From this we can say that,having more co applicant income has chance of getting loan.

# HANDLING MISSING VALUES

# In[21]:


data.isnull().sum()


# Gender,married,Dependents,Self_employed,Loanamount,loan amount term and credit history have missing values.

# In[22]:


data.head()


# In[23]:


#filling missing values with mean of numerical features
data["LoanAmount"] = data["LoanAmount"].fillna(data["LoanAmount"].mean())
data["Loan_Amount_Term"] = data["Loan_Amount_Term"].fillna(data["Loan_Amount_Term"].mean())
data["Credit_History"]= data["Credit_History"].fillna(data["Credit_History"].mean())


# In[24]:


cat_data = []#separatley finding categirical features
num_data = []
for i,c in enumerate(data.dtypes):
    if c == object:
        cat_data.append(data.iloc[:, i])
    else :
        num_data.append(data.iloc[:, i])


# In[25]:


cat_data = pd.DataFrame(cat_data).transpose()
num_data = pd.DataFrame(num_data).transpose()


# In[26]:


cat_data.head()


# In[27]:


cat_data = cat_data.apply(lambda x:x.fillna(x.value_counts().index[0]))


# In[28]:


cat_data.isnull().sum().any()#finding whether there are missing values


# In[29]:


num_data.isnull().sum().any()


# In[30]:


from sklearn.preprocessing import LabelEncoder #encoding the categorical data
le = LabelEncoder()
cat_data.head()


# In[31]:


target_values = {'Y': 1 , 'N' : 0}#encoding our target variable-loan status

target = cat_data['Loan_Status']
cat_data.drop('Loan_Status', axis=1, inplace=True)

target = target.map(target_values)
target.head()


# In[32]:


for i in cat_data:#encoding all other features
    cat_data[i] = le.fit_transform(cat_data[i])
cat_data.head()


# In[33]:


df = pd.concat([cat_data, num_data, target], axis =1)


# In[34]:


df.head()


# In[36]:


X = df.drop(['Loan_Status'],axis=1)
y = df.Loan_Status


# In[37]:


xtrain, xtest, ytrain, ytest = train_test_split(X,y,test_size=0.33, random_state=42)#splitting the data
print('Train set shape:',xtrain.shape,ytrain.shape)
print('Test set shape:',xtest.shape,ytest.shape)


# MODEL BULIDING

# In[38]:


from sklearn.linear_model import LogisticRegression
model1=LogisticRegression()


# In[39]:


model1.fit(xtrain,ytrain)


# In[40]:


from sklearn.metrics import  accuracy_score


# In[41]:


y_pred1=model1.predict(xtest)


# In[43]:


A1=accuracy_score(ytest,y_pred1)*100
A1


# In[44]:


from sklearn.ensemble import RandomForestClassifier
model2=RandomForestClassifier(n_estimators=200)


# In[45]:


model2.fit(xtrain,ytrain)
y_pred2=model2.predict(xtest)
A2=accuracy_score(ytest,y_pred2)*100
A2


# In[46]:


from sklearn.ensemble import AdaBoostClassifier


# In[47]:


model3=AdaBoostClassifier(n_estimators=100)
model3.fit(xtrain,ytrain)
y_pred3=model3.predict(xtest)
A3=accuracy_score(ytest,y_pred3)*100
A3

#SUMMARY
#I analyzed the data with some steps like data visualization.Later,I preprocessed the data using labelencoding categorical features and mean of numerical and handled the missing values.I trained the model with 3 different models and finally selected the more accuracy one i.e.,Logistic regression.Atlast,I dumped the model into a pkl file.
# In[55]:


import joblib


# In[56]:


joblib.dump(model1,open('loan_model.pkl','wb'))


# In[ ]:




