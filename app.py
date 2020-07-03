#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
from flask import Flask,request,render_template,jsonify
import joblib
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)


# In[3]:


app=Flask(__name__,template_folder='template')


# In[4]:


model=joblib.load(open('loan_model.pkl','rb'))


# In[5]:


@app.route('/',methods=['GET'])
def home():
    return render_template('loan.html')


# In[6]:


@app.route('/predict',methods=['POST'])    
def predict():
    features=[]
    
        
    input_features = [int(x) for x in request.form.values()]
    
    features.append(input_features)
    
    
                      
    prediction = model.predict(features)
    return render_template('loan.html',prediction_text="Loan is approved" if prediction == 1  else "Loan is not approved")        


# In[ ]:


if __name__=="__main__":
    app.run(debug=True,use_reloader=False)




