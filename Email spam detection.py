#!/usr/bin/env python
# coding: utf-8

# In[130]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib 
import matplotlib.pyplot as plt


# In[153]:


df = pd.read_csv("C:\\Users\\DELL\\Downloads\\mail_data.csv")


# In[154]:


df.head()


# In[155]:


data =df.where((pd.notnull(df)),'')


# In[156]:


data.head(10)


# In[157]:


data.info()


# In[158]:


data.shape


# In[159]:


data.loc[data['Category'] == 'spam','Category',] = 0
data.loc[data['Category'] == 'ham','Category',] = 1


# In[160]:


X= data['Message']
Y= data['Category']


# In[161]:


print(X)


# In[162]:


print(Y)


# In[163]:


X_train,X_test,Y_train,Y_test= train_test_split(X,Y,test_size=0.2,random_state=3)


# In[164]:


print(X.shape)
print(X_train.shape)
print(X_test.shape)


# In[165]:


print(Y.shape)
print(Y_train.shape)
print(Y_test.shape)


# In[166]:


feature_extraction = TfidfVectorizer(min_df = 1, stop_words = 'english',lowercase = 'True')
X_train_features = feature_extraction.fit_transform(X_train)
X_train_features = feature_extraction.transform(X_test)

Y_train = Y_train.astype('int')
Y_test  = Y_test.astype('int')


# In[167]:


print(X_train)


# In[168]:


print(X_train_features)


# In[195]:


from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()
model.fit(X_train_count,Y_train)


# In[196]:


from sklearn.metrics import confusion_matrix , recall_score , precision_score
from sklearn.metrics import accuracy_score


# In[197]:


mail_ham = ['Same. Wana plan a trip sometme then']
mail_ham_count = cv.transform(mail_ham)
Y_pred = model.predict(mail_ham_count)
Y_pred


# In[198]:


model.score(X_train_count,Y_train)


# In[199]:


X_test_count = cv.transform(X_test)
model.score(X_test_count,Y_test)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




