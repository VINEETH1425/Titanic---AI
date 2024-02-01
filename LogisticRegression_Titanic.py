#!/usr/bin/env python
# coding: utf-8

# In[98]:


# remember for categorical data we have to go with the countplot only.


# In[99]:


### logistic regression


# In[100]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[101]:


## load the necessary file
df = pd.read_csv("titanic-training-data.csv")


# In[102]:


df.shape # there will be no parenthesis for the shape and dtypes 


# In[103]:


df.dtypes


# In[104]:


df.head()
#here Embarked means the port at which the passenger gone into the ship


# In[105]:


df.info()


# In[106]:


df.isnull().sum()


# # EDA

# # Exploratory Data Analysis (EDA)

# In[107]:


### analyze the dependent variable, dependent variable is Survived. Here Survived is a Categorical.So go with the countplot
sns.countplot(x="Survived",data=df)


# In[108]:


## visualization
sns.countplot(x="Survived",hue="Sex",data=df)


# In[109]:


sns.countplot(x="Survived",hue="Pclass",data=df)


# In[110]:


df.drop(["PassengerId","Name","Ticket","Fare","Cabin"],axis=1,inplace =True)
df.head()


# In[111]:


df.hist(figsize=(20,30))
plt.show()


# In[112]:


sns.countplot(x="SibSp",data=df)


# In[113]:


df.isnull().sum()


# In[114]:


sns.heatmap(df.isnull())
#if we want to change the colour use cmap 


# In[115]:


sns.heatmap(df.isnull(),yticklabels=False,cmap="viridis") # in the y axis if we dont want any labels. (yticklabels)
# here we are going to show the missing values in the form of heatmap. We are not able to see the missing value of Embarked because there are only 2. that are not visible


# In[116]:


sns.boxplot(x="Pclass",y="Age",data=df)


# In[117]:


df.dropna(inplace = True)
#here I had dropped the missing values just for simplicity. If not we can replace the missing values with the median and mode.


# In[118]:


sns.heatmap(df.isnull(),yticklabels=False,cmap='viridis')


# In[119]:


df.isnull().sum()


# In[120]:


df.info()


# In[121]:


df = pd.get_dummies(df,columns=["Pclass","Sex","Embarked"])
#here we will be converting the categorial variables into dummies or indicator variables.


# In[122]:


df.isnull().sum()


# In[123]:


df.dtypes


# In[124]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


# In[125]:


X = df.drop(['Survived'],axis=1)
y = df[['Survived']]


# In[126]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.30,random_state=42)


# In[127]:


## fit the model


# In[128]:


import warnings
warnings.filterwarnings("ignore")


# In[129]:


#model = LogisticRegression(solver="lbfgs") we can write like this also. But follow the below one
model = LogisticRegression()
model.fit(X_train,y_train)
model


# In[130]:


model.score(X_train,y_train)


# In[131]:


model.score(X_test,y_test)


# In[132]:


predictions = model.predict(X_test)


# In[133]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test,predictions)


# In[134]:


from sklearn import metrics


# In[135]:


print(metrics.classification_report(y_test,predictions))


# In[136]:


cm = metrics.confusion_matrix(y_test,predictions,labels=[1,0])
df_cm = pd.DataFrame(cm,index = [i for i in ["1","0"]],
                    columns = [i for i in ["Predict 1","Predict 0"]])
plt.figure(figsize=(7,5))
sns.heatmap(df_cm,annot=True,fmt='g')


# In[137]:


from sklearn.tree import DecisionTreeClassifier
# this is a very good model but it always overfit. it means it deals good with the train but treats bad test.


# In[224]:


model2 =  DecisionTreeClassifier(max_depth=1)


# In[225]:


model2.fit(X_train,y_train)


# In[226]:


model2.score(X_train,y_train) 
#if we observe that the score has been increased with the decisiontree because it does good with the train


# In[227]:


model2.score(X_test,y_test)
#but here the score reduces. This shows overfitting takes place.


# In[ ]:




