#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[1]:


import pandas as pd
import numpy as np


# In[2]:


#!pip install -U pandas


# In[3]:



print(pd.__version__)


# In[4]:


#pip install numpy


# ------Read the Train and Test CSV Data

# In[5]:


df=pd.read_csv('C:/Users/munis/Downloads/involveai-mle-assessment/InvolveAI Take Home Assessment/adult_trdata.csv',header=None)


# In[6]:


df_test=pd.read_csv('C:/Users/munis/Downloads/involveai-mle-assessment/InvolveAI Take Home Assessment/adult_test.csv',skiprows=1,header=None)


# In[7]:


df_test.head()


# In[8]:


df


# In[9]:


df_test


# In[10]:


df.head()


# Assign Column Names to the Test and Train Dataset

# In[11]:


df.columns=['age','workclass','fnlwgt','education','education-num','marital-status','occupation','relationship','race','sex','capital-gain','capital-loss','hours-per-week','native-country','salary']


# In[12]:


df


# In[13]:


#df_test.shift(-1, axis = 0)


# In[14]:


df_test


# In[15]:



df_test.columns=['age','workclass','fnlwgt','education','education-num','marital-status','occupation','relationship','race','sex','capital-gain','capital-loss','hours-per-week','native-country','salary']


# In[16]:


df_test.columns


# In[17]:


df_test


# ------Format the values of salary column to ensure a simpler process of assigning values later

# In[18]:


df.salary.unique()


# In[19]:


df['salary'] = df['salary'].apply(lambda x: '>50K' if x == ' >50K' else '<=50K')


# In[20]:


df_test['salary'] = df_test['salary'].apply(lambda x: '>50K' if x == ' >50K' else '<=50K')


# In[21]:


df['salary'].unique()


# In[22]:


df_test.head()


# ------Removed Unnecassary and unexplainable feature 'fnlwgt'. 
# -----Caution taken to not eliminate many features inorder to reatin information for high accuracy

# 

# In[23]:


df=df.drop(['fnlwgt'],axis=1)


# In[24]:


df_test=df_test.drop(['fnlwgt'],axis=1)


# In[25]:


df_test.shape


# In[26]:


#df.info()


# In[27]:


df.shape


# In[28]:


df_test.shape


# In[29]:


#object_col = df.select_dtypes(include=object).columns.tolist()
#for col in object_col:
#    index_names_train=df.loc[df[col]==' ?', col].index
#    index_names_test=df_test.loc[df_test[col]==' ?', col].index


# In[30]:


#df.drop(index_names_train, inplace = True)
#df_test.drop(index_names_test, inplace = True)


# -------Create a seperate list of columns with type object to ensure an easy transformation of object features

# In[31]:


object_col = df.select_dtypes(include=object).columns.tolist()


# In[32]:


object_col


# ------Data Cleaning

# -------Convert the ' ?' to nan to ensure an easier way to get rid of ' ?

# In[33]:


for col in object_col:
    df.loc[df[col]==' ?', col] = np.nan
    df_test.loc[df_test[col]==' ?', col] = np.nan


# In[34]:


df = df.dropna(axis=0, how='any')
df_test = df_test.dropna(axis=0, how='any')


# In[35]:


df.shape


# In[36]:


df_test.shape


# In[37]:


dfcat=df.dtypes[df.dtypes != 'object']


# In[38]:


dfcat


# In[39]:


import matplotlib.pyplot as plt


# In[40]:


dftotal=pd.concat([df,df_test])


# -------Create histograms plots for non object columns to check the normal distribution of the features. 
# 
# --------The figure shows that the features are not normally distributed.

# In[41]:


fig = plt.figure(figsize = (9,15));
i=0
for x in dftotal.columns:
  #if df.columns.dtypes != 'object':
   if x in dfcat: 

    ax = fig.add_subplot(3, 2, i+1)
    ax.hist(dftotal[x], bins = 25, color = 'SkyBlue')
    i=i+1
    ax.set_title("'%s' Feature Distribution"%(x), fontsize = 14)
    ax.set_xlabel("Value")
    ax.set_ylabel("Count")
    ax.set_ylim((0, 2000))
    ax.set_yticks([0, 500, 1000, 1500, 2000, 5000])
    ax.set_yticklabels([0, 500, 1000, 1500, 2000])
    print(x)
#num_col = df.dtypes[df.dtypes != 'object'].index


# In[42]:


import seaborn as sn


# --------Check for corellation between the features. 
# 
# --------This plot shows that there is no major corellation between 2 numeric features.

# In[43]:


corrMatrix = dftotal.corr()
sn.heatmap(corrMatrix, annot=True)
plt.show()


# ---------Data Preprocessing

# In[44]:


df_ind=df.drop(['salary'],axis=1)


# In[45]:


dftest_ind=df_test.drop(['salary'],axis=1)


# In[46]:


num_col = df.dtypes[df.dtypes != 'object'].index


# In[47]:


y_train=df['salary']


# In[48]:


y_test=df_test['salary']


# --------Apply log transformation to deal with skewness

# In[49]:


skewed = ['capital-gain', 'capital-loss']
df_ind[skewed] = df_ind[skewed].apply(lambda x: np.log(x + 1))
dftest_ind[skewed] = dftest_ind[skewed].apply(lambda x: np.log(x + 1))


# ----------Used MinMaxScaler() to transform features by scaling each feature 
# 

# In[50]:


from sklearn import preprocessing
scaler = preprocessing.MinMaxScaler() # default=(0, 1)

df_ind = pd.DataFrame(data = df_ind)
df_ind[num_col] = scaler.fit_transform(df_ind[num_col])

# Transform the test data set
dftest_ind = pd.DataFrame(data = dftest_ind)
dftest_ind[num_col] = scaler.fit_transform(dftest_ind[num_col])


# In[51]:


type(df_ind[num_col])


# In[259]:


df_ind


# In[ ]:





# In[261]:


df_ind.age.isna()


# In[262]:


df_ind.describe()


# In[263]:


df_ind.dtypes


# In[264]:


categorical_cols=['workclass','education','marital-status','occupation','relationship','race','sex','native-country']


# In[265]:


df_cat=df_ind[categorical_cols]


# --------Use Label Encoder to encode the 

# In[266]:


from sklearn import preprocessing
le = preprocessing.LabelEncoder()
from sklearn import preprocessing
#for x in df[categorical_cols]:
df_cat=df_cat.apply(le.fit_transform)
df_non_cat=df_ind.drop(columns=categorical_cols)

x=pd.concat([df_cat,df_non_cat],axis=1)


# In[267]:


dftest_cat=dftest_ind[categorical_cols]


# In[268]:



#for x in df[categorical_cols]:
dftest_cat=dftest_cat.apply(le.fit_transform)
dftest_non_cat=dftest_ind.drop(columns=categorical_cols)

xtest=pd.concat([dftest_cat,dftest_non_cat],axis=1)


# In[269]:


y_train = y_train.apply(lambda x: 1 if x == '>50K' else 0)


# In[270]:


y_test = y_test.apply(lambda x: 1 if x == '>50K' else 0)


# In[271]:


X=pd.concat([x,xtest])


# In[272]:


Y=pd.concat([y_train,y_test])


# In[273]:


X


# In[274]:



# Feature Selection with Univariate Statistical Tests
from pandas import read_csv
from numpy import set_printoptions
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

test = SelectKBest(score_func=f_classif, k=9)
fit = test.fit(X, Y)
# summarize scores
set_printoptions(precision=3)
print(X.columns)
print(fit.scores_)

features = fit.transform(X)


# From test result the value above and using domain knowledge we eliminate features 'race', 'occupation' and 'capital-loss'
# 

# In[ ]:





# In[275]:


#chi2


# In[276]:



# Feature Selection with Univariate Statistical Tests
from pandas import read_csv
from numpy import set_printoptions
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import chi2


# feature extraction
test = SelectKBest(score_func=chi2, k=4)
fit = test.fit(X, Y)
# summarize scores
set_printoptions(precision=3)
print(X.columns)
print(fit.scores_)
#filter = test.get_support()
features = fit.transform(X)


# In[277]:


print(features[0:5,])


# From this output we can observe the most important features according to ANOVA-ftest

# In[278]:


X.head(10)


# In[279]:


x.columns


# In[280]:


x=x[[ 'marital-status', 'relationship', 'sex', 'age', 'education-num', 'capital-gain',  'hours-per-week','workclass','native-country','education']]
xtest=xtest[[ 'marital-status', 'relationship', 'sex', 'age', 'education-num', 'capital-gain',  'hours-per-week','workclass','native-country','education']]


# Used Random Forest Classifier and XG Boost Algorithm to predict based on the test dataset
# 
# Performed Grid search to get the best values for parameters inorder to get the best result

# In[283]:


from sklearn.ensemble import RandomForestClassifier
clf=RandomForestClassifier()

parameters = {
    "n_estimators":[5,10,50,100,250,500],
    "max_depth":[2,4,8,16,32,None]
    
}
from sklearn.model_selection import GridSearchCV
cv = GridSearchCV(clf,parameters,cv=5)
cv.fit(x,y_train.values.ravel())


# In[284]:


def display(results):
    print(f'Best parameters are: {results.best_params_}')
    print("\n")
    mean_score = results.cv_results_['mean_test_score']
    std_score = results.cv_results_['std_test_score']
    params = results.cv_results_['params']
    for mean,std,params in zip(mean_score,std_score,params):
        print(f'{round(mean,3)} + or -{round(std,3)} for the {params}')


# In[285]:


display(cv)


# In[286]:


from sklearn.ensemble import RandomForestClassifier

#Create a Gaussian Classifier
clf=RandomForestClassifier(max_depth= 16, n_estimators= 500)

#Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(x,y_train)

y_pred=clf.predict(xtest)

from sklearn import metrics
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# VAlidated the Result from the Machine Learning Algorithm using Grid Search

# In[287]:


from sklearn.model_selection import KFold


# In[288]:


kf= KFold(n_splits= 5, shuffle = True)
clf=RandomForestClassifier(n_estimators=100)


# In[289]:


scores=[]
for i in range(5):
    result=next(kf.split(df), None)
    model=clf.fit(x,y_train)
    predictions=clf.predict(xtest)
    scores.append(model.score(xtest,y_test))
print('Scores from each iteration:' , scores)
print('Average K-Fold Score :', np.mean(scores))


# In[290]:




from sklearn.metrics import mean_squared_error, r2_score

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


# In[291]:


from sklearn.svm import SVC
reg = make_pipeline(StandardScaler(), SVC(C=10.0,   gamma='auto'))
reg.fit(x, y_train)
y_pred = reg.predict(xtest)
#print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
#print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
#print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
reg.score(xtest, y_test)


# In[ ]:





# Used XGBoost Algorithm to get a higher accuracy score of 0.835

# In[293]:


from sklearn.datasets import make_hastie_10_2
from sklearn.ensemble import GradientBoostingClassifier


xgb = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,
     max_depth=1, random_state=0).fit(x, y_train)
xgb.score(xtest, y_test)


# In[294]:


scores=[]
for i in range(5):
    result=next(kf.split(df), None)
    model=xgb.fit(x,y_train)
    predictions=xgb.predict(xtest)
    scores.append(model.score(xtest,y_test))
print('Scores from each iteration:' , scores)
print('Average K-Fold Score :', np.mean(scores))


# In[ ]:





# We observe that the salary range is affected by the following Index 'marital-status', 'relationship', 'sex', 'age', 'education-num', 'capital-gain',  'hours-per-week','workclass','native-country','education' features and the XGBoost Algorithm performs best with an accuracy of 83.5% 
# Also the ensemble methods perform significantly faster than SVM

# In[ ]:





# In[ ]:





# In[ ]:




