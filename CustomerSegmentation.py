#!/usr/bin/env python
# coding: utf-8

# ## Importing necessary packages

# In[1]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
#from sklearn.impute import KNNImputer
from scipy.stats import chi2_contingency
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
from sklearn import metrics
#from sklearn.feature_extraction.text import TfidfVectorizer
#tfidf = TfidfVectorizer()
import seaborn as sb
from sklearn.cluster import KMeans
import xlrd
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.decomposition import PCA
import warnings
import pickle
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.decomposition import PCA

labelEncoder = LabelEncoder()
get_ipython().run_line_magic('matplotlib', 'inline')
scaler = MinMaxScaler()
warnings.filterwarnings('ignore')
sb.set(context="notebook", palette="Spectral", style = 'darkgrid' ,font_scale = 1.5, color_codes=True)


# ## Reading data

# In[2]:


df = pd.read_csv('QUB_Insurance_Data_Assignment_Training.csv')


# In[405]:


df1 = pd.read_csv('QUB_Insurance_Data_Assignment_Scoring.csv')


# ## Cleaning and Preprocessing of data
# 
# 1. We can see that there are null values,categorical variables, other variables that provide the same data and 'PerfChannel' having 3 different categories that denote the same channels. 
# 2. Therefore we impute the null values and ignore the columns with correlated data and also avoid the 'Occupation' column with string data as we can see that from the data, each occupation only covers 1-5 people and most rows are found to be None.
# 3. The other categorical variables with multiples classes can be encoded using a LabelEncoder as the categories are distinguishable and less in number.
# 4. The function module can be used for the scoring dataset as well.

# In[279]:


def preprocessing(df):
    try:
        df=df.fillna(0)
        df=df.set_index('CustomerID')
        #target=df['PrefChannel']
        df=df.drop(['GivenName','MiddleInitial','Surname','Occupation','Gender'],axis=1)
        df['Title'] = labelEncoder.fit_transform(df['Title'].astype(str))
        #df['Gender'] = labelEncoder.fit_transform(df['Gender'].astype(str))
        df['CreditCardType'] = labelEncoder.fit_transform(df['CreditCardType'].astype(str))
        df['Location'] = labelEncoder.fit_transform(df['Location'].astype(str))
        df['MotorInsurance'] = labelEncoder.fit_transform(df['MotorInsurance'].astype(str))
        df['MotorType'] = labelEncoder.fit_transform(df['MotorType'].astype(str))
        df['HealthInsurance'] = labelEncoder.fit_transform(df['HealthInsurance'].astype(str))
        df['HealthType'] = labelEncoder.fit_transform(df['HealthType'].astype(str))
        df['TravelInsurance'] = labelEncoder.fit_transform(df['TravelInsurance'].astype(str))
        df['TravelType'] = labelEncoder.fit_transform(df['TravelType'].astype(str))
        df['MotorValue'] = scaler.fit_transform(df['MotorValue'].values.reshape(-1,1))
        df['Age'] = scaler.fit_transform(df['Age'].values.reshape(-1,1))
        return df
    except Exception as e:
        print("Exception in preprocessing(): ",e)
        return None


# In[281]:


df=preprocessing(df)


# In[282]:


label=df['PrefChannel']
df=df.drop(['PrefChannel'],axis=1)


# In[285]:


df.head()


# ## Finding correlation between variables

# In[286]:


corr=df.corr()
corr.style.background_gradient(cmap='coolwarm').set_precision(2)


# 1. We can see that correlations can be mapped between features corresponding to Motor Insurance, Health Insuranc and Travel Insurance and a positive correlation between Health and Age.

# ## Clustering into categories using Unsupervised Learning
# 
# 1. PCA Component Analysis: This is a dimensionality reduction technique that would find the features that are most important to distinguish categories and reduce the dimensionality of the dataset.
# 2. Elbow-Graph: This is used to identify the number of distinguishable clusters
# 3. Clustering: We use K-Means clustering.

# In[287]:


def clustering(k,data):
    try:
        kmeans = KMeans(n_clusters = k, init = 'k-means++', random_state = 42)
        model = kmeans.fit(data)
        clusters = model.predict(data)
        X = np.array(data)
        plt.figure(figsize=(15,7))
        sb.scatterplot(X[clusters == 0, 0], X[clusters == 0, 1], color = 'darkgoldenrod', label = 'Cluster 1',s=50)
        sb.scatterplot(X[clusters == 1, 0], X[clusters == 1, 1], color = 'firebrick', label = 'Cluster 2',s=50)
        sb.scatterplot(X[clusters == 2, 0], X[clusters == 2, 1], color = 'orange', label = 'Cluster 3',s=50)
        #sb.scatterplot(X[clusters == 3, 0], X[clusters == 3, 1], color = 'yellow', label = 'Cluster 4',s=50)
        print(model.cluster_centers_)
        centers = np.array(model.cluster_centers_)
        plt.scatter(centers[:,0], centers[:,1], marker="x", color='g')
        plt.grid(False)
        plt.title('Clusters')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.legend()
        plt.savefig('Clusters')
        plt.show()
        return model,clusters
    except Exception as e:
        print("Exception during KMeans clustering - ",e)


# In[288]:


def PCAnalysis(data):
    try:
        pca = PCA(n_components=len(list(data.columns)))
        principalComponents = pca.fit_transform(data)
        PCA_components = pd.DataFrame(principalComponents)
        features = range(pca.n_components_)
        plt.bar(features, pca.explained_variance_ratio_, color='darkorange')
        plt.xlabel('PCA features')
        plt.ylabel('variance %')
        plt.xticks(features)
        plt.savefig("PCA")
        return PCA_components
    except Exception as e:
        print("Exception during PCA - ",e)


# In[289]:


def Elbow_Graph(n,PCA_components):
    try:
        wcss = []
        for i in range(1, 11):
            kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
            kmeans.fit(PCA_components.iloc[:,:n])
            wcss.append(kmeans.inertia_)
        plt.figure(figsize=(10,5))
        sb.lineplot(range(1, 11), wcss,marker='o',color='red')
        plt.title('The Elbow Method')
        plt.xlabel('Number of clusters')
        plt.ylabel('WCSS')
        plt.savefig('ElbowGraph')
        plt.show()
        return None
    except Exception as e:
        print("Exception during plotting Elbow Graph - ",e)


# In[290]:


PCA_components=PCAnalysis(df)


# 1. We can find that after the first three features, there is a dip in importance of the other categories
# 2. We run the Elbow Graph check to find distingushable clusters.

# In[291]:


Elbow_Graph(3,PCA_components)


# 1. We can find the elbow to be closer to 3 clusters.
# 2. Hence we categorise the data into 3 categories.

# In[292]:


clustering_model,clusters = clustering(3,PCA_components.iloc[:,:3])


# ## Writing the clustered categories and encoding target varible for predictive analysis

# In[293]:


label=label.replace('E','Email').replace('P','Phone').replace('S','SMS')


# In[294]:


temp['Cluster']=pd.Series(clusters,index=df.index)
df['Cluster']=pd.Series(clusters,index=df.index)


# In[295]:


df['PrefChannel']=label
df['PrefChannel'] = labelEncoder.fit_transform(df['PrefChannel'].astype(str))


# In[296]:


df['PrefChannel'].value_counts()


# ## Model training and analysis using Training Data

# In[297]:


X=df.drop(['PrefChannel'],axis=1)
y=df['PrefChannel']


# In[299]:


corr=X.corr()
corr.style.background_gradient(cmap='coolwarm').set_precision(2)


# In[307]:


df.apply(lambda x: chi2_contingency(pd.crosstab(x,df.PrefChannel)))


# In[336]:


df.columns


# In[369]:


X=df.drop([ 'Age', 'Location', 'MotorInsurance', 'MotorType', 'HealthInsurance', 'HealthType',
       'HealthDependentsAdults', 'HealthDependentsKids', 'TravelInsurance',
       'TravelType', 'PrefChannel'],axis=1)
y=df['PrefChannel']


# In[370]:


X_train, X_test, y_train, y_test = train_test_split(X, y,random_state=0)


# ## Logistic Regression

# In[371]:


logreg = LogisticRegression()
LRModel=logreg.fit(X_train,y_train)
y_pred = LRModel.predict(X_test)
print("Accuracy Score:",accuracy_score(y_test, y_pred)*100)
score=precision_score(y_test,y_pred,average='macro')
print("Precision is", score*100)


# ## Decision Tree

# In[372]:


DTree = DecisionTreeClassifier()
DTModel=DTree.fit(X_train,y_train)
y_pred = DTModel.predict(X_test)
print("Accuracy Score:",accuracy_score(y_test, y_pred)*100)
score=precision_score(y_test,y_pred,average='macro')
print("Precision is", score*100)


# ## Gaussian NB

# In[373]:


GNB = GaussianNB()
GNBModel=GNB.fit(X_train,y_train)
y_pred = GNBModel.predict(X_test)
print("Accuracy Score:",accuracy_score(y_test, y_pred)*100)
score=precision_score(y_test,y_pred,average='macro')
print("Precision is", score*100)


# ## K-Nearest Neighbours

# In[379]:


knn = KNeighborsClassifier(n_neighbors=5)
KNNModel=knn.fit(X_train,y_train)
y_pred = KNNModel.predict(X_test)
print("Accuracy Score:",accuracy_score(y_test, y_pred)*100)
score=precision_score(y_test,y_pred,average='macro')
print("Precision is", score*100)


# ## Random Forest Classifier

# In[375]:


clf = RandomForestClassifier(n_estimators = 100)   
RFCModel=clf.fit(X_train,y_train)
y_pred = RFCModel.predict(X_test)
print("Accuracy Score:",accuracy_score(y_test, y_pred)*100)
score=precision_score(y_test,y_pred,average='macro')
print("Precision is", score*100)


# ## Linear SVC

# In[376]:


lsvc = LinearSVC()
LSVC=lsvc.fit(X_train, y_train)
y_pred = LSVC.predict(X_test)
accuracy=accuracy_score(y_test,y_pred)
print("Accuracy Score:",accuracy_score(y_test, y_pred)*100)
score=precision_score(y_test,y_pred,average='macro')
print("Precision is", score*100)


# ## Cross Validation and Selection

# In[377]:


models = [
    RandomForestClassifier(n_estimators=200, max_depth=3, random_state=0),
    LinearSVC(),
    KNeighborsClassifier(n_neighbors=5),
    DecisionTreeClassifier(),
    GaussianNB(),
    LogisticRegression(random_state=0),
]
CV = 5
cv_df = pd.DataFrame(index=range(CV * len(models)))
entries = []
for model in models:
    model_name = model.__class__.__name__
    accuracies = cross_val_score(model, X_train, y_train, scoring='accuracy', cv=CV)
    for fold_idx, accuracy in enumerate(accuracies):
        entries.append((model_name, fold_idx, accuracy))
cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy'])


sb.boxplot(x='model_name', y='accuracy', data=cv_df)
sb.stripplot(x='model_name', y='accuracy', data=cv_df, 
              size=8, jitter=True, edgecolor="gray", linewidth=2)
plt.show()


# In[378]:


cv_df.groupby('model_name').accuracy.mean()


# ## Model Selected : K-Nearest Neighbours
# 
# ### Classification Report and Confusion Matrix

# In[381]:


print(classification_report(y_test,y_pred))


# In[382]:


cf_matrix=metrics.confusion_matrix(y_test, y_pred)
sb.heatmap(cf_matrix/np.sum(cf_matrix), annot=True, 
            fmt='.2%', cmap='Blues')


# ## Comparing Categories clustered to target variable

# In[383]:


print(pd.crosstab(df.Cluster, df.PrefChannel))

fig = plt.figure(figsize=(4,4))
df.groupby('PrefChannel')['Cluster'].count().plot.bar(ylim=0)
plt.show()


# 1. We can find that the categorization doesnt prove significant enought for the prediction of channels.
# 2. Hence we clean the scoring data and use the features as well for predictive analysis.

# ## Retraining the model using entire Training Data

# In[384]:


knn = KNeighborsClassifier(n_neighbors=5)
KNNModel=knn.fit(X,y)


# In[406]:


scoring_df=preprocessing(df1)


# In[407]:


PCA2=PCAnalysis(scoring_df)


# In[408]:


scoring_df.head()


# In[409]:


result=clustering_model.predict(PCA2.iloc[:,:3])


# In[410]:


#temp1['Cluster']=result
scoring_df['Cluster']=result
#df1['Cluster']=result


# In[411]:


scoring_df=scoring_df.drop([ 'Age', 'Location', 'MotorInsurance', 'MotorType', 'HealthInsurance', 'HealthType',
       'HealthDependentsAdults', 'HealthDependentsKids', 'TravelInsurance',
       'TravelType'],axis=1)


# In[412]:


predictions= KNNModel.predict(scoring_df)


# In[416]:


scoring_df['PrefChannel']=predictions


# In[417]:


scoring_df=scoring_df.drop(['PerfChannel'],axis=1)


# In[414]:


df1['PrefChannel']=predictions


# In[45]:


df1.to_csv("Results.csv",index=True)


# In[395]:


df1['PrefChannel'].value_counts()


# In[418]:


result=scoring_df.groupby(scoring_df['PrefChannel']).mean()


# In[419]:


result


# In[ ]:




