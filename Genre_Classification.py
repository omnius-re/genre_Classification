#!/usr/bin/env python
# coding: utf-8

# In[79]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import random


genres = pd.read_csv("genres_v2.csv")
genres.head()


# In[3]:


genres.info() #To display all columns and their datatype


# In[6]:


df = genres.drop(["id", "song_name", "uri", "track_href", "analysis_url", "Unnamed: 0", "title", "time_signature", "type", "duration_ms"], axis= 1)
#drop features that are irrelevant and cannot be scaled numerically
df.head()


# In[7]:


df.describe() #helps understand the statistical information of each feature


# In[ ]:


for i in df.columns: #plots the range of values in each features
    plt.figure()
    sns.countplot(data=df.loc[:,:"tempo"], x=i)
    plt.ylabel("Count")
    plt.title("Count of Features")
    plt.show()


# In[8]:


df["genre"].value_counts() #identifies all target values and their counts


# In[9]:


df.corr() #find how features correlate between themselves


# In[10]:


sns.heatmap(df.corr(), annot=False, cmap='coolwarm').set(title = "Correlation between Features") #plots correlation between features


# In[101]:


#drops key and mode features due to low correlations
df = genres.drop(["id", "song_name", "uri", "track_href", "analysis_url", "Unnamed: 0", "title", "time_signature", "type", "duration_ms", "key", "mode"], axis= 1)


# In[103]:


target = df["genre"]
target.value_counts()


# In[114]:


#merges similar genres together and reduces range of target values
df = df.replace("Dark Trap", "Rap")
df = df.replace("Underground Rap", "Rap")
df = df.replace("trap", "Hiphop")
df = df.replace("Hiphop", "Rap")
df = df.replace("Trap Metal", "Rap")
df = df.replace("Emo", "Rap")

#removes pop genre for insufficient row values
df.drop(df.loc[df['genre']=="Pop"].index, inplace=True)
df = df.reset_index(drop = True)


# In[115]:


drops = []
for i in range(len(df)):
    if df.iloc[i]["genre"] == "Rap":
        if random.random()<0.85:
            drops.append(i)
            
df.reset_index(drop=True, inplace=True)
df.drop(index = drops,  inplace=True)


# In[116]:


#sets danceability to tempo as features and sets genre as target
features = df.loc[:,:"tempo"]
target = df["genre"]


# In[117]:


target


# In[118]:


target.value_counts() #checks all genres and values


# In[119]:


#creates train and test data
xtrain, xtest, ytrain, ytest = train_test_split(features, target, test_size = 0.2, random_state = 42, shuffle = True)


# In[120]:


xtrain.info()


# In[121]:


ytrain.value_counts()


# In[122]:


#uses K-Neighbor Classification algorithm to create a classification model using train data
knn = KNeighborsClassifier(n_neighbors=4)
knn.fit(xtrain, ytrain)


# In[123]:


neighbors = 10
train_accs = []
test_accs = []

for neighbor in range(1, neighbors):
    
    #calculates training accuracy for every increase in neighbor count
    knn = KNeighborsClassifier(n_neighbors=neighbor)
    knn.fit(xtrain, ytrain)
    train_acc = knn.score(xtrain, ytrain)
    train_accs.append(train_acc)
    
    #calculates testing accuracy for every increase in neighbor count
    ypred = knn.predict(xtest)
    test_acc = accuracy_score(ytest, ypred)
    test_accs.append(test_acc)
    

    print(f"Neighbor {neighbor}: Training Accuracy = {train_acc}: Testing Accuracy = {test_acc}")


# In[124]:


train_accs


# In[131]:


max(test_accs)


# In[125]:


#graphs training accuracy over neighbors

plt.plot(range(1, neighbors), train_accs, 'b.-')
plt.xlabel('Neighbors')
plt.ylabel('Training Accuracy')
plt.title('Training Accuracy over Neighbors')
plt.grid(True)
plt.show()


# In[126]:


#graphs testing accuracy over neighbors

plt.plot(range(1, neighbors), test_accs, 'b.-')
plt.xlabel('Neighbors')
plt.ylabel('Testing Accuracy')
plt.title('Testing Accuracy over Neighbors')
plt.grid(True)
plt.show()


# In[127]:


#checks prediction for every row
for value in ypred:
    print(value)


# In[128]:


#checks for all inaccurate predictions

errors = []
for i in range(len(ypred)):
    if ypred[i] != (ytest.iloc[i]):
        errors.append((ytest.iloc[i], ypred[i]))

errors


# In[129]:


#shows error count for each genre label
error_counts = {}
for true_label, pred_label in errors:
    if true_label not in error_counts:
        error_counts[true_label] = 0
    else:
        error_counts[true_label] += 1

error_counts

