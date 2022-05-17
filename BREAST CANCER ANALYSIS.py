#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# In[3]:


breast_cancer = pd.read_csv("C:\\Users\\Dell\\Downloads\\breast_cancer.csv")


# In[5]:


breast_cancer.head()


# In[5]:


breast_cancer.tail()


# In[6]:


breast_cancer.shape


# In[7]:


breast_cancer.columns


# In[8]:


breast_cancer.info()


# In[9]:


breast_cancer.describe()

breast_cancer.isnull().sum()
# In[10]:


breast_cancer.dropna(inplace = True)


# In[11]:


breast_cancer.isnull().sum()


# In[12]:


breast_cancer.nunique()


# In[13]:


breast_cancer.Gender.unique()


# In[14]:


breast_cancer.Gender.value_counts()


# In[15]:


plt.figure(figsize=(15, 6))
sns.countplot('Gender', data = breast_cancer)
plt.xticks(rotation =0)
plt.show()


# In[16]:


plt.figure(figsize=(15,6))
explode = [0.3, 0.02]
colors = sns.color_palette('bright')
plt.pie(breast_cancer['Gender'].value_counts(), labels=['Female','Male'], colors = colors, autopct ='%0.0f%%', explode = explode, shadow = 'True', startangle = 90)
plt.show()


# In[17]:


bins = list(range(20, 105, 5))
plt.figure(figsize = (8,5))
plt.hist(breast_cancer['Age'].astype(int), width =4, align ='mid', bins = bins, color = 'red', edgecolor = 'black')
plt.xticks(bins)
plt.xlabel('Ages')
plt.title('Ages in dataset')
plt.yticks(np.arange(0,65,5))
plt.show()


# In[18]:


breast_cancer.Histology.unique()


# In[19]:


breast_cancer.Histology.value_counts()


# In[20]:


plt.figure(figsize = (15,6))
sns.countplot('Histology', data = breast_cancer)
plt.xticks(rotation =0)
plt.show()


# In[21]:


plt.figure(figsize = (15, 6))
explode = [0.3, 0.02, 0.01]
colors = sns.color_palette('bright')
plt.pie(breast_cancer['Histology'].value_counts(), labels = ['Infiltrating Ductal Carcinoma', 'Infiltrating Lobular Carcinoma', 'Mucinous Carcinoma'],
       colors = colors, autopct = '%0.0f%%', explode = explode, shadow = 'True', startangle = 90)
plt.show()


# In[22]:


breast_cancer.Tumour_Stage.unique()


# In[23]:


breast_cancer.Tumour_Stage.value_counts()


# In[24]:


breast_cancer['Age'].head()


# In[25]:


plt.figure(figsize = (15, 6))
sns.countplot(x = 'Age', hue ='Tumour_Stage', data = breast_cancer)
plt.xticks(rotation=0)
plt.show()


# In[26]:


plt.figure(figsize=(15,6))
sns.countplot(x ='Age', hue ='Histology', data = breast_cancer)
plt.xticks(rotation =0)
plt.show()


# In[27]:


protein_types = breast_cancer[['Protein1', 'Protein2', 'Protein3', 'Protein4']]


# In[28]:


for i in protein_types.columns:
    sns.boxplot(x=protein_types[i], orient ='h', palette ='Set2')
    plt.show()


# In[29]:


breast_cancer_type_protein = breast_cancer[['Histology','Protein1', 'Protein2', 'Protein3', 'Protein4']]


# In[30]:


breast_cancer_type_protein.head()


# In[31]:


plt.figure(figsize=(15,6))
sns.barplot(x = 'Histology', y='Protein1', data =breast_cancer_type_protein)
plt.xticks(rotation =0)
plt.show()


# In[32]:


plt.figure(figsize=(15, 6))
sns.barplot(x = 'Histology', y='Protein2', data = breast_cancer_type_protein)
plt.xticks(rotation =0)
plt.show()


# In[33]:


plt.figure(figsize=(15, 6))
sns.barplot(x = 'Histology', y='Protein3', data = breast_cancer_type_protein)
plt.xticks(rotation =0)
plt.show()


# In[34]:


plt.figure(figsize=(15, 6))
sns.boxplot(x = 'Histology', y='Protein1', data = breast_cancer_type_protein)
plt.xticks(rotation =0)
plt.show()


# In[35]:


plt.figure(figsize=(15, 6))
sns.boxplot(x = 'Histology', y='Protein2', data = breast_cancer_type_protein)
plt.xticks(rotation =0)
plt.show()


# In[36]:


plt.figure(figsize=(15, 6))
sns.boxplot(x = 'Histology', y='Protein3', data = breast_cancer_type_protein)
plt.xticks(rotation =0)
plt.show()


# In[37]:


breast_cancer_stage_protein = breast_cancer[['Tumour_Stage','Protein1', 'Protein2', 'Protein3', 'Protein4' ]]


# In[38]:


breast_cancer_stage_protein.head()


# In[39]:


plt.figure(figsize=(15, 6))
sns.barplot(x = 'Tumour_Stage', y='Protein1', data = breast_cancer_stage_protein)
plt.xticks(rotation =0)
plt.show()


# In[40]:


plt.figure(figsize=(15, 6))
sns.barplot(x = 'Tumour_Stage', y='Protein2', data = breast_cancer_stage_protein)
plt.xticks(rotation =0)
plt.show()


# In[41]:


plt.figure(figsize=(15, 6))
sns.barplot(x = 'Tumour_Stage', y='Protein3', data = breast_cancer_stage_protein)
plt.xticks(rotation =0)
plt.show()


# In[42]:


plt.figure(figsize=(15, 6))
sns.boxplot(x = 'Tumour_Stage', y='Protein1', data = breast_cancer_stage_protein)
plt.xticks(rotation =0)
plt.show()


# In[43]:


plt.figure(figsize=(15, 6))
sns.boxplot(x = 'Tumour_Stage', y='Protein2', data = breast_cancer_stage_protein)
plt.xticks(rotation =0)
plt.show()


# In[44]:


plt.figure(figsize=(15, 6))
sns.boxplot(x = 'Tumour_Stage', y='Protein3', data = breast_cancer_stage_protein)
plt.xticks(rotation =0)
plt.show()


# In[45]:


breast_cancer_age_protein = breast_cancer[['Age', 'Protein1', 'Protein2', 'Protein3', 'Protein4' ]]


# In[46]:


breast_cancer_age_protein.head()


# In[47]:


plt.figure(figsize=(15, 6))
sns.barplot(x = 'Age', y='Protein1', data = breast_cancer_age_protein)
plt.xticks(rotation =0)
plt.show()


# In[48]:


plt.figure(figsize=(15, 6))
sns.barplot(x = 'Age', y='Protein2', data = breast_cancer_age_protein)
plt.xticks(rotation =0)
plt.show()


# In[49]:


plt.figure(figsize=(15, 6))
sns.barplot(x = 'Age', y='Protein3', data = breast_cancer_age_protein)
plt.xticks(rotation =0)
plt.show()


# In[50]:


n_markers = breast_cancer[['Histology', 'ER status', 'PR status', 'HER2 status']]


# In[51]:


n_markers.head()


# In[52]:


n_markers['ER status'].unique()


# In[53]:


n_markers['ER status'].value_counts()


# In[54]:


n_markers['PR status'].value_counts()


# In[55]:


n_markers['HER2 status'].value_counts()


# In[56]:


plt.figure(figsize=(15, 6))
sns.countplot(x = 'ER status', hue ='Patient_Status', data = breast_cancer)
plt.xticks(rotation =0)
plt.show()


# In[57]:


plt.figure(figsize=(15, 6))
sns.countplot(x = 'PR status', hue ='Patient_Status', data = breast_cancer)
plt.xticks(rotation =0)
plt.show()


# In[58]:


plt.figure(figsize=(15, 6))
sns.countplot(x = 'HER2 status', hue ='Patient_Status', data = breast_cancer)
plt.xticks(rotation =0)
plt.show()


# In[59]:


plt.figure(figsize=(15, 6))
sns.countplot(x = 'Age', hue ='Patient_Status', data = breast_cancer)
plt.xticks(rotation =0)
plt.show()


# In[60]:


plt.figure(figsize=(15, 6))
sns.countplot(x = 'Tumour_Stage', hue ='Patient_Status', data = breast_cancer)
plt.xticks(rotation =0)
plt.show()


# In[61]:


from sklearn import preprocessing
label_encoder = preprocessing.LabelEncoder()


# In[62]:


breast_cancer['Tumour_Stage'] = label_encoder.fit_transform(breast_cancer['Tumour_Stage'])


# In[63]:


breast_cancer['Histology'] = label_encoder.fit_transform(breast_cancer['Histology'])


# In[64]:


breast_cancer['ER status'] = label_encoder.fit_transform(breast_cancer['ER status'])


# In[65]:


breast_cancer['PR status'] = label_encoder.fit_transform(breast_cancer['PR status'])


# In[66]:


breast_cancer['HER2 status'] = label_encoder.fit_transform(breast_cancer['HER2 status'])


# In[67]:


breast_cancer['Surgery_type'] = label_encoder.fit_transform(breast_cancer['Surgery_type'])


# In[68]:


breast_cancer['Patient_Status'] = label_encoder.fit_transform(breast_cancer['Patient_Status'])


# In[72]:


x = breast_cancer.drop(['Patient_ID', 'Age', 'Gender',
                       'Date_of_Surgery', 'Date_of_Last_Visit'], axis =1)
y = breast_cancer.Patient_Status


# In[73]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


# In[74]:


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size =0.3)


# In[75]:


model =LogisticRegression()
model.fit(X_train, y_train)


# In[76]:


y_pred = model.predict(X_test)


# In[77]:


print("Training Accuracy: ", model.score(X_train, y_train))
print("Testing Accuracy: ", model.score(X_test, y_test))


# In[ ]:





# In[ ]:





# In[ ]:




