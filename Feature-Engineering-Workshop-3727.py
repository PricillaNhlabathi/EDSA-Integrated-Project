#!/usr/bin/env python
# coding: utf-8

# # Feature Engineering 
# 
# Feature Engineering is the process of transforming raw data into meaningful features that can be used as input for advanceced visualisations or machine learning algorithms.
# 
# It involves selecting, creating, and transforming features to hopefully enhance the dataset.
# 
# Poorly designed features can lead to a disruptive dataset. 
# 

# ## Types of Feature Engineering
# 
# * **Handling Missing Values**
# 
#     Filling missing values with appropriate strategies, e.g., mean, median, or constant values.
# 
# * **Encoding Categorical Variables**
# 
#     Converting categorical data into numeric form, such as one-hot encoding or label encoding. Only needed if you are building a model
# 
# * **Binning Numeric Variables**
# 
#     Grouping continuous data into bins or categories to simplify the representation.
# 
# * **Feature Scaling**
# 
#     Scaling features to bring them to a similar range, e.g., Min-Max scaling or Standard scaling.
# 
# * **Creating New Features**
# 
#     Generating new features by combining or transforming existing ones.
# 
# * **Handling Outliers**
# 
#     Managing extreme values that can affect model performance.
# 
# * **Feature Joining**
# 
#     Creating new features by combining multiple existing features.

# ## Imports and Dataset

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[7]:


df = pd.read_excel('insurance_claims_raw.xlsx')


# In[8]:


df.head()


# ### Missing Values

# In[9]:


null_counts = df.isnull().sum()
null_counts


# In[10]:


df_new = df.drop("_c39", axis = 1)


# In[11]:


df_new.head()


# ### Binding Numeric Data

# In[12]:


df_new.describe()


# In[13]:


# Choose the column for the histogram
column_name = 'age'

# Plot the histogram
plt.hist(df[column_name], bins=3, edgecolor='black')

# Add labels and title
plt.xlabel(column_name)
plt.ylabel('Frequency')
plt.title(f'Histogram of {column_name}')

# Display the histogram
plt.show()


# In[14]:


bin_edges = [0, 30, 55, 100]  # Define the bin edges
bin_labels = ['Young Adult', 'Middle Aged', 'Elderly']  # Corresponding labels for each bin

# Create a new column based on the bin labels
df_new['ages_category'] = pd.cut(df_new['age'], bins=bin_edges, labels=bin_labels)


# In[15]:


df_new.head()


# In[16]:


bin_edges_customer = [0, 25, 150, 500]  # Define the bin edges
bin_labels_customer = ['New Client', 'Established Client', 'Long-Term Client']  # Corresponding labels for each bin

# Create a new column based on the bin labels
df_new['customer_category'] = pd.cut(df_new['months_as_customer'], bins=bin_edges_customer, labels=bin_labels_customer)


# In[17]:


df_new.head()


# ## Creating New Features

# In[18]:


df_new["Contract Years"] = df_new["months_as_customer"]/12


# In[19]:


df_new.head()


# ## Feature Joining

# In[20]:


df_new['total_premiums_paid'] = (df_new['policy_annual_premium']/12) * df_new['months_as_customer']


# In[21]:


df_new.head()


# In[22]:


df_new['net_value_of_customer'] = df_new['total_premiums_paid'] - df_new['total_claim_amount']


# In[23]:


df_new.head()


# ## Saving the csv for late

# In[24]:


df_new.to_csv('Advanced Features Claims Data.csv')


# ## Go wild
# 
# Go out a see what other features you can create that will be useful for our visualisations

# In[ ]:




