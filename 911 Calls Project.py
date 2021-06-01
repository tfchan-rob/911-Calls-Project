#!/usr/bin/env python
# coding: utf-8

# # 911 Calls Project

# Emergency (911) Calls: Fire, Traffic, EMS for Montgomery County, PA
# 
# You can get a quick introduction to this Dataset with this kernel: Dataset Walk-through
# 
# Acknowledgements: Data provided by montcoalert.org

# For this project we will be analyzing some 911 call data from Kaggle. The data contains the following fields:
# 
# lat : String variable, Latitude
# 
# lng: String variable, Longitude
# 
# desc: String variable, Description of the Emergency Call
# 
# zip: String variable, Zipcode
# 
# title: String variable, Title
# 
# timeStamp: String variable, YYYY-MM-DD HH:MM:SS
# 
# twp: String variable, Township
# 
# addr: String variable, Address
# 
# e: String variable, Dummy variable (always 1)

# ## Data and Setup

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


df = pd.read_csv('911.csv')


# In[4]:


df.head()


# In[5]:


df.info()


# ## Basic Questions

# ** What are the top 5 zipcodes for 911 calls? **

# In[6]:


df['zip'].value_counts().head(5)


# ** What are the top 5 townships (twp) for 911 calls? **

# In[7]:


df['twp'].value_counts().head(5)


# ** Take a look at the 'title' column, how many unique title codes are there? **

# In[8]:


df['title'].nunique()


# ## Creating new features

# ** In the titles column there are "Reasons/Departments" specified before the title code. These are EMS, Fire, and Traffic. Use .apply() with a custom lambda expression to create a new column called "Reason" that contains this string value.**
# 
# *For example, if the title column value is EMS: BACK PAINS/INJURY , the Reason column value would be EMS. *

# In[9]:


df['Reason'] = df['title'].apply(lambda title:title.split(':')[0])


# In[10]:


df['Reason']


# In[11]:


df['Reason'].value_counts()


# ** Now use seaborn to create a countplot of 911 calls by Reason. **

# In[12]:


sns.countplot(x=df['Reason'],data=df)


# ** Now let us begin to focus on time information. What is the data type of the objects in the timeStamp column? **

# In[13]:


ts= df['timeStamp']


# In[14]:


type(ts.iloc[0])


# **Use [pd.to_datetime] to convert the column from strings to DateTime objects. **

# In[15]:


df['timeStamp'] = pd.to_datetime(df['timeStamp'])


# In[16]:


type(df['timeStamp'].iloc[0])


# **Now that the timestamp column are actually DateTime objects, use .apply() to create 3 new columns called Hour, Month, and Day of Week.**

# In[17]:


df['Hour'] = df['timeStamp'].apply(lambda timeStamp:timeStamp.hour)


# In[18]:


df['Hour']


# In[19]:


df['Month'] = df['timeStamp'].apply(lambda timeStamp:timeStamp.month)


# In[20]:


df['Day of Week'] = df['timeStamp'].apply(lambda timeStamp:timeStamp.dayofweek)


# In[21]:


dmap = {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}


# In[22]:


df['Day of Week'] = df['Day of Week'].map(dmap)


# In[23]:


df['Day of Week']


# ** Now use seaborn to create a countplot of the Day of Week column with the hue based off of the Reason column. **

# In[24]:


sns.countplot(x=df['Day of Week'],data=df,hue=df['Reason'],palette='rainbow')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)


# ** Now do the same for Month: **

# In[25]:


sns.countplot(x=df['Month'],data=df,hue=df['Reason'],palette='rainbow')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)


# **Did you notice something strange about the Plot?**
# 
# _____
# 
# ** You should have noticed it was missing some Months, let's see if we can maybe fill in this information by plotting the information in another way, possibly a simple line plot that fills in the missing months, in order to do this, we'll need to do some work with pandas... **

# In[26]:


byMonth = df.groupby('Month').count()


# In[27]:


byMonth.head()


# ** Now create a simple plot off of the dataframe indicating the count of calls per month. **

# In[28]:


byMonth['twp'].plot()


# In[29]:


sns.lmplot(x='Month',y= 'twp',data=byMonth.reset_index())


# ** Create a new column called 'Date' that contains the date from the timeStamp column. You'll need to use apply along with the .date() method. ** 

# In[30]:


df['Date'] = df['timeStamp'].apply(lambda timeStamp:timeStamp.date())


# In[31]:


df['Date']


# ** Now groupby this Date column with the count() aggregate and create a plot of counts of 911 calls.**

# In[32]:


byDate = df.groupby('Date').count()


# In[33]:


byDate


# In[34]:


byDate['twp'].plot()
plt.tight_layout()


# ** Now recreate this plot but create 3 separate plots with each plot representing a Reason for the 911 call**

# In[35]:


df[df['Reason']=='EMS'].groupby('Date').count()['twp'].plot()
plt.tight_layout()
plt.title('EMS')


# In[36]:


df[df['Reason']=='Fire'].groupby('Date').count()['twp'].plot()
plt.tight_layout()
plt.title('Fire')


# In[37]:


df[df['Reason']=='Traffic'].groupby('Date').count()['twp'].plot()
plt.tight_layout()
plt.title('Traffic')


# ** Now let's move on to creating heatmaps with seaborn and our data. We'll first need to restructure the dataframe so that the columns become the Hours and the Index becomes the Day of the Week. **

# In[51]:


byHour = df.groupby(by=['Day of Week','Hour']).count()['Reason'].unstack()
byHour


# In[63]:


plt.figure(figsize=(12,6))
sns.heatmap(byHour,linecolor='white',linewidths=1)


# In[62]:


sns.clustermap(byHour,linecolor='white',linewidths=1)


# ** Now repeat these same plots and operations, for a DataFrame that shows the Month as the column. **

# In[56]:


byMonth= df.groupby(by=['Day of Week','Month']).count()['Reason'].unstack()
byMonth


# In[61]:


plt.figure(figsize=(10,5))
sns.heatmap(byMonth,cmap='coolwarm',linecolor='white',linewidths=1)


# In[64]:


sns.clustermap(byMonth,cmap='coolwarm',linecolor='white',linewidths=1)


# In[ ]:




