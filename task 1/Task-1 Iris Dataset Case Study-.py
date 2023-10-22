#!/usr/bin/env python
# coding: utf-8

# # Lets Grow More 
# ## Data Science Intern
# ### TASK 1 :- Iris Flowers Classification ML Project
# 

# ### Objective: 
# Our main objective is to classify the flowers into their respective species - Iris setosa, Iris virginica and Iris versicolor by using various possible plots.

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# ## Reading the Data

# In[6]:


data = pd.read_csv('Iris.csv')
data.head()


# ## Data Inspection

# In[7]:


data.shape


# In[8]:


data.info()


# Conclusion` : As can be seen from the above info, it is a balanced dataset.

# ## Statistical Insights

# In[9]:


data.describe()


# In[10]:


data.columns


# In[11]:


data['Species'].value_counts()


# In[12]:


#Dropping unnecessary columns
data = data.drop('Id',axis=1)
data.head(3)


# ## Analysing the Dataset

# ### Understanding the specieswidth vs specieslength distribution using a Scatter Plot

# In[13]:


sns.scatterplot(data=data,x='SepalLengthCm', y='SepalWidthCm')
plt.show()


# Conclusion:` The plot doesnot convey much information about the nature of distribution of the sepal length vs sepal width. Hence, we would use different colours(based on their class type) to interpret the distribution nature.

# In[14]:


plt.figure(figsize=(20,20))
sns.set_style("whitegrid")
sns.FacetGrid(data, hue="Species").map(plt.scatter,'SepalLengthCm','SepalWidthCm').add_legend()
plt.show()


# ### `Conclusion:`
# 
# * The blue points represent the Setosa species and as seen from the graph, it can be easily seperated from the other species based on the width and length measurements.
# * However, we cannot seperate the other two species i.e. Versicolor and Virginica by using the same graph as they overlap each other. Hence we would plot a pairplot to furthur analyze our data.
# 
# 
# ### `Note`:
# We are using a pairplot to check all the possible combinations of graphs that can be plotted, so as to find a way out to seggregate the Versicolor and Virginica flowers as well.

# In[15]:


plt.figure(figsize=(10,10))
sns.set_style("whitegrid")
sns.pairplot(data = data, hue="Species", size=3);
plt.show()


# ### Inferences
# * The graph for Petal Width vs Petal Length shows the best results and can be used to seggregate all the three species.
# * In can be clearly seen from the graph, the Setosa species can be easily seperated , on the other hand, we can see some overlaps in the case of Versicolor and Virginica , however they can be linearly seperated.

# ## Correlation Between the numeric variables

# In[16]:


data.corr()


# ## Plotting the correlation using a heatmap

# In[17]:


plt.figure(figsize = (10,6))
sns.heatmap(data.corr(), cmap='Blues', annot = True)


# ### Conclusion:
# 
# * From the graph, it can be clearly seen that the columns Petal length and petal width hold a strong correlation (=0.96). Earlier we had got the same observations while plotting a pairplot.
# * Apart from this, the columns Sepal length and Petal length also hold a high corelation(=0.87).
# * Sepal length and Petal width alzo hold a good correlation (=0.82).

# ## Observing the distribution nature of all the 4 columns (Using a Distplot)

# In[18]:


data.columns


# In[19]:


cols = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
plt.figure(figsize=(35,5))
i = 1
for col in cols:
    plt.subplot(1,10,i)
    sns.distplot(data[col])
    i = i+1
plt.tight_layout()
plt.show()


# ### Conclusion:
# It can be observed from the above distplots that the distribution of the columns follow a Normal distribution

# ## Observing the distribution of the data across the various columns using a Histogram

# In[20]:


data.columns


# In[21]:


data.describe()


# Using the describe() we can see that the maximum value for the columns:
# 
# * Sepal Length = 8
# * Sepal Width = 5
# * Petal length = 7
# * Petal Width = 3
# 
# Hence, we'll set our bin size of the histogram based on the above specified values.

# In[22]:


fig, axes = plt.subplots(2, 2, figsize=(15,10))
axes[0,0].set_title("Distribution of Sepal Width")
axes[0,0].hist(data['SepalWidthCm'], bins=5);
axes[0,1].set_title("Distribution of Sepal Length")
axes[0,1].hist(data['SepalLengthCm'], bins=8);
axes[1,0].set_title("Distribution of Petal Width")
axes[1,0].hist(data['PetalWidthCm'], bins=5);
axes[1,1].set_title("Distribution of Petal Length")
axes[1,1].hist(data['PetalLengthCm'], bins=8)
plt.show()


# ### Conclusions:
# * The highest frequency of Sepal length ranges between 6.0 - 6.5 which is around 32
# * The highest frequency of Sepal Width ranges between 3.0 - 3.5 which is around 69
# * The highest frequency of Petal length ranges between 1.0 - 1.8 which is around 50
# * The highest frequency of Petal Width ranges between 0.0 - 0.5 which is around 50

# ## Univariate Analysis of all the 4 columns(using Distplots)

# In[23]:


sns.FacetGrid(data,hue="Species",height=5).map(sns.distplot,"SepalLengthCm").add_legend()


# ### Conclusion:
# It can be clearly seen that the flower species cannot be seperated based on the Sepal length as the values overlap a lot.

# In[24]:


sns.FacetGrid(data,hue="Species",height=5).map(sns.distplot,"SepalWidthCm").add_legend()


# ### Conclusion:
# It can be clearly seen that the flower species cannot be seperated based on the Sepal Width as here also, the values overlap a lot. the overlapping is more intense in this case as compared to the overlapping in the case of Sepal Length.

# In[25]:


sns.FacetGrid(data,hue="Species",height=5).map(sns.distplot,"PetalLengthCm").add_legend()


# ### Conclusion:
# From the graph, it can be seen that Setosa is easily segregable, whereas Versicolor and Virginica do overlap at some points (near 4.5-5). The column Petal length can be used to seperate the species 

# In[26]:


sns.FacetGrid(data,hue="Species",height=5).map(sns.distplot,"PetalWidthCm").add_legend()


# ### Conclusion:
# From the graph, it can be seen that Setosa is easily segregable, whereas Versicolor and Virginica do overlap at some points (near 1.5-2.0). The column Petal Width can also be used to seperate the species .

# In[27]:


fig, axes = plt.subplots(2, 2, figsize=(16,12))
axes[0,0].set_title("Distribution of Sepal Length")
sns.boxplot(y="SepalLengthCm", x= "Species", data=data,  orient='v' , ax=axes[0, 0])
axes[0,1].set_title("Distribution of Sepal Width")
sns.boxplot(y="SepalWidthCm", x= "Species", data=data,  orient='v' , ax=axes[0, 1])
axes[1,0].set_title("Distribution of Petal Length")
sns.boxplot(y="PetalLengthCm", x= "Species", data=data,  orient='v' , ax=axes[1, 0])
axes[1,1].set_title("Distribution of Petal Width")
sns.boxplot(y="PetalWidthCm", x= "Species", data=data,  orient='v' , ax=axes[1, 1])
plt.show()


# ### Conclusions:
# 
# We can see that the species Setosa doesnot have any outliers in case of Sepal Length or Sepal Width, however, it does have few outliers in Petal length and Petal Width.
# In terms of features like: Petal Width / Length, Virginca has quiet high values as compared to the other two species. Also, Setosa has the least values for the same features.
# It is also observed that for the feature Sepal Width, Setosa has a wide range of values as compared to the other species

# ### Let's Dive a lil Deeper !!
# To furthur analyze the distribution we are using a violin plot.
# 
# Violin plots are used when you want to observe the distribution of numeric data, and are especially useful when you want to make a comparison of distributions between multiple groups. The peaks, valleys, and tails of each group's density curve can be compared to see where groups are similar or different.

# In[28]:


fig, axes = plt.subplots(2, 2, figsize=(16,12))
axes[0,0].set_title("Distribution of Sepal Length")
sns.violinplot(y="SepalLengthCm", x= "Species", data=data,  orient='v' , ax=axes[0, 0])
axes[0,1].set_title("Distribution of Sepal Width")
sns.violinplot(y="SepalWidthCm", x= "Species", data=data,  orient='v' , ax=axes[0, 1])
axes[1,0].set_title("Distribution of Petal Length")
sns.violinplot(y="PetalLengthCm", x= "Species", data=data,  orient='v' , ax=axes[1, 0])
axes[1,1].set_title("Distribution of Petal Width")
sns.violinplot(y="PetalWidthCm", x= "Species", data=data,  orient='v' , ax=axes[1, 1])
plt.show()


# ### Conclusions:
# 
# The kernel density in the Violin plots helps us understand the full distribution of the data in terms of density.

# 

# 

# In[ ]:





# In[ ]:





# In[ ]:




