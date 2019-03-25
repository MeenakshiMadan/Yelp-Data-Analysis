##Importing all the neccessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb

##Reading in the data for the Business 
data_business=pd.read_csv("yelp_business.csv")
data_business.head()

##Dropping not needed columns
data_business=data_business.drop(['business_id','neighborhood'], axis=1)
##Checking missing Values
data_business.isnull().any()
##printing the number of null values in each column
print(data_business.isnull().sum())

##Since there is only 1 missing reocrd in state, city , longitude , latitude 
# we can delete that 
##for the postal code too, we cant immitate a postal code so droppin missing value
data_business=data_business.dropna(subset=['city','state','longitude','latitude','postal_code'])
#checking the data after handling missing values
print(data_business.isnull().sum())

###Getting the overall distribution of the star rating for all the business
x=data_business['stars'].value_counts()
x=x.sort_index()
###Plot
plt.figure(figsize=(10,4))
ax=sb.barplot(x.index,x.values,alpha=0.8)
plt.title("Star Rating Distribution")
plt.xlabel("Rating", fontsize=12)
plt.ylabel("Number of Businesses", fontsize=12)

##Adding text labels to the plot
plt.show()

###Finding the most popular business in Categories
data_categories=' '.join(data_business['categories'])
category=pd.DataFrame(data_categories.split(';'),columns=['category'])
x=category.category.value_counts()


###Ploting the chart
plt.figure(figsize=(18,6))
axis_1 = sb.barplot(x.index, x.values, alpha=0.8)
plt.title("Top categories",fontsize=25)
locs, labels = plt.xticks()
plt.setp(labels, rotation=80)
plt.xlabel('Category', fontsize=18)
plt.ylabel('# businesses', fontsize=18)
plt.show()

##Filtering the data for only Restaurants
data_business=data_business[data_business['categories'].str.match('Restaurants')]
data_business.head()
