import pandas as pd
import numpy as np
import matplotlib.pyplot as ml
import seaborn as sns
import nltk
from nltk.corpus import stopwords
from textblob import TextBlob
import string
from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

##Reading in the data for Yelp Reviews
data_reviews=pd.read_csv("yelp_review.csv")
data_reviews.head
data_reviews.shape
print(data_reviews.isnull().sum())

##Reading in the data for the Business from the combined file
data_business=pd.read_csv("combined_yelp_new.csv")
data_business.head()

df = pd.merge(data_reviews,data_business, how = 'inner', left_on = 'business_id', right_on = 'Business_id')
df.shape
df.head()

###seperating the reviews 
df_1_star= df[df['stars_x'] == 1]
df_1_star.head()
df_2_star= df[(df['stars_x'] > 1) & (df['stars_x'] < 3)]
df_2_star.head()
df_3_star= df[(df['stars_x'] >= 3) & (df['stars_x'] < 4)]
df_3_star.head()
df_4_star= df[(df['stars_x'] >= 4) & (df['stars_x'] < 5)]
df_4_star.head()
df_5_star= df[df['stars_x'] == 5]
df_5_star.head()

##Removing the stopwords from reviews 1 star
from nltk.corpus import stopwords
stop=stopwords.words('english')
df_1_star['text']=df_1_star['text'].apply(lambda x:" ".join(x.lower() for  x in str(x).split() if x not in stop))
df_1_star['text'].head()

##Removing the stopwords from reviews 2 star
from nltk.corpus import stopwords
stop=stopwords.words('english')
df_2_star['text']=df_2_star['text'].apply(lambda x:" ".join(x.lower() for  x in str(x).split() if x not in stop))
df_2_star['text'].head()

##Removing the stopwords from reviews 3 star
from nltk.corpus import stopwords
stop=stopwords.words('english')
df_3_star['text']=df_3_star['text'].apply(lambda x:" ".join(x.lower() for  x in str(x).split() if x not in stop))
df_3_star['text'].head()

##Removing the stopwords from reviews 4 star
from nltk.corpus import stopwords
stop=stopwords.words('english')
df_4_star['text']=df_4_star['text'].apply(lambda x:" ".join(x.lower() for  x in str(x).split() if x not in stop))
df_4_star['text'].head()

##Removing the stopwords from reviews 5 star
from nltk.corpus import stopwords
stop=stopwords.words('english')
df_5_star['text']=df_5_star['text'].apply(lambda x:" ".join(x.lower() for  x in str(x).split() if x not in stop))
df_5_star['text'].head()

#Removing unuseful characters from 1 star
df_1_star['text']=df_1_star['text'].str.replace('[^\w\s]','')
df_1_star['text'].head()

#Removing unuseful characters from 2 star
df_2_star['text']=df_2_star['text'].str.replace('[^\w\s]','')
df_2_star['text'].head()

#Removing unuseful characters from 3 star
df_3_star['text']=df_3_star['text'].str.replace('[^\w\s]','')
df_3_star['text'].head()

#Removing unuseful characters from 4 star
df_4_star['text']=df_4_star['text'].str.replace('[^\w\s]','')
df_4_star['text'].head()

#Removing unuseful characters from 4 star
df_5_star['text']=df_5_star['text'].str.replace('[^\w\s]','')
df_5_star['text'].head()

######preparing the wordcloud for Star 1 reviews
#creating the stopword list
stopwords=set(STOPWORDS)
stopwords.update(["eat","know","the","4","le","10","as","name","went","im","0","if","came","35","ive","ok"
                 ,"get","is","its","so","food","good","place","service"])
                 
text_wordcloud1=" ".join(keyword for keyword in df_1_star.text)
text_wordcloud2=" ".join(keyword for keyword in df_2_star.text)
text_wordcloud3=" ".join(keyword for keyword in df_3_star.text)
text_wordcloud4=" ".join(keyword for keyword in df_4_star.text)
text_wordcloud5=" ".join(keyword for keyword in df_5_star.text)

##creating the wordcloud for 1 star reviews
wordcloud1 = WordCloud(stopwords=stopwords,max_words=500,
                      background_color='white',width=3000,
                      collocations=False,min_font_size=10,
                      height=3000).generate(text_wordcloud1)
                      
###creating the wordcloud for 2 star reviews
wordcloud2 = WordCloud(stopwords=stopwords,max_words=500,
                      background_color='white',width=3000,
                      collocations=False,min_font_size=10,
                      height=3000).generate(text_wordcloud2)
                      
###creating the wordcloud for 3 star review
wordcloud3 = WordCloud(stopwords=stopwords,max_words=500,
                      background_color='white',width=3000,
                      collocations=False,min_font_size=10,
                      height=3000).generate(text_wordcloud3)
                      
###creating the wordcloud for 4 star reviews
wordcloud4 = WordCloud(stopwords=stopwords,max_words=500,
                      background_color='white',width=3000,
                      collocations=False,min_font_size=6,
                      height=3000).generate(text_wordcloud4)  

##creating the wordcloud for 5 star reviews
wordcloud5 = WordCloud(stopwords=stopwords,max_words=500,
                      background_color='white',width=3000,
                      collocations=False,min_font_size=6,
                      height=3000).generate(text_wordcloud5)

#display the generated wordcloud
ml.figure(1,figsize=(20, 20))
ml.imshow(wordcloud1, interpolation='bilinear')
ml.axis("off")
ml.show()

#display the generated wordcloud for 2 star review
ml.figure(1,figsize=(20, 20))
ml.imshow(wordcloud2, interpolation='bilinear')
ml.axis("off")
ml.show()

#display the generated wordcloud for 3 star review
ml.figure(1,figsize=(20, 20))
ml.imshow(wordcloud3, interpolation='bilinear')
ml.axis("off")
ml.show()

#display the generated wordcloud for 4 star review
ml.figure(1,figsize=(20, 20))
ml.imshow(wordcloud4, interpolation='bilinear')
ml.axis("off")
ml.show()

#display the generated wordcloud for 5 star review
ml.figure(1,figsize=(20, 20))
ml.imshow(wordcloud5, interpolation='bilinear')
ml.axis("off")
ml.show()

wordcloud1.to_file("1starreview.png")
wordcloud2.to_file("2starreview.png")
wordcloud3.to_file("3starreview.png")
wordcloud4.to_file("4starreview.png")
wordcloud4.to_file("5starreview.png")

df.drop(df[df.start_x=='t'].index , inplace =True)

##Finding the length of each review
df['lengthoftext']=df['text'].apply(len)
df.head()

##Visualizing the relationship between the length and the star that are given
new_plot = sns.FacetGrid(data=df, col='stars_x')
new_plot.map(ml.hist, 'lengthoftext', bins=50)

sns.boxplot(x='stars_x', y='lengthoftext', data=df)
df.groupby('stars_x').describe()

###Sentiment analysis
####defining the polarity for the sentiment
def sentiment(x):
    sentiment = TextBlob(x)
    return sentiment.sentiment.polarity

#####TRying to do it for the top 10 reviews 
bs_cnt = pd.DataFrame(df['Name'].value_counts()[:10])
bs_cnt

top_reviewd_bs = df.loc[df['Name'].isin(bs_cnt.index)]
top_reviewd_bs.shape

top_reviewd_bs['text_sentiment'] = top_reviewd_bs['text'].apply(sentiment)
top_reviewd_bs['sentiment'] = ''
top_reviewd_bs['sentiment'][top_reviewd_bs['text_sentiment'] > 0] = 'positive'
top_reviewd_bs['sentiment'][top_reviewd_bs['text_sentiment'] < 0] = 'negative'
top_reviewd_bs['sentiment'][top_reviewd_bs['text_sentiment'] == 0] = 'neutral'

ml.figure(figsize=(6,6))
ax = sns.countplot(top_reviewd_bs['sentiment'])
ml.title('Review Sentiments');

#####there are a lot more positive reviews than there are negative
