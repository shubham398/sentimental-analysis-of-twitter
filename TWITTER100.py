# General:
import tweepy           # To consume Twitter's API
import pandas as pd    # To handle data
import numpy as np      # For number computing
from textblob import TextBlob
import re
import csv

# For plotting and visualization:
##%matlotlib inline
from IPython.display import display
import matplotlib.animation as animation
import matplotlib.pyplot as plt
##%matlotlib inline
##import seaborn as sns

plt.style.use('ggplot')
fig=plt.figure()
ax1=fig.add_subplot(1,1,1)
ax2=fig.add_subplot(1,1,1)
consumer_key = ''
consumer_secret = ''
access_token = ''
access_token_secret = ''


# API's setup:
def twitter_setup():
    """
    Utility function to setup the Twitter's API
    with our access keys provided.
    """
    # Authentication and access using keys:
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)

    # Return API with authentication:
    api = tweepy.API(auth)
    return api

extractor = twitter_setup()

# We create a tweet list as follows:
tweets = extractor.user_timeline(screen_name="jntsngh3", count=200)
print("Number of tweets extracted: {}.\n".format(len(tweets)))

# We print the most recent 5 tweets:
print("5 recent tweets:\n")
for tweet in tweets[:5]:
    print(tweet.text)
    print()

data = pd.DataFrame(data=[tweet.text for tweet in tweets], columns=['Tweets'])

# We display the first 10 elements of the dataframe:
display(data.head(10))

print(dir(tweets[0]))

# We print info from the first tweet:
print(tweets[0].id)
print(tweets[0].created_at)
print(tweets[0].source)
print(tweets[0].favorite_count)
print(tweets[0].retweet_count)
print(tweets[0].geo)
print(tweets[0].coordinates)
print(tweets[0].entities)

# We add relevant data:
data['len']  = np.array([len(tweet.text) for tweet in tweets])
data['ID']   = np.array([tweet.id for tweet in tweets])
data['Date'] = np.array([tweet.created_at for tweet in tweets])
data['Source'] = np.array([tweet.source for tweet in tweets])
data['Likes']  = np.array([tweet.favorite_count for tweet in tweets])
data['RTs']    = np.array([tweet.retweet_count for tweet in tweets])

# We extract the mean of lenghts:
mean = np.mean(data['len'])

print("The lenght's average in tweets: {}".format(mean))

# We extract the tweet with more FAVs and more RTs:

fav_max = np.max(data['Likes'])
rt_max  = np.max(data['RTs'])

fav = data[data.Likes == fav_max].index[0]
rt  = data[data.RTs == rt_max].index[0]

# Max FAVs:
print("The tweet with more likes is: \n{}".format(data['Tweets'][fav]))
print("Number of likes: {}".format(fav_max))
print("{} characters.\n".format(data['len'][fav]))

# Max RTs:
print("The tweet with more retweets is: \n{}".format(data['Tweets'][rt]))
print("Number of retweets: {}".format(rt_max))
print("{} characters.\n".format(data['len'][rt]))

# We create time series for data:

tlen = pd.Series(data=data['len'].values, index=data['Date'])
tfav = pd.Series(data=data['Likes'].values, index=data['Date'])
tret = pd.Series(data=data['RTs'].values, index=data['Date'])

# Lenghts along time:
ax1.plot(tlen)
plt.show()
ax2.plot(tfav)
plt.show()

# Likes vs retweets visualization:
##tfav.plot(figsize=(16,4), label="Likes", legend=True)
##tret.plot(figsize=(16,4), label="Retweets", legend=True)
####tlen.show()
##tfav.show()
##tret.show()

##saveFile=open('twit2.html','a')
##str1=str(tweets).encode(encoding='utf-8',errors='strict')
##str2=str(str1)
##saveFile.write(str2)
##saveFile.write('\n')
##saveFile.close()
outtweets = [[tweet.id, tweet.created_at, tweet.text.encode("utf-8"),tweet.source,tweet.favorite_count,tweet.retweet_count] for tweet in tweets]
with open('%s_tweets.csv' % 'jntsngh3', 'w') as f: 
    writer = csv.writer(f)
    writer.writerow(["id","created_at","text","source","likes","RTs"])
    writer.writerows(outtweets)
pass 
                

def clean_tweet(tweet):
    '''
    Utility function to clean the text in a tweet by removing
    links and special characters using regex.
    '''
    return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split())

def analize_sentiment(tweet):
    '''
    Utility function to classify the polarity of a tweet
    using textblob.
    '''
    analysis = TextBlob(clean_tweet(tweet))
    if analysis.sentiment.polarity > 0:
        return 1
    elif analysis.sentiment.polarity == 0:
        return 0
    else:
        return -1

# We create a column with the result of the analysis:
data['SA'] = np.array([ analize_sentiment(tweet) for tweet in data['Tweets'] ])

# We display the updated dataframe with the new column:
display(data.head(10))

# We construct lists with classified tweets:

pos_tweets = [ tweet for index, tweet in enumerate(data['Tweets']) if data['SA'][index] > 0]
neu_tweets = [ tweet for index, tweet in enumerate(data['Tweets']) if data['SA'][index] == 0]
neg_tweets = [ tweet for index, tweet in enumerate(data['Tweets']) if data['SA'][index] < 0]

# We print percentages:

print("Percentage of positive tweets: {}%".format(len(pos_tweets)*100/len(data['Tweets'])))
print("Percentage of neutral tweets: {}%".format(len(neu_tweets)*100/len(data['Tweets'])))
print("Percentage de negative tweets: {}%".format(len(neg_tweets)*100/len(data['Tweets'])))

##fig, axes = plt.subplots(nrows=2, ncols=1)
##fig.set_size_inches(15, 5)
##plt.subplot(211).axes.get_xaxis().set_visible(False)
##tlen.plot(kind='line', title='Tesla Sentiment')
##plt.subplot(212)
##tfav.plot(kind='area', title='Volume')
