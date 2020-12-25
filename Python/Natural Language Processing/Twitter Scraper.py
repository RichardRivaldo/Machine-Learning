# Libraries

import tweepy
import pandas as pd

# Authentication

consumer_key = ""
consumer_secret = ""
access_token = ""
access_token_secret = ""

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth,wait_on_rate_limit=True)

query_list = []

count = 100000

tweets_list = []
    
for text in query_list:
    tweets = tweepy.Cursor(api.search, q = text).items(count)
    for tweet in tweets:
        tweets_list.append([tweet.id, tweet.created_at, tweet.user, tweet.text, tweet.favorite_count])
tweets_df = pd.DataFrame(tweets_list)