# Libraries

import tweepy
import pandas as pd

# Authentication

consumer_key = "joTldkRdQkxSDy4nqiSAwHAPk"
consumer_secret = "G9o0G2ahpX9YdiEjdQvO3WjwQnSkQfIfJ5uYHz4dVdIC1TAioO"
access_token = "1342376024898633728-XjWe3dA5XAxLmp4uAe3fCwvZkL8X15"
access_token_secret = "7wdpOMNyU3bB9RphV5aEt1GVa5bex56DK4wu7NpxkBS0Z"

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