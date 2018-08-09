def timefunc(func):
	def f(*args, **kwargs):
		from time import time
		start = time()
		rv = func(*args, **kwargs)
		finish = time()
		print('Run time is.', finish - start)
		return rv
	return f
# HIGHER ORDER DECORATOR
def ntimes(n):
	def inner(f):
		def wrapper(*args, **kwargs):
			for _ in range(n):
				print('running {.__name__}'.format(f))
				rv = f(*args, **kwargs)
			return rv
		return wrapper
	return inner

class Auth(object):
	def json_auth(self):
		import tweepy
		access_token = "access token here"
		access_token_secret = "acces token here"
		consumer_key = "acces token here"
		consumer_secret = "acces token here"
		auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
		auth.set_access_token(access_token, access_token_secret)
		api = tweepy.API(auth, parser=tweepy.parsers.JSONParser(),wait_on_rate_limit=True,wait_on_rate_limit_notify=True)
		return api		
	def auth(self):
		import tweepy
		access_token = "acces token here"
		access_token_secret = "acces token here"
		consumer_key = "acces token here"
		consumer_secret = "acces token here"
		auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
		auth.set_access_token(access_token, access_token_secret)
		api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)
		return api

class PullTwitterData(object):
	def __init__(self, target_id=None):
		self.api = Auth().auth()
		self.json_api = Auth().json_auth()
		self.target_id = target_id

	def get_twitter_client_api(self):
		return self.api
	def get_twitter_client_api_json(self):
		return self.json_api
## TIMELINE METHODS ##
	def my_home_timeline(self, query_size=None):
		from tweepy import Cursor
		home_timeline_tweets = []; print('INITIALIZING LOOP NOW')
		for tweet in Cursor(self.api.home_timeline, id=self.target_id).items(query_size):
			home_timeline_tweets.append(tweet); print('APPENDING DATA NOW')
		return home_timeline_tweets
	def target_statuses_lookup(self):
		results = []; print('INITIALIZING STATUS LOOP NOW')
		for status in self.api.statuses_lookup(id=self.target_id):
			results.append(status); print('APPENDING USER STATUS NOW')
		return results
	def target_user_timeline(self, query_size=None):
		from tweepy import Cursor
		tweets = []; print('INITIALIZING LOOP NOW')
		for tweet in Cursor(self.api.user_timeline, id=self.target_id).items(query_size):
			tweets.append(tweet); print('APPENDING DATA NOW')
		return tweets
## USER METHODS ##
	def target_get_user(self, query_size=None):
		from tweepy import Cursor
		results = []; print('INITIALIZING LOOP NOW')
		for userid in tweepy.Cursor(self.api.get_user, user_id=target_id).items(query_size):
			results.append(userid); print('APPENDING USERID DATA NOW')
		return results
	def target_followers(self, query_size=None):
		from tweepy import Cursor
		results = []; print('INITIALIZING LOOP NOW')
		for follower in tweepy.Cursor(self.api.followers,user_id=target_id).items(query_size):
			results.append(follower)
		return results
	## FRIENDSHIP METHODS ##
# FRIENDS_IDS
	def friend_ids(self, query_size=None):
		from tweepy import Cursor
		results = []; print('APPENDING FRIEND IDS NOW')
		for friends in tweepy.Cursor(self.api.friends_ids, user_id=user).items(query_size):
			results.append(friends)
		return results
# FOLLOWERS IDS
	def follower_ids(self, query_size=None):
		from tweepy import Cursor
		results = []; print('APPENDING FOLLOWER IDS COMPLETE')
		for friends in tweepy.Cursor(self.api.followers_ids, user_id=user).pages(query_size):
			results.append(friends); print('APPENDING FOLLOWER IDS NOW')
		return results
# WTF IS THIS
	# def friends_tweets(self, query_size=None):
	# 	from tweepy import Cursor
	# 	friend_list = []; print('APPENDING FRIENDS COMPLETE')
	# 	for friend in Cursor(self.api.friends, id=self.target_id).items(query_size):
	# 		friend_list.append(friend); print('APPENDING DATA NOW')
	# 	return friend_list
	## HELP METHODS ##
# SEARCH
	def twitter_search(self, query_size):
		from tweepy import Cursor
		results = []; print('INITIALIZING LOOP NOW')
		for tweet in tweepy.Cursor(self.api.search, q=self.target_id, show_user=True).items(query_size):
			results.append(tweet); print('APPENDING TWITTER SEARCH NOW', len(results))
		return results
	## TREND METHODS ##
# TRENDS_AVAILABLE
	# def twitter_trends(self):
	# 	import tweepy; from flatten_json import flatten; import pandas as pd
	# 	trends = {}; print('INITIALIZING LOOP NOW')
	# 	for trend in self.json_api.trends_available():
	# 		# trend1 = [flatten(d) for d in trend]
	# 		trends.append(trend1); print('APPENDING TRENDS NOW')
	# 	df = pd.DataFrame(trends)
		# df = pd.DataFrame(data=[i.parentid for i in trends], columns=['parentid'])
		# df['woeid'] = np.array([i.woeid for i in trends])
		# df['url'] = np.array([i.url for i in trends])
		# df['name'] = np.array([i.name for i in trends])
		# df['countryCode'] = np.array([i.countryCode for i in trends])
		# df['country'] = np.array([i.country for i in trends])
		# return df

class AnalyzeTweets(object):
	def count_punctuation(self, tweets):
		import string
		count = sum([1 for char in tweets if char in string.punctuation])
		return round(count/(len(tweets) - tweets.count(" ")), 3)*100

	def remove_punctuation(self, tweets):
		import string
		return "".join([word for word in tweets if word not in string.punctuation])

	def preprocessed_tweets(self, tweets):
		import string, nltk, re
		stopwords = nltk.corpus.stopwords.words('english')
		text = "".join([word for word in tweets if word not in string.punctuation])
		tokens = re.split(r'\W+', text)
		text = [word for word in tokens if word not in stopwords]
		return str(text)

	def analyze_sentiment(self, tweets):
		from textblob import TextBlob
		analysis = TextBlob(self.remove_punctuation(tweets))
		if analysis.sentiment.polarity > 0:
			return 1
		elif analysis.sentiment.polarity == 0:
			return 0
		else:
			return -1

	def processed_tweets(self, tweets):
		import pandas as pd; import numpy as np
		df = pd.DataFrame(data=[tweet.text for tweet in tweets], columns=['Tweets'])
		df['Amount_of_Punctuation_in_Twwet'] = df['Tweets'].apply(lambda x: self.count_punctuation(x))
		df['ML_Preprocessed'] = df['Tweets'].apply(lambda x: self.preprocessed_tweets(x.lower()))
		df['Tweet Length'] = df['Tweets'].apply(lambda x: len(x) - x.count(" "))
		df['stripped_punc'] = df['Tweets'].apply(lambda x: self.remove_punctuation(x))
		df['Sentiment'] = df['stripped_punc'].apply(lambda x: self.analyze_sentiment(x))
		df['ID'] = np.array([tweet.id for tweet in tweets])
		df['user_follower_count'] = np.array([tweet.user.followers_count for tweet in tweets])
		df['user_friends_count'] = np.array([tweet.user.friends_count for tweet in tweets])
		df['user_name'] = np.array([tweet.user.name for tweet in tweets])
		df['created_at'] = np.array([tweet.created_at for tweet in tweets])
		df['user_screen_name'] = np.array([tweet.user.screen_name for tweet in tweets])
		df['user_desc'] = np.array([tweet.user.description for tweet in tweets])
		df['Source'] = np.array([tweet.source for tweet in tweets])
		df['favorite_count'] = np.array([tweet.favorite_count for tweet in tweets])
		df['favorited'] = np.array([tweet.favorited for tweet in tweets])
		df['retweet_count'] = np.array([tweet.retweet_count for tweet in tweets])
		df['retweeted'] = np.array([tweet.retweeted for tweet in tweets])
		# df['retweets'] = np.array([tweet.retweets for tweet in tweets])
		# df['author'] = np.array([tweet.author for tweet in tweets])
		df['geo'] = np.array([tweet.geo for tweet in tweets])
		df['coordinates'] = np.array([tweet.coordinates for tweet in tweets])
		df['place'] = np.array([tweet.place for tweet in tweets])
		# df['entities'] = np.array([tweet.entities for tweet in tweets])
		df['source_url'] = np.array([tweet.source_url for tweet in tweets])
		df['user_location'] = np.array([tweet.user.location for tweet in tweets])
		df['user_timezone'] = np.array([tweet.user.time_zone for tweet in tweets])
		del df['stripped_punc']
		return df

	def raw_tweets(self, tweets):
		import pandas as pd; import numpy as np
		df = pd.DataFrame(data=[tweet.text for tweet in tweets], columns=['Tweets'])
		df['ID'] = np.array([tweet.id for tweet in tweets])
		df['user_follower_count'] = np.array([tweet.user.followers_count for tweet in tweets])
		df['user_friends_count'] = np.array([tweet.user.friends_count for tweet in tweets])
		df['user_name'] = np.array([tweet.user.name for tweet in tweets])
		df['created_at'] = np.array([tweet.created_at for tweet in tweets])
		df['user_screen_name'] = np.array([tweet.user.screen_name for tweet in tweets])
		df['user_desc'] = np.array([tweet.user.description for tweet in tweets])
		df['Source'] = np.array([tweet.source for tweet in tweets])
		df['favorite_count'] = np.array([tweet.favorite_count for tweet in tweets])
		df['favorited'] = np.array([tweet.favorited for tweet in tweets])
		df['retweet_count'] = np.array([tweet.retweet_count for tweet in tweets])
		df['retweeted'] = np.array([tweet.retweeted for tweet in tweets])
		# df['retweets'] = np.array([tweet.retweets for tweet in tweets])
		# df['author'] = np.array([tweet.author for tweet in tweets])
		df['geo'] = np.array([tweet.geo for tweet in tweets])
		df['coordinates'] = np.array([tweet.coordinates for tweet in tweets])
		df['place'] = np.array([tweet.place for tweet in tweets])
		# df['entities'] = np.array([tweet.entities for tweet in tweets])
		df['source_url'] = np.array([tweet.source_url for tweet in tweets])
		df['user_location'] = np.array([tweet.user.location for tweet in tweets])
		df['user_timezone'] = np.array([tweet.user.time_zone for tweet in tweets])
		# del df['Tweets']
		return df















