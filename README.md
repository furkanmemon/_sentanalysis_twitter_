# _sentanalysis_twitter_

'''run train.py first.
after this the sentiment_mod.py
later stream.py'''


train.py is for importing the data and sorting the data in the required format(part of speech)
training the different classifiers on the dataset.
all the trained classifiers are saved using pickle.


sentiment_mod.py accesses the saved classifiers and creates a new custom classifier combining
all the different classifiers used
 
sentiment_mod.py also has a sentiment function which takes in input, converts it into features and classifies sentiment.


stream.py is using the tweepy package to get data from twitter.

you need a developer account on twitter for this.

change the value of variable "track" on the last line of stream.py to choose subject for analysis. for eg:ferrari




 
