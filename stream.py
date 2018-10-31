from tweepy import Stream
from tweepy import OAuthHandler
from tweepy.streaming import StreamListener
import json
import sentiment_mod as s



#consumer key, consumer secret, access token, access secret.
ckey="NhFVzmiUPRI8lN93IYGwd1Izp"
csecret="sMBQkKdYYBRAoTJ6OeNG4f0GEcKdCr6sfo3GspXjdlTX8j5Nw4"
atoken="215169252-PeG79rEKsYvcEvljvVGlwcIvaOgHQlDA2N6hvc12"
asecret="GTgxYfscksPIXwGaveqqKqCdVhTg4SFAf98aYYQaVuIZp"

class listener(StreamListener):

    def on_data(self, data):
        all_data = json.loads(data)

        tweet = all_data["text"]
        sentiment_value,confidence = s.sentiment(tweet)
        print(tweet, sentiment_value, confidence)

        if confidence*100 >= 80:
            output = open("twitter-out.txt","a")
            output.write(sentiment_value)
            output.write('\n')
            output.close()

        return True

    def on_error(self, status):
        print(status)

auth = OAuthHandler(ckey, csecret)
auth.set_access_token(atoken, asecret)

twitterStream = Stream(auth, listener())
twitterStream.filter(track=["love"])
