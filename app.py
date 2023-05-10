from flask import Flask, request
import joblib
from sklearn.preprocessing import Normalizer
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd
from scipy.sparse import hstack

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get the input string from the request object
        input_string = request.form['input_string']

        # Process the input string
        output_string = process_input_string(input_string)

        # Return the processed string as a response
        return output_string
    else:
        # Render the HTML template with a form to submit the input string
        return '''
            <form method="POST">
                <input type="text" name="input_string">
                <input type="submit" value="Submit">
            </form>
        '''

def process_input_string(input_string):
    # Load the saved model
    vectorizer = joblib.load('vectorizer_model1.pkl')
    model = joblib.load('xb_tfidf_model1.pkl')
    # Predicting sentiment of new tweets
    new_tweet = [input_string]

    sent = SentimentIntensityAnalyzer()
    neg = []
    neu = []
    pos = []
    com = []
    def update(k):
        neg.append(k["neg"])
        neu.append(k["neu"])
        pos.append(k["pos"])
        com.append(k["compound"])
        
    from tqdm import tqdm
    for i in tqdm(new_tweet):
        update(sent.polarity_scores(i))

    # create a dictionary from the lists
    data = {'OriginalTweet': new_tweet, 'negative': neg, 'neutral': neu, 'positive': pos, 'compound': com}

    # create a DataFrame from the dictionary
    df_future_unseen_data = pd.DataFrame(data)

    X_te_tweet_tfidf_future = vectorizer.transform(df_future_unseen_data['OriginalTweet'])

    normalizer = Normalizer()
    normalizer.fit(df_future_unseen_data['negative'].values.reshape(-1,1))
    X_te_neg_future = normalizer.transform(df_future_unseen_data['negative'].values.reshape(-1,1))

    normalizer = Normalizer()
    normalizer.fit(df_future_unseen_data['neutral'].values.reshape(-1,1))
    X_te_neu_future = normalizer.transform(df_future_unseen_data['neutral'].values.reshape(-1,1))

    normalizer = Normalizer()
    normalizer.fit(df_future_unseen_data['positive'].values.reshape(-1,1))
    X_te_pos_future = normalizer.transform(df_future_unseen_data['positive'].values.reshape(-1,1))

    normalizer = Normalizer()
    normalizer.fit(df_future_unseen_data['compound'].values.reshape(-1,1))
    X_te_com_future = normalizer.transform(df_future_unseen_data['compound'].values.reshape(-1,1))

    X_te_future = hstack((X_te_tweet_tfidf_future, X_te_neg_future, X_te_neu_future, X_te_pos_future, X_te_com_future)).tocsr()

    sentiment = model.predict(X_te_future)
    # print('Sentiment:', sentiment)

    if sentiment[0] == 0:
        s = "Neutral"
    elif sentiment[0] == 1:
        s = "Positive"
    elif sentiment[0] == 2:
        s = "Extremely Negative"
    elif sentiment[0] == 3:
        s = "Negative"
    elif sentiment[0] == 4:
        s = "Extremely positive"
    else:
        pass
    # Process the input string and return the result
    return s.upper()

if __name__ == "__main__":
    app.run(debug=True)
