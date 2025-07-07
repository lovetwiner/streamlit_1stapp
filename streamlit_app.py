import streamlit as st
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer

# Download NLTK resources if not already downloaded
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# Initialize NLTK objects
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

# Define preprocessing functions
def clean_text(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    return text

def remove_stopwords(text):
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return words

def lem_word(word_list):
    return [lemmatizer.lemmatize(w) for w in word_list]

def stem_word(word_list):
    return [stemmer.stem(word) for word in word_list]

def combine_text(list_of_text):
    return ' '.join(list_of_text)

def preprocess_text(text):
    cleaned_text = clean_text(text)
    words_without_stopwords = remove_stopwords(cleaned_text)
    lemmatized_words = lem_word(words_without_stopwords)
    stemmed_words = stem_word(lemmatized_words)
    processed_text = combine_text(stemmed_words)
    return processed_text

# Load the vectorizers and model
try:
    with open("mnb_model.pkl", "rb") as f:
        mnb_model = pickle.load(f)
    with open("count_vectorizer.pkl", "rb") as f:
        count_vectorizer = pickle.load(f)
except FileNotFoundError:
    st.error("Model files not found. Please ensure 'count_vectorizer.pkl' and 'mnb_model.pkl' are in the same directory.")
    st.stop()

st.title('Tweet Sentiment Analysis (Count Vectorizer)')
st.write("Enter a tweet to analyze its sentiment using the Count Vectorizer model.")

user_tweet = st.text_area("Enter your tweet here:")

if st.button("Analyze"):
    if user_tweet:
        processed_tweet = preprocess_text(user_tweet)
        count_vec_tweet = count_vectorizer.transform([processed_tweet])
        prediction = mnb_model.predict(count_vec_tweet)
        sentiment = "Positive" if prediction[0] == 0 else "Negative"
        st.success(f"Count Vectorizer Prediction: {sentiment}")
    else:
        st.warning("Please enter a tweet to analyze.")
