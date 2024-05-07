import streamlit as st
import pickle
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

def preprocess_text(text):
    # Lowercasing
    text = text.lower()

    # Tokenization
    tokens = word_tokenize(text)

    # Removing Punctuation
    tokens = [word for word in tokens if word.isalnum()]

    # Removing Stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    # Join tokens back into sentences
    processed_text = ' '.join(tokens)

    return processed_text

try:
    tfidf = pickle.load(open('vectorizer.pkl','rb'))
    model = pickle.load(open('model.pkl','rb'))
except FileNotFoundError:
    st.error("Model or vectorizer file not found. Please make sure the file paths are correct.")
else:
    st.title("Sentiment Analyzer")

    input_sms = st.text_area("Write a review here")

    if st.button('Predict'):
        if not input_sms.strip():
            st.warning("Please enter a review.")
        else:
            # 1. preprocess
            transformed_sms = preprocess_text(input_sms)
            # 2. vectorize
            vector_input = tfidf.transform([transformed_sms])
            # 3. predict
            result = model.predict(vector_input)
            # 4. Display
            sentiment = ["positive","negative"]
            st.header(sentiment[int(result)])