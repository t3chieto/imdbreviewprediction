import streamlit as st
from keras.models import load_model
from keras.datasets import imdb
from keras.preprocessing import sequence

model = load_model('model.keras')

words_index = imdb.get_word_index()

def preprocess_text(text):
    words = text.lower().split()
    encoded_text = [words_index.get(word,2)+3 for word in words]
    padded_review = sequence.pad_sequences([encoded_text],maxlen=5000)
    return padded_review

st.title("Movie Sentiment Prediction")

input = st.text_input('opinion')

process_input = preprocess_text(input)

prediction = model.predict(process_input)

sentiment = 'Positive' if prediction[0][0]>0.5 else 'Negative'

st.write(sentiment)
st.write(prediction[0][0])






