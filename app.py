import numpy as np
import streamlit as st
from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.models import load_model
from helper import preprocess_text

## Step: 1
# Load the IMDB dataset word index
word_index = imdb.get_word_index()

## Load the pre-trained model with relu activation
model = load_model('model.keras')

## Step: 2 - helper functions -> helper.py


## Step: 3 Prediction function
def predict_sentiment(review):
    preprocess_input = preprocess_text(review)
    prediction = model.predict(preprocess_input)
    sentiment = 'Positive' if prediction[0][0]>0.5 else 'Negative'
    return sentiment, prediction[0][0]


## Streamlit app
st.title("IMDB Movie Review Sentiment Analysis.")
st.write('Enter a movie review to classify it as positive or negative. ')


# User input
user_input = st.text_area('Movie Review')

if st.button('Classify'):
    preprocessed_input = preprocess_text(user_input)
    ## make prediction
    prediction = model.predict(preprocessed_input)
    sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'
    
    # Display the result
    st.write(f'Sentiment: {sentiment}')
    st.write(f'Prediction Score: {prediction[0][0]}')
else:
    st.write('Please enter a movie review.')

