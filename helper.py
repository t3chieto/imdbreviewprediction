from keras.datasets import imdb
from keras.preprocessing import sequence

words_index = imdb.get_word_index()

reverse_word_index = {value:key for key,value in words_index.items()}

def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i,'?') for i in encoded_review])

def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [words_index.get(word,2)+3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review],maxlen=5000)
    return padded_review