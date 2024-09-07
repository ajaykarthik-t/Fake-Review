import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, LSTM, Dense, SpatialDropout1D

# Load the dataset with updated cache function
@st.cache_data
def load_data():
    df = pd.read_csv('fake reviews dataset.csv')
    df['label'] = df['label'].map({'CG': 1, 'OR': 0})
    return df

df = load_data()

# Define the feature (X) and target (y) variables
X = df['text_']  # The review text
y = df['label']  # The labels

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert the text data into TF-IDF features for Logistic Regression
vectorizer = TfidfVectorizer(max_features=3000)  # Reduced features
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Initialize and train the Logistic Regression model
logistic_model = LogisticRegression(max_iter=500)  # Reduced iterations
logistic_model.fit(X_train_tfidf, y_train)

# Calculate accuracy for Logistic Regression
logistic_pred = logistic_model.predict(X_test_tfidf)
logistic_accuracy = accuracy_score(y_test, logistic_pred)

# Prepare the data for LSTM
max_words = 3000  # Reduced words
max_len = 100
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(X_train)

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

X_train_pad = pad_sequences(X_train_seq, maxlen=max_len)
X_test_pad = pad_sequences(X_test_seq, maxlen=max_len)

# Try to load the LSTM model if it was previously saved
try:
    lstm_model = load_model('lstm_model.h5')
except:
    # Build and train the LSTM model if it was not found
    lstm_model = Sequential()
    lstm_model.add(Embedding(max_words, 100, input_length=max_len))
    lstm_model.add(SpatialDropout1D(0.2))
    lstm_model.add(LSTM(50, dropout=0.2, recurrent_dropout=0.2))  # Reduced LSTM units
    lstm_model.add(Dense(1, activation='sigmoid'))

    lstm_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    lstm_model.fit(X_train_pad, y_train, epochs=2, batch_size=32, validation_data=(X_test_pad, y_test))  # Reduced epochs
    lstm_model.save('lstm_model.h5')  # Save the trained model

# Calculate accuracy for LSTM
lstm_pred = lstm_model.predict(X_test_pad)
lstm_pred_binary = (lstm_pred > 0.5).astype(int)  # Convert probabilities to binary output
lstm_accuracy = accuracy_score(y_test, lstm_pred_binary)

# Streamlit UI
st.title("Review Classification")
st.write("Select the algorithm and enter a review below to classify whether it's fake (CG) or genuine (OR):")

# Display accuracy of both models
st.write(f"Logistic Regression Model Accuracy: {logistic_accuracy:.2f}")
st.write(f"LSTM Model Accuracy: {lstm_accuracy:.2f}")

# Algorithm selection
algorithm = st.selectbox("Choose the algorithm", ["Logistic Regression", "LSTM"])

# User input
user_review = st.text_area("Review Text", "")

def predict_with_logistic(review_text):
    review_tfidf = vectorizer.transform([review_text])
    prediction = logistic_model.predict(review_tfidf)
    label = 'Computer Generated' if prediction[0] == 1 else 'Original Review'
    return label

def predict_with_lstm(review_text):
    review_seq = tokenizer.texts_to_sequences([review_text])
    review_pad = pad_sequences(review_seq, maxlen=max_len)
    prediction = lstm_model.predict(review_pad)
    label = 'Computer Generated' if prediction[0][0] > 0.5 else 'Original Review'
    return label

if st.button("Classify"):
    if user_review:
        if algorithm == "Logistic Regression":
            result = predict_with_logistic(user_review)
        else:
            result = predict_with_lstm(user_review)
        st.write(f'The review is classified as: {result}')
    else:
        st.write("Please enter a review to classify.")
