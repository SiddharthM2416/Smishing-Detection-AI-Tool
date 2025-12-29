'''import streamlit as st
import pickle

# load model + vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

st.title("Spam vs Ham Classifier")

user_input = st.text_area("Enter a message:")

if st.button("Predict"):
    if user_input.strip() != "":
        x = vectorizer.transform([user_input])
        prediction = model.predict(x)[0]

        if prediction == 1:
            st.error("🚨 This message is SPAM!")
        else:
            st.success("✅ This message is HAM (not spam).")
    else:
        st.warning("Please type something first.")'''
        
import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# 1. Initialize NLTK resources
ps = PorterStemmer()

# Download necessary NLTK data (handles errors if already downloaded)
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

def transform_text(text):
    """
    Same preprocessing function used in the Jupyter Notebook
    """
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# 2. Load model + vectorizer
try:
    model = pickle.load(open("model.pkl", "rb"))
    vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
except FileNotFoundError:
    st.error("Model files not found. Please ensure 'model.pkl' and 'vectorizer.pkl' are in the same folder.")
    st.stop()

st.title("Spam vs Ham Classifier")

user_input = st.text_area("Enter a message:")

if st.button("Predict"):
    if user_input.strip() != "":
        # 3. FIX: Preprocess the input BEFORE vectorizing
        transformed_sms = transform_text(user_input)
        
        # Vectorize the transformed text
        x = vectorizer.transform([transformed_sms])
        
        # Predict
        prediction = model.predict(x)[0]

        if prediction == 1:
            st.error("🚨 This message is SPAM!")
        else:
            st.success("✅ This message is HAM (not spam).")
    else:
        st.warning("Please type something first.")