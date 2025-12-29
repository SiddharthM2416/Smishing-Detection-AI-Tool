import streamlit as st
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
        st.warning("Please type something first.")