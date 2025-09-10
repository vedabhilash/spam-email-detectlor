import streamlit as st
import pickle

@st.cache_resource
def load_model():
    with open("spam_classifier.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

st.title("📧 Spam Email/SMS Detector")
st.write("Type a message below and check if it's spam or not.")

user_input = st.text_area("Enter message text here:")

if st.button("Predict"):
    if not user_input.strip():
        st.warning("Please enter some text first.")
    else:
        prediction = model.predict([user_input])[0]
        if prediction == "spam":
            st.error("🚫 This message is SPAM")
        else:
            st.success("✅ This message is NOT spam")
