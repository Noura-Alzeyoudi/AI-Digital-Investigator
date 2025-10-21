import streamlit as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

st.set_page_config(page_title="AI Digital Investigator", page_icon="🕵️‍♀️")

st.title("🕵️‍♀️ AI Digital Investigator")
st.write("Type a message or email content below to check if it’s **suspicious** or **safe**.")

emails = [
    "You won a million dollars! Click here to claim.",
    "Please find the attached report for today’s meeting.",
    "Urgent: Your password will expire in 24 hours.",
    "Here is the invoice for your recent purchase.",
    "Congratulations! You have been selected for a prize."
]
labels = [1, 0, 1, 0, 1]

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(emails)
model = MultinomialNB()
model.fit(X, labels)

user_input = st.text_area("✉️ Enter text to analyze:")
if st.button("Analyze"):
    if user_input.strip() == "":
        st.warning("Please enter some text first.")
    else:
        test_X = vectorizer.transform([user_input])
        prediction = model.predict(test_X)[0]
        if prediction == 1:
            st.error("⚠️ This message seems **suspicious**.")
        else:
            st.success("✅ This message seems **safe**.")
