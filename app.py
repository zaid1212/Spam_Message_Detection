import streamlit as st
import nltk
from nltk.stem import PorterStemmer
import pickle
from nltk.corpus import stopwords
import string
import base64


st.image("BMS_logo.png")
ps = PorterStemmer()

st.primaryColor = "#F63366"
st.backgroundColor = "#FFFFFF"
st.secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
font = "sans serif"


def transform_text(text):
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


tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))
accuracy = str(pickle.load((open('accuracy.pkl','rb'))))
st.title('Email/SMS Spam Classifier')
st.subheader("Model: Random Forest")

input_sms = st.text_area("Enter the message")

if st.button('Predict'):

    transform_sms = transform_text(input_sms)
    vector_input = tfidf.transform([transform_sms])
    result = model.predict(vector_input)[0]

    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")


st.subheader("Made by: Madhumitha,Fardeen,Zaid")
st.subheader('Accuracy of the model is:')
st.subheader(accuracy)


def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )
add_bg_from_local('bmsit.jpg')
