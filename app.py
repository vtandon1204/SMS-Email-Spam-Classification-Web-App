import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()


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

tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

st.title("Email/SMS Spam Classifier")

input_sms = st.text_area("Enter the message")

if st.button('Predict'):

    # 1. preprocess
    transformed_sms = transform_text(input_sms)
    # 2. vectorize
    vector_input = tfidf.transform([transformed_sms])
    # 3. predict
    result = model.predict(vector_input)[0]
    # 4. Display
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")



# import streamlit as st
# import pickle
# import string
# from nltk.corpus import stopwords 
# import nltk
# from nltk.stem.porter import PorterStemmer

# nltk.download('punkt')

# ps = PorterStemmer()

# tfidf = pickle.load(open('vectorizer.pkl','rb'))
# model = pickle.load(open('model.pkl','rb'))

# def transform_text(text):
#     text = text.lower()
#     text = nltk.word_tokenize(text)

#     text = [word for word in text if word.isalnum()]

#     text = [word for word in text if word not in stopwords.words('english') and word not in string.punctuation]

#     text = [ps.stem(word) for word in text]

#     return " ".join(text)

# st.title("Email/SMS Spam Classifier")
# input_sms = st.text_area("Enter the message")

# if st.button("Predict"):
#     transformed_sms = transform_text(input_sms)
#     vector_input = tfidf.transform([transformed_sms])
#     result = model.predict(vector_input)[0]
#     if result == 1:
#         st.header("Spam")
#     else:
#         st.header("Not Spam")
