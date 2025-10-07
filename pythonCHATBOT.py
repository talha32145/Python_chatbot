import streamlit as st
import pandas as pd
import numpy as np
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')

from nltk.tokenize import word_tokenize
# ---- Page Setup ----
st.set_page_config(page_title="Python Code Helper", page_icon="🤖", layout="wide")

# ---- Custom CSS ----
st.markdown("""
    <style>
        body {
            background-color: #0E1117;
            color: white;
        }
        .main {
            background-color: #0E1117;
        }
        .stTextInput, .stTextArea, .stButton>button {
            border-radius: 12px;
        }
        .user-bubble {
            background-color: #1E90FF;
            color: white;
            padding: 10px 15px;
            border-radius: 18px;
            margin: 5px 0;
            max-width: 75%;
            align-self: flex-end;
            text-align: right;
        }
        .bot-bubble {
            background-color: #2E2E2E;
            color: #E0E0E0;
            padding: 10px 15px;
            border-radius: 18px;
            margin: 5px 0;
            max-width: 75%;
            align-self: flex-start;
            text-align: left;
        }
        .chat-container {
            display: flex;
            flex-direction: column;
            gap: 10px;
            padding: 10px;
            background-color: #121418;
            border-radius: 16px;
            max-height: 70vh;
            overflow-y: auto;
        }
        .title {
            text-align: center;
            color: #00FFAA;
            font-size: 2rem;
            font-weight: bold;
            margin-bottom: 10px;
        }
        .footer {
            text-align: center;
            font-size: 0.9rem;
            color: #aaa;
            margin-top: 20px;
        }
    </style>
""", unsafe_allow_html=True)

# ---- Title ----
st.markdown("<div class='title'>🤖 Python Code Helper Chatbot</div>", unsafe_allow_html=True)

# ---- Load dataset ----
df = pd.read_csv("Python_codes.csv")
df["question"] = df["question"].replace("nan", np.nan)
df = df.dropna(subset=["question"])

# ---- Preprocessing ----
def preprocessing(text):
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words("english"))
    tokens = [word for word in tokens if word.lower() not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word, pos="v") for word in tokens]
    return " ".join(tokens)

df["clean_question"] = df["question"].apply(preprocessing)

# ---- TF-IDF Vectorizer ----
vectorizer = TfidfVectorizer()
x = vectorizer.fit_transform(df["clean_question"])

# ---- Chatbot Logic ----
def python_chatbot(user_input):
    user_input_clean = preprocessing(user_input)
    vector_input = vectorizer.transform([user_input_clean])
    similarity = cosine_similarity(x, vector_input)
    index = similarity.argmax()
    return df["code"].iloc[index]

# ---- Session State ----
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ---- Chat UI ----
st.markdown("<div class='chat-container'>", unsafe_allow_html=True)
for role, text in st.session_state.chat_history:
    if role == "You":
        st.markdown(f"<div class='user-bubble'>👤 {text}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='bot-bubble'>🤖 {text}</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# ---- Input Box ----
user_input = st.chat_input("How to print hello world...")

if user_input:
    if user_input.lower() in ["bye", "exit", "okay bye", "ok bye", "goodbye"]:
        bot_reply = "Thank you for chatting! See you soon 👋"
    elif user_input.lower() in ["hi sir","hi","hello sir","hey sir","hey"]:
        bot_reply = "Hi there! I'm your Python code assistant. How can I help you today?"
    elif user_input.lower() in ["thanks", "thank you"]:
        bot_reply = "You're welcome! 😊"
    else:
        bot_reply = python_chatbot(user_input)

    st.session_state.chat_history.append(("You", user_input))
    st.session_state.chat_history.append(("Bot", bot_reply))
    st.rerun()

# ---- Footer ----
st.markdown("<div class='footer'>Made with ❤️ by Talha</div>", unsafe_allow_html=True)
