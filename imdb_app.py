import streamlit as st
import pandas as pd
import re
import string
import joblib

st.set_page_config(
    page_title="Movie Reviews Sentiment Analysis",
    page_icon="🎬",
    layout="centered"
)

@st.cache_resource
def load_model():
    model = joblib.load('movie_sentiment_model.pkl')
    vectorizer = joblib.load('movie_tfidf_vectorizer.pkl')
    return model, vectorizer

model, vectorizer = load_model()

def clean_text(text):
    text = text.lower()
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text

st.title("🎬 Sentiment Analysis on Movie Reviews")
st.write("Bu uygulama, girdiğiniz film yorumunun duygu durumunu (Pozitif veya Negatif) analiz eder.")

user_review = st.text_area("Film Yorumunuzu Girin:", "This movie was an absolute masterpiece! The acting and cinematography were breathtaking.")

if st.button("Duygu Analizi Yap"):
    if user_review.strip() == "":
        st.warning("Lütfen geçerli bir yorum metni girin.")
    else:
        cleaned = clean_text(user_review)
        vec = vectorizer.transform([cleaned])
        prediction = model.predict(vec)[0]
        probability = model.predict_proba(vec)[0]
        
        st.markdown("---")
        st.subheader("📊 Analiz Sonucu:")
        if prediction == 1:
            st.success(f"🟢 **POZİTİF YORUM** (Güven Skoru: %{probability[1]*100:.2f})")
        else:
            st.error(f"🔴 **NEGATİF YORUM** (Güven Skoru: %{probability[0]*100:.2f})")