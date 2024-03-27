import streamlit as st
from transformers import pipeline
from deep_translator import GoogleTranslator

def sentiment_analysis_multilingual(text):
    sentiment_pipeline = pipeline("sentiment-analysis", model= "distilbert/distilbert-base-uncased-finetuned-sst-2-english")
    result = sentiment_pipeline(text)
    label = result[0]['label']
    sentiment = "Positive" if label == "POSITIVE" else "Negative"
    return text, sentiment

def translate_text(text):
    translated_text = GoogleTranslator(source='auto', target='en').translate(text)
    return translated_text

def main():
    st.title("Sentiment Analysis Multilingual App")
    input_text = st.text_input("Enter text for sentiment analysis:")
    translate_button = st.checkbox("Translate text")
    if translate_button:
            translated_text = translate_text(input_text)
            st.markdown(f'Translated Text: {translated_text}')

    if st.button("Analyze Sentiment"):
        if input_text:
            if translate_button:
                input_text, sentiment = sentiment_analysis_multilingual(input_text)
            else:
                input_text, sentiment = sentiment_analysis_multilingual(input_text)
            st.markdown("Sentiment Result:")
            if sentiment == "Positive":
                st.success("Positive")
            else:
                st.error("Negative")
        else:
            st.warning("Please enter some text for analysis.")

if __name__ == "__main__":
    main()
