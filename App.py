import streamlit as st
import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

def load_model():
    model = pickle.load(open("/Users/swayampatil/PycharmProjects/TRY003-FLASk/mcdmodel.pkl", "rb"))
    vectorizer = pickle.load(open("/Users/swayampatil/PycharmProjects/TRY003-FLASk/vectorizer.pkl", "rb"))
    return model, vectorizer

def preprocess_input(review, vectorizer):
    review_tfidf = vectorizer.transform([review])
    input_data = pd.DataFrame(review_tfidf.toarray(), columns=vectorizer.get_feature_names_out())
    return input_data

def predict_sentiment(review, model, vectorizer):
    input_data = preprocess_input(review, vectorizer)
    prediction = model.predict(input_data)[0]
    return str(prediction)

def main():
    model, vectorizer = load_model()

    # Custom theme colors
    primary_color = "#904949"
    background_color = "#f3ca1d"
    secondary_background_color = "#abc0d2"
    text_color = "#cb213e"
    font = "serif"

    # Set CSS styles
    css = f"""
        <style>
        body {{
            background-color: {background_color};
            color: {text_color};
            font-family: {font};
        }}
        .stButton button {{
            background-color: {primary_color};
            color: white;
        }}
        .stTextInput textarea {{
            background-color: {secondary_background_color} !important;
            color: {text_color} !important;
        }}
        </style>
    """
    st.markdown(css, unsafe_allow_html=True)

    st.sidebar.title("About")
    st.sidebar.info(
        "This web app performs sentiment analysis on reviews related to McDonald's. "
        "Enter your review in the text area and click the 'Predict' button to see the sentiment prediction."
    )
    st.sidebar.info(
        "I am an aspiring web developer with a passion for machine learning and web application development. I enjoy working with Streamlit and I'm eager to learn and explore new technologies."
    )

    st.title("McDonald's Sentiment Analysis")
    review = st.text_area("Enter your review", height=200)

    if st.button("Predict"):
        if not review:
            st.error("Error: No review provided")
        else:
            prediction = predict_sentiment(review, model, vectorizer)
            st.write(f"The sentiment is: {prediction}")
            if prediction == "Positive":
                st.write("😊")  # Positive emoticon
            elif prediction == "Negative":
                st.write("😔")  # Negative emoticon
            else:
                st.write("😐")  # Neutral emoticon

if __name__ == "__main__":
    main()

