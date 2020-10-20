from numpy.lib.function_base import vectorize
import pandas as pd
from sklearn.model_selection import train_test_split
import streamlit as st
from PIL import Image
from time import sleep


# @st.cache
def get_model():
    # df = pd.read_json("fake_news_reddit_cikm20.json")

    # Olenainen koodi lataamiseen
    import joblib
    vectorizer = joblib.load("tfidf_vectorizer.joblib")
    # tfidf_vectors = joblib.load("tfidf_vectors.joblib")
    model = joblib.load("tfidf-model-with-all-data-balanced-2.joblib")

    return vectorizer, model
    # X = tfidf_vectors
    # y = df["label"]
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)

def main():

    imageLocation = st.empty()
    image1 = Image.open('terminator1.jpg')
    image2 = Image.open('terminator2.jpg')
    image3 = Image.open('terminator3.jpg')
    imageLocation.image(image1, width=500)

    vectorizer, model = get_model()
    
    user_input = st.text_area(label="Paste article body", height=350)

    if st.button("Inspect news article"):

        document_to_predict = user_input
        document_to_predict_as_tfidf_vector = vectorizer.transform([document_to_predict])

        result = model.predict(document_to_predict_as_tfidf_vector)
        if result[0] == 1:
            imageLocation.image(image2, width=500)
            st.write("FAKE!")
            sleep(3)
            imageLocation.image(image3, width=500)
        else:
            st.write("LEGIT!")

if __name__ == "__main__":
    main()




