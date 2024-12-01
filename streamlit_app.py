import joblib
import streamlit as st

# Available models and vectorizers
model_to_vectorizer = {
    "AdaBoostClassifier": ["CountVectorizer", "HashingVectorizer", "TfidfVectorizer"],
    "BaggingClassifier": ["CountVectorizer", "HashingVectorizer", "TfidfVectorizer"],
    "BernoulliNB": ["CountVectorizer", "HashingVectorizer", "TfidfVectorizer"],
    "CalibratedClassifierCV": ["CountVectorizer", "HashingVectorizer", "TfidfVectorizer"],
    "DecisionTreeClassifier": ["CountVectorizer", "HashingVectorizer", "TfidfVectorizer"],
    "DummyClassifier": ["CountVectorizer", "HashingVectorizer", "TfidfVectorizer"],
    "ExtraTreesClassifier": ["CountVectorizer", "HashingVectorizer", "TfidfVectorizer"],
    "GradientBoostingClassifier": ["CountVectorizer", "HashingVectorizer", "TfidfVectorizer"],
    "KNeighborsClassifier": ["CountVectorizer", "HashingVectorizer", "TfidfVectorizer"],
    "OneVsRestClassifier": ["CountVectorizer", "HashingVectorizer", "TfidfVectorizer"],
    "PassiveAggressiveClassifier": ["CountVectorizer", "HashingVectorizer", "TfidfVectorizer"],
    "RandomForestClassifier": ["CountVectorizer", "HashingVectorizer", "TfidfVectorizer"],
    "RidgeClassifierCV": ["CountVectorizer", "HashingVectorizer", "TfidfVectorizer"],
    "RidgeClassifier": ["CountVectorizer", "HashingVectorizer", "TfidfVectorizer"],
    "SGDClassifier": ["CountVectorizer", "HashingVectorizer", "TfidfVectorizer"]
}

# Available models and vectorizers for user selection
available_models = list(model_to_vectorizer.keys())
available_vectorizers = ["CountVectorizer", "HashingVectorizer", "TfidfVectorizer"]

# User selects a model and vectorizer
selected_model = st.selectbox("Select Model", available_models)
selected_vectorizer = st.selectbox("Select Vectorizer", model_to_vectorizer[selected_model])

# Check if the selected model and vectorizer combination exists
model_path = f"models/{selected_model}_{selected_vectorizer}.joblib"
vectorizer_path = f"models/{selected_vectorizer}.joblib"

# Load the model and vectorizer if they exist
try:
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)

    # User input for prediction
    user_input = st.text_input("Enter text for prediction:")

    if user_input:
        # Vectorize the input text
        input_vector = vectorizer.transform([user_input])

        # Make the prediction
        prediction = model.predict(input_vector)

        # Show the result
        st.write(f"Prediction: {prediction[0]}")

except FileNotFoundError:
    st.error(f"Error: The model or vectorizer file for {selected_model} with {selected_vectorizer} was not found.")
