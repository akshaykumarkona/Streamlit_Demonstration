import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import os

df = pd.read_csv("Cleaned_dataset.csv")

st.title("Interactive Machine Learning Model Trainer and Predictor")
st.write("### Dataset Preview")
st.dataframe(df.head())

# Preprocessing (Categorical to Numerical)
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = df[col].astype('category').cat.codes

X = df.drop('Survived', axis=1)
y = df['Survived']

# Model Selection by User
model_choice = st.selectbox("Select Machine Learning Model", ["Random Forest", "SVM"])

# Feature Selection by User
selected_features = st.multiselect("Select Features to Use for Training", X.columns.tolist())

if 'model_trained' not in st.session_state:
    st.session_state['model_trained'] = False
    st.session_state['trained_model_path'] = None

# Training Model
if model_choice and selected_features:
    if st.button("Train Model"):
        X_selected = X[selected_features]
        X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

        if model_choice == "Random Forest":
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            model = SVC(kernel='linear', probability=True, random_state=42)

        model.fit(X_train, y_train)

        # Saving the trained model as a pickle file
        model_filename = "trained_model.pkl"
        with open(model_filename, 'wb') as file:
            pickle.dump((model, selected_features), file)

        # Displaying training results
        predictions = model.predict(X_test)
        st.write("### Model Trained Successfully!")
        st.write(f"Accuracy: {accuracy_score(y_test, predictions):.4f}")
        st.write("Classification Report:")
        st.text(classification_report(y_test, predictions))

        st.success("Model saved as 'trained_model.pkl'.")
        st.session_state['model_trained'] = True
        st.session_state['trained_model_path'] = model_filename
else:
    st.warning("Please select a model and features to train.")

# Prediction using trained model
st.write("---")
st.header("Make Predictions with Trained Model")

# Checking if the model has been trained
if st.session_state['model_trained']:
    if st.session_state['trained_model_path'] and os.path.exists(st.session_state['trained_model_path']):
        with open(st.session_state['trained_model_path'], 'rb') as file:
            trained_model, trained_features = pickle.load(file)

        # Displaying selected features for input
        st.write("### Enter values for selected features:")
        user_input = {}
        for feature in trained_features:
            if feature == "Sex":
                user_input[feature] = st.selectbox(f"Select {feature}", ["Male (0)", "Female (1)"])
                user_input[feature] = 0 if user_input[feature] == "Male (0)" else 1
            else:
                user_input[feature] = st.number_input(f"Enter value for {feature}", value=0.0)
        if st.button("Make Prediction"):
            # Convert user input to a dataframe
            input_data = pd.DataFrame([list(user_input.values())], columns=trained_features)
            prediction = trained_model.predict(input_data)

            # Display prediction result
            if prediction[0] == 1:
                st.success("Prediction: Survived")
            else:
                st.error("Prediction: Not Survived")
    else:
        st.error("Trained model file not found. Please train the model again.")
else:
    st.warning("Please train the model first before making predictions.")
