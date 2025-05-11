import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

st.title("Interactive ML Model Selection")

df = pd.read_csv("Cleaned_dataset.csv")
st.write("### Dataset Preview:")
st.write(df.head())

# Converting categorical columns to numerical
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = df[col].astype('category').cat.codes

# User selecting the target variable
target = st.selectbox("Select Target Variable:", df.columns)

features = st.multiselect("Select Features:", [col for col in df.columns if col != target])

if len(features) > 0:
    # User selecting the Model
    model_choice = st.selectbox("Select Model:", ["RandomForest", "SVM"])

    if model_choice:
        if st.button("Run Model"):
            X = df[features]
            y = df[target]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            if model_choice == "RandomForest":
                model = RandomForestClassifier(n_estimators=100, random_state=42)
            else:
                model = SVC(kernel='linear', random_state=42)

            model.fit(X_train, y_train)
            predictions = model.predict(X_test)

            st.write(f"### {model_choice} Classifier Results:")
            st.write("**Accuracy:**", accuracy_score(y_test, predictions))
            st.text("Classification Report:")
            st.text(classification_report(y_test, predictions))
