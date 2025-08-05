import streamlit as st
import numpy as np
import pandas as pd
import joblib

st.set_page_config(layout="wide", page_title = "Iris Classification", page_icon = ":cherry_blossom:")

# load Model
model = joblib.load('model.joblib')

# Get Prediction
def get_predict(data:pd.DataFrame, model):
    """Get Prediction

    Args:
        data (pd.DataFrame): DataFrame
        model (_type_): Model Classifier
    """

    prediction = model.predict(data)
    predict_proba = model.predict_proba(data)

    map_label = {0: "Setosa",1: "Versicolor",2: "Virginica"}
    prediction_label = map(lambda x: map_label[x], list(prediction))

    

    return{
        "prediction": prediction,
        "prob": predict_proba,
        "label": list(prediction_label)
    }

st.title("Halo bapak ibu sekalian")
st.write('Selamat datang XD')

# Set Columns
left, right = st.columns(2, gap="medium")

# Sepal Input
left.subheader("Sepal Information")
sepal_length = left.number_input("Sepal Length", min_value=0.0, max_value=10.0, value=5.0, step=0.1)
sepal_width = left.number_input("Sepal Width", min_value=0.0, max_value=10.0, value=5.0, step=0.1)

# Petal Input
right.subheader("Petal Information")
petal_length = right.number_input("Petal Length", min_value=0.0, max_value=10.0, value=5.0, step=0.1)
petal_width = right.number_input("Petal Width", min_value=0.0, max_value=10.0, value=5.0, step=0.1)

predict = st.button("Predict", use_container_width=True)
if predict :
    df = pd.DataFrame(
        np.array([[sepal_length, sepal_width, petal_length, petal_width]]),
        columns=["sepal length (cm)", "sepal width (cm)", "petal length (cm)", "petal width (cm)"]
    )

    st.write(df)

    result = get_predict(df, model)

    label = result["label"][0]
    prediction = result["prediction"][0]
    prob = result["prob"][0][prediction]

    st.write(f"Your iris species is **{prob:.0%} {label}**")



