import streamlit as st
import numpy as np
from tensorflow.keras.datasets import fashion_mnist
from model.train import train_model

st.set_page_config(page_title="CNN Fashion Classifier", layout="wide")

st.title("Fashion Classification using CNN")

labels = [
    "T-shirt",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle Boot"
]


@st.cache_resource
def load_cnn():
    return train_model()


model = load_cnn()

(_, _), (X_test, y_test) = fashion_mnist.load_data()

X_eval = X_test.reshape(-1, 28, 28, 1) / 255.0

predictions = model.predict(X_eval[:1000], verbose=0)
pred_classes = np.argmax(predictions, axis=1)

accuracy = np.mean(pred_classes == y_test[:1000])

st.success(f"Model Accuracy: {accuracy:.4f}")

index = st.slider("Select Test Image", 0, len(X_test)-1, 0)

image = X_test[index]

st.image(
    image,
    caption=f"Actual: {labels[y_test[index]]}",
    width=300
)

img = image.reshape(1, 28, 28, 1) / 255.0

prediction = model.predict(img, verbose=0)
predicted_class = np.argmax(prediction)

st.success(f"Predicted: {labels[predicted_class]}")