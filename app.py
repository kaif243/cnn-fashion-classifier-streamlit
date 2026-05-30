import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

st.set_page_config(page_title="Fashion Classification using CNN")

st.title("Fashion Classification using CNN")

# Load dataset
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

# Normalize data
X_train = X_train.reshape(-1, 28, 28, 1) / 255.0
X_test = X_test.reshape(-1, 28, 28, 1) / 255.0

# Build CNN model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train model
model.fit(X_train[:5000], y_train[:5000], epochs=2, verbose=0)

# Evaluate model
loss, accuracy = model.evaluate(X_test[:1000], y_test[:1000], verbose=0)

st.success(f"Model Accuracy: {accuracy:.4f}")

# Class names
class_names = [
    "T-shirt/Top",
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

# Sidebar navigation
st.sidebar.title("Image Navigator")

idx = st.sidebar.slider(
    "Select Test Image",
    0,
    len(X_test)-1,
    0
)

sample = X_test[idx]

# Prediction
prediction = model.predict(sample.reshape(1,28,28,1), verbose=0)
predicted_class = np.argmax(prediction)

st.subheader("Prediction Result")
st.write(f"Predicted Class: **{class_names[predicted_class]}**")

# Display image
fig, ax = plt.subplots()
ax.imshow(sample.reshape(28,28), cmap="gray")
ax.axis("off")
st.pyplot(fig)