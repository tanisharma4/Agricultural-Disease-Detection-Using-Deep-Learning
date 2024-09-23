import streamlit as st
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Sequential
import numpy as np

# Define constants
IMAGE_SIZE = 256
BATCH_SIZE = 32
EPOCHS = 50

# Function to load dataset
def load_data(directory):
    # Load and preprocess dataset
    dataset = tf.keras.preprocessing.image_dataset_from_directory(
        directory,
        shuffle=True,
        image_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE
    )

    return dataset

# Function to create model
def create_model(input_shape, n_classes):
    model = Sequential([
        Input(shape=input_shape),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(64, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(64, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(n_classes, activation='softmax')
    ])
    return model

@st.cache
def preprocess_data(dataset):
    # Preprocess the data
    processed_data = dataset.map(lambda x, y: (x / 255.0, y))
    return processed_data

def main():
    # Specify the directory containing the dataset
    dataset_directory = "C:/Users/KIIT/Desktop/minor/Potato_disease/training/PlantVillage"

    # Load dataset
    dataset = load_data(dataset_directory)
    class_names = dataset.class_names

    # Display class names
    st.write("Class Names:", class_names)

    # Split dataset into train, validation, and test sets
    train_size = 0.8
    val_size = 0.1
    test_size = 0.1
    train_ds = dataset.take(int(len(dataset) * train_size))
    test_ds = dataset.skip(int(len(dataset) * train_size))
    val_ds = test_ds.take(int(len(test_ds) * val_size))
    test_ds = test_ds.skip(int(len(test_ds) * val_size))

    # Preprocess the data
    train_ds = preprocess_data(train_ds)
    val_ds = preprocess_data(val_ds)
    test_ds = preprocess_data(test_ds)

    # Create and compile model
    model = create_model(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3), n_classes=len(class_names))
    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=['accuracy']
    )

    # Train model
    st.write("Training Model...")
    history = model.fit(
        train_ds,
        epochs=EPOCHS,
        validation_data=val_ds
    )

    # Evaluate model
    st.write("Evaluating Model...")
    scores = model.evaluate(test_ds)
    st.write("Test Loss:", scores[0])
    st.write("Test Accuracy:", scores[1])

    # Display training and validation metrics
    st.write("Training History:")
    st.line_chart(history.history)

    # Display sample predictions
    st.write("Sample Predictions:")
    for i, (image, label) in enumerate(test_ds.take(9)):
        st.image(image.numpy().astype("uint8"), caption=f"Sample Image {i+1}")
        predictions = model.predict(np.expand_dims(image, axis=0))
        predicted_class = class_names[np.argmax(predictions[0])]
        actual_class = class_names[label.numpy()]
        st.write(f"Actual: {actual_class}, Predicted: {predicted_class}")

if __name__ == "__main__":
    main()

