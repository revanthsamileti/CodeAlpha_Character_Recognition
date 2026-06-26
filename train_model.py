import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Input
from tensorflow.keras.callbacks import ModelCheckpoint

num_classes = 10 

def load_mnist_data():
    print("Loading standard MNIST dataset (Digits 0-9)...")
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    
    # Normalize and add channel dimension
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    
    return x_train, y_train, x_test, y_test

def build_model(input_shape):
    """Builds a CNN model suitable for digit recognition."""
    inputs = Input(shape=input_shape)
    
    # Data Augmentation
    x = tf.keras.layers.RandomRotation(0.1)(inputs)
    x = tf.keras.layers.RandomZoom(0.1)(x)
    
    x = Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)
    
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)
    
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs, outputs)
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

if __name__ == "__main__":
    x_train, y_train, x_test, y_test = load_mnist_data()
    
    print("Building model...")
    model = build_model((28, 28, 1))
    model.summary()
    
    # Save the best model
    checkpoint = ModelCheckpoint('mnist_model.h5', monitor='val_accuracy', save_best_only=True, mode='max')
    
    print("\nStarting training...")
    history = model.fit(
        x_train, y_train,
        batch_size=128,
        epochs=10, 
        validation_data=(x_test, y_test),
        callbacks=[checkpoint]
    )
    
    print("Training complete! Model saved as 'mnist_model.h5'.")
    print("You can now run 'python app.py' to draw numbers and get predictions.")
