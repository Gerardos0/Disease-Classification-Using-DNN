import numpy as np
import matplotlib.pyplot as plt
import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical

def build_nn(input_dim, num_classes):
    model = Sequential([
        Dense(512, activation='relu', input_shape=(input_dim,)),
        Dropout(0.4),
        Dense(256, activation='relu'),
        Dropout(0.4),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_nn(model, x_train, y_train, x_test, y_test):
    start = time.time()
    history = model.fit(
        x_train, y_train,
        epochs=32,
        batch_size=128,
        validation_data=(x_test, y_test),
        verbose=0
    )
    end = time.time()
    print(f"Training time: {end - start:.2f} sec")
    return history

def plot_history(history):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(history.history['accuracy'], label='Train')
    ax[0].plot(history.history['val_accuracy'], label='Test')
    ax[0].set_title('Accuracy'); ax[0].legend()
    ax[1].plot(history.history['loss'], label='Train')
    ax[1].plot(history.history['val_loss'], label='Test')
    ax[1].set_title('Loss'); ax[1].legend()
    plt.show()

def summarize_metrics(history):
    labels = ['accuracy', 'val_accuracy', 'loss', 'val_loss']
    for key in labels:
        values = history.history[key]
        if 'loss' in key:
            best_epoch = np.argmin(values)
        else:
            best_epoch = np.argmax(values)
        print(f"Best {key}: {values[best_epoch]:.4f} @ epoch {best_epoch+1}")
