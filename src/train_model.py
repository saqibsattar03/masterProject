import os

import matplotlib.pyplot as plt
from keras.src import regularizers
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Bidirectional, SpatialDropout1D, Bidirectional, BatchNormalization, GlobalAveragePooling1D,Embedding, Conv1D, MaxPooling1D, Flatten, Dense, Dropout, LSTM, Input, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
import time
import psutil
from tensorflow.keras.utils import model_to_dot

def build_cnn_model(vocab_size, max_len, embedding_dim=50):
    """
    Building CNN model.
    """
    model = Sequential([
        Embedding(70, embedding_dim, input_length=max_len),
        Conv1D(16,kernel_size=3, activation='tanh'),
        MaxPooling1D(4),
        Flatten(),
        Dense(2,activation="tanh"),
        Dropout(0.4),
        Dense(1,activation="sigmoid"),
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    save_model_architecture(model,"CNN_architecture.png")
    model.summary()
    return model


def build_lstm_model(vocab_size, max_len, embedding_dim=50):
    """
    Building LSTM model.
    """
    model = Sequential([
        Embedding(70,embedding_dim, input_length=max_len,trainable=True),
        Bidirectional(LSTM(8, return_sequences=True)),
        # LSTM(1, return_sequences=False),
        BatchNormalization(),
        Dense(1, activation="sigmoid")
    ])
    # optimizer = Adam(learning_rate=0.001)
    model.compile( loss='binary_crossentropy', metrics=['accuracy'])
    save_model_architecture(model, "LSTM_architecture.png")
    model.summary()
    return model

def build_hybrid_model(vocab_size, max_len, embedding_dim=50, dropout_rate=0.4):
    """
    Build a hybrid model combining CNN and LSTM.
    """
    # CNN branch
    cnn_input = Input(shape=(max_len,), name="cnn_input")
    cnn_embedding = Embedding(
        input_dim=70,
        output_dim=embedding_dim,
        input_length=max_len,
        trainable=True,
    )(cnn_input)
    cnn_conv = Conv1D(32, 3, activation='relu')(cnn_embedding)
    cnn_pool = MaxPooling1D(pool_size=2)(cnn_conv)
    cnn_flat = Flatten()(cnn_pool)
    cnn_flat = Dropout(dropout_rate)(cnn_flat)

    # LSTM branch
    lstm_input = Input(shape=(max_len,), name="lstm_input")
    lstm_embedding = Embedding(
        input_dim=70,
        output_dim=embedding_dim,
        input_length=max_len,
    )(lstm_input)
    lstm_out = LSTM(16, return_sequences=False)(lstm_embedding)
    lstm_out = Dropout(dropout_rate)(lstm_out)

    # Concatenate CNN and LSTM outputs
    merged = concatenate([cnn_flat, lstm_out])
    merged_dense = Dense(32, activation='relu')(merged)
    merged_dense = Dropout(dropout_rate)(merged_dense)
    final_output = Dense(1, activation='sigmoid', name="output")(merged_dense)

    # Compile the model
    model = Model(inputs=[cnn_input, lstm_input], outputs=final_output)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    save_model_architecture(model, "Hybrid_architecture.png")
    model.summary()
    return model

def train_model(model, X_train, y_train, X_test, y_test, model_name, epochs=20, batch_size=64):
    # Measure time and memory
    start_time = time.time()
    process = psutil.Process()
    initial_memory = process.memory_info().rss

    print(f"Training {model_name} model...")
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping],
        verbose=1
    )

    training_time = time.time() - start_time
    final_memory = process.memory_info().rss
    memory_usage = (final_memory - initial_memory) / (1024 * 1024)  # Convert to MB

    MODEL_PATH = f'models/{model_name.lower()}_model.h5'
    model.save(MODEL_PATH)
    print(f"{model_name} Training Time: {training_time:.2f} seconds")
    print(f"{model_name} Memory Usage: {memory_usage:.2f} MB")

    return model, history, training_time, memory_usage

def save_model_architecture(model, file_name, output_dir="src/results"):

    try:
        dot = model_to_dot(model, dpi=65)
        file_path = os.path.join(output_dir, file_name)
        os.makedirs(output_dir, exist_ok=True)
        with open(file_path, "wb") as f:
            f.write(dot.create(prog='dot', format='png'))
        print(f"Model architecture saved successfully to {file_path}")

    except FileNotFoundError as e:
        print(f"Error: File not found - {e}")

    except Exception as e:
        print(f"An error occurred: {e}")