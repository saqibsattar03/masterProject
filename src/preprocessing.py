import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import os
import pickle
import json

from results.visualizations import plot_histogram, plot_balanced_histogram



def save_max_len(max_len, filepath='./config.json'):
    """
    Args:
        max_len: The maximum sequence length to save.
        filepath: The path to the configuration file.
    """
    try:
        # Ensure the directory for the file exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # Load existing configuration if the file exists
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                config = json.load(f)
            print(f"Existing configuration loaded from {filepath}")
        else:
            # If file does not exist, create an empty configuration
            config = {}

        # Update the max_len value
        config["max_len"] = max_len

        # Save the updated configuration back to the file
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=4)
        print(f"max_len value updated in {filepath}")

    except Exception as e:
        print(f"Error updating max_len in configuration: {e}")

def load_and_preprocess_data(file_path, test_size=0.2, max_words=70):
    """
    Load, preprocess, and tokenize the dataset.
    Args:
        file_path (str): Path to the dataset CSV file.
        test_size (float): Fraction of data to use for testing.
        max_words (int): Maximum number of words for tokenization.

    Returns:
        X_train_S, X_test, y_train_S, y_test, tokenizer, max_len
    """
    try:
        # Load the dataset
        print("Loading data...")
        data = pd.read_csv(file_path)

        # Check for required columns
        if 'label' not in data.columns or 'query' not in data.columns:
            raise ValueError("Dataset must contain 'label' and 'query' columns.")

        # Plot histogram before data cleaning
        plot_histogram(data, 'label', 'Label Distribution Before Cleaning', 'Labels', 'Count', 'src/results/before_cleaning_histogram.png')

        # Handle missing values
        print("Checking and handling missing values...")
        data = data.dropna(subset=['label'])  # Drop rows with missing labels
        data['query'] = data['query'].fillna("").astype(str)  # Fill missing text with empty strings
        data['label'] = pd.to_numeric(data['label'], errors='coerce')  # Convert labels to numeric
        data = data.dropna(subset=['label']).astype({'label': 'int'})  # Drop invalid rows
        print(f"Rows after cleaning: {len(data)}")

        # Plot histogram after data cleaning
        plot_histogram(data, 'label', 'Label Distribution After Cleaning', 'Labels', 'Count', 'src/results/after_cleaning_histogram.png')

        # Separate features and labels
        X = data['query']
        y = data['label']

        # Tokenize text data
        print("Tokenizing text data...")
        tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
        tokenizer.fit_on_texts(X)
        sequences = tokenizer.texts_to_sequences(X)
        max_len = max(len(seq) for seq in sequences)  # Maximum sequence length
        X_padded = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')

        save_max_len(max_len)


        # Split into training and testing sets
        print("Splitting data into training and testing sets...")
        X_train, X_test, y_train, y_test = train_test_split(X_padded, y, test_size=test_size, stratify=y, random_state=42)

        # Apply SMOTE for data balancing
        # print("Applying SMOTE to balance training data...")
        # smote = SMOTE(random_state=42)
        # X_train_S, y_train_S = smote.fit_resample(X_train, y_train)
        # print(f"Training data size after SMOTE: {len(X_train_S)}")

        # Plot balanced data histogram
        # plot_balanced_histogram(y_train_S, 'Label Distribution After Data Balancing', 'Labels', 'Count', 'src/results/after_smote_histogram.png')

        # Save tokenizer for reuse
        TOKENIZER_PATH = 'models/tokenizer.pkl'
        os.makedirs(os.path.dirname(TOKENIZER_PATH), exist_ok=True)
        with open(TOKENIZER_PATH, 'wb') as file:
            pickle.dump(tokenizer, file)
        print(f"Tokenizer saved at {TOKENIZER_PATH}")

        return X_train, X_test, y_train, y_test, tokenizer, max_len

    except Exception as e:
        print(f"Error in preprocessing: {e}")
        raise