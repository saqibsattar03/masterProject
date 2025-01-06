from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import pickle
import json
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

app = Flask(__name__)

MODEL_PATH = 'models/hybrid_model.h5'
TOKENIZER_PATH = 'models/tokenizer.pkl'
CONFIG_PATH = './config.json'

hybrid_model = load_model(MODEL_PATH)

with open(TOKENIZER_PATH, 'rb') as file:
    tokenizer = pickle.load(file)

with open(CONFIG_PATH, 'r') as file:
    config = json.load(file)

max_len = config['max_len']

def preprocess_query(query, tokenizer, max_len=max_len):
    """
    Preprocess a single query for model input.
    """
    sequence = tokenizer.texts_to_sequences([query])
    padded_sequence = pad_sequences(sequence, maxlen=max_len, padding='post', truncating='post')
    return np.array(padded_sequence)

@app.route('/detect', methods=['POST'])
def detect_sql_injection():
    """
    Endpoint for detecting SQL injection.
    """
    try:
        data = request.get_json()
        query = data.get("query", "")

        if not query:
            return jsonify({"error": "No query provided"}), 400

        # Preprocess the query
        preprocessed_query = preprocess_query(query, tokenizer, max_len)

        # Duplicate the input for both branches of the hybrid model
        prediction = hybrid_model.predict([preprocessed_query, preprocessed_query])
        print(f"Prediction... {prediction}")
        is_malicious = bool(prediction[0][0] >= 0.5)

        return jsonify({
            "query": query,
            "is_malicious": is_malicious,
            "confidence": float(prediction[0][0])
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
