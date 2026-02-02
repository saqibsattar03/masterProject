# Real-Time SQL Injection Detection Using Hybrid CNN-LSTM Deep Learning Approach

**MSc Cyber Security Dissertation Project**  
**Author**: Saqib Sattar (Matriculation: S2360784)  
**University**: Glasgow Caledonian University  
**Supervisor**: Pius Owoh  
**Year**: 2025  

## Project Overview
This project develops a **real-time SQL injection (SQLi) detection system** using a hybrid Convolutional Neural Network (CNN) + Long Short-Term Memory (LSTM) deep learning model.  

- CNN extracts local malicious patterns (e.g., suspicious keywords, syntax like 'OR 1=1', '--') from SQL queries.  
- LSTM captures long-range sequential dependencies in the query structure.  
- The hybrid model achieves **98.08% accuracy** on the test set (with 99.14% precision and 99.63% ROC-AUC), outperforming standalone CNN and LSTM models.  

A **Flask API** enables real-time inference: send SQL queries via POST requests (e.g., Postman), and the system instantly predicts whether the query is **malicious** or **benign**, with confidence score. Malicious queries are flagged for blocking (simulated in this demo).

This work addresses a key OWASP Top 10 vulnerability and demonstrates deployable AI for web application security.

## Key Features
- Hybrid CNN-LSTM architecture (compared against standalone CNN and LSTM)  
- Real-time detection via Flask REST API (live testing with Postman)  
- Preprocessing: tokenization, padding, dataset balancing, 80/20 train-test split  
- Full evaluation: confusion matrix, precision, recall, F1-score, ROC-AUC, computational cost  

## Installation & Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/saqibsattar03/masterProject.git
   cd masterProject

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
3. Run the Flask API:
   ```bash
   python app.py
   → Server starts at http://127.0.0.1:5000

**How to Test Real-Time Detection (Demo)**
Use Postman or curl to send a POST request to /detect:
Example Request Body (JSON):
JSON{
  "query": "SELECT * FROM users WHERE id = 1 OR 1=1 --"
}
Expected Response (Malicious example):
JSON{
  "query": "SELECT * FROM users WHERE id = 1 OR 1=1 --",
  "is_malicious": true,
  "confidence": 0.98
}
Benign example:
JSON{
  "query": "SELECT name FROM products WHERE id = 123",
  "is_malicious": false,
  "confidence": 0.12
}

**Project Structure**
textmasterProject/
├── app.py                  # Flask API for real-time prediction
├── models/                 # Trained model & tokenizer
│   ├── hybrid_model.h5
│   └── tokenizer.pkl
├── data/raw/               # Original dataset
├── src/                    # Training & preprocessing scripts
├── config.json             # Model parameters (max_len, etc.)
├── requirements.txt
└── README.md

**Results Summary (From Dissertation)**

Model Accuracy: 98.08% (hybrid CNN-LSTM on test set)
Key Metrics: 99.14% Precision, high Recall, 99.63% ROC-AUC
Real-Time Performance: Low-latency inference tested live via Postman (block/pass decisions)
Full details, confusion matrix, ROC curves, training plots, and methodology in the dissertation.


**Full Dissertation**
Download the full MSc dissertation PDF here
https://github.com/saqibsattar03/masterProject/blob/master/Saqib-Sattar-MSc-Dissertation-Real-Time-SQL-Injection-Detection.pdf


**Future Improvements**

Cloud deployment (Render, Heroku, AWS)
Add attention mechanism to LSTM
Adversarial testing (evasion techniques)
Integration with real WAF/proxy


**License**
MIT License

**Contact**
GitHub: @saqibsattar03
LinkedIn: https://www.linkedin.com/in/saqib-sattar-53864a136/

**Thank you for checking out the project!**
