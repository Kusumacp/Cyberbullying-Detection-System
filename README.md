# Cyberbullying-Detection-System
Project Overview

The increasing popularity of online social platforms has improved communication, but it has also increased negative behavior such as cyberbullying. This project focuses on building an automatic cyberbullying detection system that not only detects bullying content but also identifies the type of bullying present in the text.
This system compares Machine Learning (ML) and Deep Learning (DL) approaches to classify text data. The dataset contains text samples labeled as bullying or non-bullying (and by type), which are cleaned and preprocessed before training the models.

Objectives: 

Detect whether a given text contains cyberbullying or not
Classify the type of bullying from the text
Compare performance of ML and DL models
Provide Explainable AI (XAI) using SHAP to understand model predictions
Deploy the model using a Flask web application with an HTML interface

Models Used

Machine Learning Models
Logistic Regression (LR)
Random Forest (RF)
AdaBoost
Text features are represented using Count Vectorization.

Deep Learning Models

Convolutional Neural Network (CNN)
BiLSTM
GRU
Stacked Ensemble Model

These models use pre-trained GloVe embeddings to capture semantic meaning and contextual relationships in text.

Evaluation Metrics
Models are evaluated using:
Accuracy
Precision
Recall
F1-score
Confusion Matrix

The experimental results show that Deep Learning models outperform Machine Learning models, with CNN and BiLSTM giving especially strong performance.

Explainable AI (XAI)

To improve transparency, SHAP (SHapley Additive exPlanations) is used to:
Explain model predictions
Identify important words/features that influence the output
Make the system more interpretable and trustworthy

Web Application

The trained model is deployed using:
Flask for backend
HTML/CSS for frontend
Users can enter text in the web interface and get:
Whether the text is cyberbullying or not
The type of bullying detected
Explanation of the prediction (using XAI)

Technologies Used

Python
Jupyter Notebook
Scikit-learn
TensorFlow / Keras
NLP (CountVectorizer, GloVe)
Flask
HTML / CSS
SHAP (Explainable AI) 


How to Run the Project

1. Clone the repository:
   git clone https://github.com/kusuma-c-p/Cyberbullying-Detection-System.git

2. Install required libraries:
    numpy, pandas, scikit-learn, tensorflow, flask, shap 

3. Go to the Flask app folder and run:
   python app.py

4. Open your browser and go to:
   http://127.0.0.1:5000/






