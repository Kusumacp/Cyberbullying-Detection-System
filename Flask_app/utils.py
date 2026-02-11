# utils.py
'''
import pickle
import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam

# ---------------------------
# Load Tokenizer from .pkl
# ---------------------------
def load_tokenizer(tokenizer_path):
    with open(tokenizer_path, 'rb') as f:
        tokenizer = pickle.load(f)
    return tokenizer

# ---------------------------
# Preprocess input text
# ---------------------------
def preprocess_text(text, tokenizer, max_len=100):
    # Convert text to sequence
    sequence = tokenizer.texts_to_sequences([text])
    # Pad the sequence
    padded = pad_sequences(sequence, maxlen=max_len, padding='post')
    return padded

# ---------------------------
# Load DL model (with compile=False fix)
# ---------------------------
def load_dl_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    # Load model without compiling (to avoid variable mismatch issues)
    model = load_model(model_path, compile=False)

    # Choose loss function based on model type
    if 'binary' in model_path:
        loss_fn = 'binary_crossentropy'
    else:
        loss_fn = 'categorical_crossentropy'

    # Re-compile with correct optimizer and loss
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss=loss_fn,
                  metrics=['accuracy'])
    
    return model '''
# utils.py
import shap
import numpy as np
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import tensorflow as tf

# Load tokenizer from .pkl file
def load_tokenizer(tokenizer_path):
    with open(tokenizer_path, 'rb') as f:
        tokenizer = pickle.load(f)
    return tokenizer

# Load deep learning model from .h5 file
def load_dl_model(model_path):
    return load_model(model_path, compile=False)

# Preprocess input text
def preprocess_text(text, tokenizer, max_len=100):
    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=max_len, padding='post', truncating='post')
    return padded

# SHAP Explanation (text output only)



# ----------------------
# SHAP Explanation (for DL)


def explain_with_shap(text, model, tokenizer, max_len=100):
    try:
        # Tokenize
        sequence = tokenizer.texts_to_sequences([text])
        padded = pad_sequences(sequence, maxlen=max_len, padding='post')

        # Define a prediction function
        def predict_fn(x):
            return model.predict(x)

        # Use KernelExplainer with a simple background
        background = np.zeros((1, max_len))  # dummy input
        explainer = shap.KernelExplainer(predict_fn, background)

        shap_values = explainer.shap_values(padded, nsamples=100)

        # If model output is single neuron (binary classification)
        if isinstance(shap_values, list):
            shap_values = shap_values[0]

        word_index = tokenizer.word_index
        index_word = {v: k for k, v in word_index.items()}
        words = [index_word.get(idx, '') for idx in sequence[0]]

        shap_vals = shap_values[0][:len(words)]

        # Get top contributing words
        top_contributors = sorted(zip(words, shap_vals), key=lambda x: abs(x[1]), reverse=True)
        top_contributors = [(w, round(float(v), 4)) for w, v in top_contributors if w != ''][:5]

        explanation = "Top contributing words: " + ", ".join([f"{w} ({v})" for w, v in top_contributors])
        return explanation

    except Exception as e:
        return f"SHAP explanation error: {str(e)}"
