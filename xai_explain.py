import shap
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import os
import sys

# Add flask_app folder to import utils
sys.path.append(os.path.join(os.getcwd(), 'flask_app'))
from Flask_app.utils import preprocess_text

# Max sequence length (same as used in training)
max_len = 100

# Load tokenizer
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

# Load models
binary_model_path = 'Flask_app/models/stacked_binary_model.h5'
type_model_path = 'Flask_app/models/stacked_type_model.h5'
binary_model = load_model(binary_model_path, compile=False)
type_model = load_model(type_model_path, compile=False)

# Class labels
binary_class = ['Non-Cyberbullying', 'Cyberbullying']
type_classes = ['ethnicity', 'general', 'other', 'political', 'religious', 'sexual']

# Convert text to padded sequence
def text_to_sequence(text):
    text = str(text)
    seq = tokenizer.texts_to_sequences([text])
    return pad_sequences(seq, maxlen=max_len, padding='post', truncating='post')

# Get readable tokens from the input
def get_tokens(input_seq):
    index_word = {v: k for k, v in tokenizer.word_index.items()}
    return [index_word.get(i, "<OOV>") for i in input_seq[0] if i != 0]

# SHAP explanation
def explain_prediction(text):
    input_seq = text_to_sequence(text)
    tokens = get_tokens(input_seq)

    # --- Binary prediction ---
    binary_pred = binary_model.predict(input_seq)[0][0]
    binary_label = binary_class[int(round(binary_pred))]

    print("\n=== Binary Classification (Cyberbullying) ===")
    print(f"Input Text: {text}")
    print(f"Prediction: {binary_label} (Confidence: {binary_pred:.2f})")

    # SHAP explanation for binary
    explainer_bin = shap.KernelExplainer(binary_model.predict, np.zeros((1, max_len)))
    shap_values_bin = explainer_bin.shap_values(input_seq, nsamples=100)
    values_bin = np.array(shap_values_bin[0][0]).flatten()[:len(tokens)]

    top_words_bin = sorted(zip(tokens, values_bin), key=lambda x: abs(x[1]), reverse=True)[:5]

    print("Top Influential Words:")
    for word, val in top_words_bin:
        impact = "increased" if val > 0 else "decreased"
        print(f"  - '{word}' {impact} the cyberbullying probability (SHAP value: {val:.4f})")

    # --- Multi-Class Type prediction ---
    type_probs = type_model.predict(input_seq)[0]
    type_index = np.argmax(type_probs)
    type_label = type_classes[type_index]
    type_confidence = type_probs[type_index]

    print("\n=== Multi-Class Type Classification ===")
    print(f"Predicted Bullying Type: {type_label} (Confidence: {type_confidence:.2f})")

    # SHAP explanation for multi-class safely
    explainer_type = shap.KernelExplainer(type_model.predict, np.zeros((1, max_len)))
    shap_values_type_all = explainer_type.shap_values(input_seq, nsamples=100)

    # Handle multi-class SHAP output
    if isinstance(shap_values_type_all, list) and len(shap_values_type_all) == len(type_classes):
        values_type = shap_values_type_all[type_index][0]
    else:
        values_type = shap_values_type_all[0]

    # Flatten to 1D array
    values_type = np.array(values_type).flatten()[:len(tokens)]

    top_words_type = sorted(zip(tokens, values_type), key=lambda x: abs(x[1]), reverse=True)[:5]

    print("Top Influential Words for Type Prediction:")
    for word, val in top_words_type:
        impact = "increased" if val > 0 else "decreased"
        print(f"  - '{word}' {impact} the probability for type '{type_label}' (SHAP value: {val:.4f})")


# Example usage
if __name__ == "__main__":
    user_input = input("Enter a social media text to explain: ").strip()
    explain_prediction(user_input)
