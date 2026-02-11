'''from flask import Flask, render_template, request
from utils import load_tokenizer, load_dl_model, preprocess_text
import numpy as np
import shap
import os
import tensorflow as tf

app = Flask(__name__)

# Load tokenizer
tokenizer = load_tokenizer("tokenizer.pkl")
base_dir = os.path.dirname(os.path.abspath(__file__))

# Load models once
binary_model_path = os.path.join(base_dir, 'models', 'stacked_binary_model.h5')
type_model_path = os.path.join(base_dir, 'models', 'stacked_type_model.h5')

binary_model = load_dl_model(binary_model_path)
type_model = load_dl_model(type_model_path)

# Labels
type_labels = ['ethnicity', 'other', 'political', 'religious', 'sexual', 'threat', 'troll', 'vocational']

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    shap_explanation = None

    if request.method == 'POST':
        user_text = request.form.get('text', '')  # safer way to get input

        if not user_text.strip():
            result = "Please enter some text to analyze."
            return render_template('index.html', result=result)

        try:
            # Preprocess input
            processed = preprocess_text(user_text, tokenizer)  # shape: (1, 100)

            # 1. Binary prediction
            binary_pred = binary_model.predict(processed)
            pred_score = float(binary_pred[0][0])  # Ensure scalar
            is_bullying = pred_score > 0.5

            if is_bullying:
                # 2. Type prediction
                type_pred = type_model.predict(processed)
                if type_pred is not None and len(type_pred[0]) == len(type_labels):
                    max_prob = np.max(type_pred)
                    if max_prob > 0.3:
                        bullying_type = type_labels[np.argmax(type_pred)]
                        result = f"✅ Cyberbullying Detected — Type: {bullying_type.capitalize()}"
                    else:
                        result = "✅ Cyberbullying Detected — Type: Uncertain"
                else:
                    result = "✅ Cyberbullying Detected — Type: Unknown"
            else:
                result = "❎ Not Cyberbullying"

            # 3. SHAP Explanation (final bulletproof fix)
            try:
                background = np.zeros_like(processed)
                explainer = shap.GradientExplainer(binary_model, background)
                shap_values_raw = explainer.shap_values(processed)

                # Ensure numpy array
                shap_values = shap_values_raw[0] if isinstance(shap_values_raw, list) else shap_values_raw
                shap_values = np.array(shap_values)  # Ensure it's np.array, even if scalar

                # Handle bad shape (0D or 1D)
                if shap_values.ndim < 2:
                    shap_explanation = "SHAP explanation not available (invalid or scalar output)."
                else:
                    # Decode input indices to words
                    reverse_word_index = {v: k for k, v in tokenizer.word_index.items()}
                    input_indices = processed[0]
                    input_words = [reverse_word_index.get(idx, '') for idx in input_indices if idx != 0]

                    # Extract SHAP scores
                    if shap_values.ndim == 3:
                        scores = shap_values[0][:len(input_words)]
                    else:  # 2D
                        scores = shap_values[0][:len(input_words)]

                    # Pair and sort
                    explanation = sorted(zip(input_words, scores), key=lambda x: abs(x[1]), reverse=True)

                    top_explanation = [
                        f"• '{word}' contributed {'positively' if score > 0 else 'negatively'} ({score:.4f})"
                        for word, score in explanation[:5] if word != ''
                    ]
                    shap_explanation = "🔍 Why this prediction?\n" + "\n".join(top_explanation)

            except Exception as e:
                shap_explanation = f"SHAP explanation error: {str(e)}"

        except Exception as e:
            result = f"❗ Error occurred: {str(e)}"
            shap_explanation = None

    return render_template('index.html', result=result, explanation=shap_explanation)

if __name__ == '__main__':
    app.run(debug=True)'''
    



from flask import Flask, render_template, request
from utils import load_tokenizer, load_dl_model, preprocess_text, explain_with_shap
import numpy as np
import os

app = Flask(__name__)

# Load tokenizer
tokenizer = load_tokenizer("tokenizer.pkl")
base_dir = os.path.dirname(os.path.abspath(__file__))

# Load deep learning models
binary_model = load_dl_model(os.path.join(base_dir, 'models', 'stacked_binary_model.h5'))
type_model = load_dl_model(os.path.join(base_dir, 'models', 'stacked_type_model.h5'))

# Bullying type labels
type_labels = ['ethnicity', 'general', 'other', 'political', 'religious', 'sexual']

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    shap_explanation = None

    if request.method == 'POST':
        user_text = request.form.get('text', '')

        if not user_text.strip():
            result = "⚠️ Please enter some text."
            return render_template('index.html', result=result)

        try:
            # Preprocess
            processed = preprocess_text(user_text, tokenizer, max_len=100)

            # Binary prediction
            binary_pred = binary_model.predict(processed)
            pred_score = float(binary_pred[0][0])
            is_bullying = pred_score > 0.5

            # SHAP Explanation
            shap_explanation = explain_with_shap(user_text, binary_model, tokenizer, max_len=100)

            if is_bullying:
                type_pred = type_model.predict(processed)
                if type_pred is not None and len(type_pred[0]) == len(type_labels):
                    max_prob = np.max(type_pred)
                    if max_prob > 0.3:
                        bullying_type = type_labels[np.argmax(type_pred)]
                        result = f"✅ Cyberbullying Detected — Type: {bullying_type.capitalize()}"
                    else:
                        result = "✅ Cyberbullying Detected — Type: Uncertain"
                else:
                    result = "✅ Cyberbullying Detected — Type: Unknown"
            else:
                result = "❎ Not Cyberbullying"

        except Exception as e:
            result = f"❗ Error occurred: {str(e)}"
            shap_explanation = None

    return render_template('index.html', result=result, explanation=shap_explanation)

if __name__ == '__main__':
    app.run(debug=True)
