from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import shap
import os

# Disable GPU to avoid SHAP CUDA issues
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Load the trained model and preprocessors
with open("models/crop_model.pkl", "rb") as f:
    model = pickle.load(f)
with open("models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
with open("models/label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# Define feature names
FEATURES = ["Nitrogen", "Phosphorus", "Potassium", "Temperature", "Humidity", "pH", "Rainfall"]

# Initialize Flask app
app = Flask(__name__)

@app.route("/")
def index():
    """Render the homepage with an input form."""
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    """Predict the recommended crop based on user input."""
    try:
        # Get user input from form
        user_input = [float(request.form[key]) for key in FEATURES]

        # Scale the input data
        input_scaled = scaler.transform([user_input])

        # Make prediction
        prediction = model.predict(input_scaled)
        predicted_crop = label_encoder.inverse_transform(prediction)[0]

        # Explain prediction using SHAP
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_scaled)

        # Extract SHAP values for the predicted class
        if isinstance(shap_values, list):
            shap_values = np.array(shap_values)  # Convert to NumPy array

        if len(shap_values.shape) == 3:
            shap_values = np.transpose(shap_values, (2, 0, 1))

        predicted_class_index = np.argmax(prediction)
        shap_values_for_prediction = shap_values[predicted_class_index][0]

        # Convert SHAP values to a dictionary
        shap_explanation = {FEATURES[i]: float(shap_values_for_prediction[i]) for i in range(len(FEATURES))}

        return jsonify({"crop": predicted_crop, "explanation": shap_explanation})
    
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route("/feature-importance")
def feature_importance():
    """Serve the SHAP feature importance plot."""
    return jsonify({"image_url": "/static/shap_feature_importance.png"})

if __name__ == "__main__":
    app.run(debug=True)
