
from flask import Flask, request, render_template, jsonify
import joblib
import os

# Load pre-trained model and vectorizer
model_path = "C:/Users/aakan/OneDrive/Desktop/anushka/sentimentAnalysis/models/sentiment_model.pkl"
vectorizer_path = "C:/Users/aakan/OneDrive/Desktop/anushka/sentimentAnalysis/models/vectorizer.pkl"

if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
    raise FileNotFoundError("Model or vectorizer file not found. Please check the paths.")

model = joblib.load(model_path)
vectorizer = joblib.load(vectorizer_path)

#app = Flask(__name__)
app = Flask(__name__, template_folder='C:/Users/aakan/OneDrive/Desktop/anushka/sentimentAnalysis/templates')
# Home route
@app.route("/")
def home():
    return render_template("index.html")

# Prediction route
@app.route("/predict", methods=["POST"])
def predict():
    # Get the text input from the user
    input_text = request.form.get("text")

    if not input_text:
        return jsonify({"error": "Please provide text input for sentiment analysis."})

    # Transform the input text using the loaded vectorizer
    transformed_text = vectorizer.transform([input_text])

    # Predict sentiment using the loaded model
    prediction = model.predict(transformed_text)

    # Map prediction to sentiment label
    sentiment = "Positive" if prediction[0] == 4 else "Negative"

    return jsonify({"input_text": input_text, "sentiment": sentiment})

if __name__ == "__main__":
    app.run(debug=True)

