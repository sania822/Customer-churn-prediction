from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load model when server starts
model = joblib.load("model.pkl")

@app.route("/")
def home():
    return "Churn Prediction API Running"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json["features"]
        data = np.array(data).reshape(1, -1)

        prediction = model.predict(data)[0]

        return jsonify({
            "prediction": int(prediction)
        })

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)