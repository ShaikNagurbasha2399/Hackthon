from flask import Flask, request, jsonify
import tensorflow as tf
import joblib
import pandas as pd
from flask_cors import CORS


# Limit TensorFlow thread usage (important on Render free tier)
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)


# Load model and scaler (compile disabled for inference)
model = tf.keras.models.load_model("typing_days_regression_model.h5", compile=False)
scaler = joblib.load("scaler.pkl")

# Initialize Flask app
app = Flask(__name__)
CORS(app)


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Expecting JSON input
        data = request.get_json(force=True)

        # Extract input values
        prev_speed = data.get("prev_speed")
        prev_acc = data.get("prev_acc")
        desired_speed = prev_speed + 10   # You’re auto-deriving this
        desired_acc = prev_acc            # Keeping it same
        time = 10                         # Fixed value

        columns = ["prev_speed", "prev_acc", "desired_speed", "desired_acc", "time"]

        # Convert to numpy array
        input_data = pd.DataFrame([[prev_speed, prev_acc, desired_speed, desired_acc, time]], columns=columns)

        # Scale input
        input_scaled = scaler.transform(input_data)

        # Make prediction
        prediction = model.predict(input_scaled)
        predicted_days = float(prediction[0][0])

        # Return result as JSON
        return jsonify({
            "predicted_no_of_days": round(predicted_days, 2)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(port=7000)
