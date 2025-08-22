from flask import Flask, request, jsonify
import tensorflow as tf
import joblib
import pandas as pd
from flask_cors import CORS
import os

# Disable GPU (Render free tier doesn’t support CUDA anyway)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Limit TensorFlow thread usage (important on small machines)
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

# Load model and scaler
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

        if prev_speed is None or prev_acc is None:
            return jsonify({"error": "Missing required fields: prev_speed, prev_acc"}), 400

        # Derive additional inputs
        desired_speed = prev_speed + 10   # Auto-deriving this
        desired_acc = prev_acc            # Keeping it same
        time = 10                         # Fixed value

        columns = ["prev_speed", "prev_acc", "desired_speed", "desired_acc", "time"]

        # Prepare dataframe
        input_data = pd.DataFrame([[prev_speed, prev_acc, desired_speed, desired_acc, time]], columns=columns)

        # Scale input
        input_scaled = scaler.transform(input_data)

        # Predict
        prediction = model.predict(input_scaled)
        predicted_days = float(prediction[0][0])

        return jsonify({
            "predicted_no_of_days": round(predicted_days, 2)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == '__main__':
    # Render (and other PaaS) assign port dynamically via $PORT
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
