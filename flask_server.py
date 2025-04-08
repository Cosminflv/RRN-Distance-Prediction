import numpy as np
from flask import request, jsonify, Flask

from rnn_model import RNNTracker

app = Flask(__name__)

# Load or initialize your trained tracker model.
# If you have a saved model file, you could load it using RNNTracker.load()
# For example:
tracker = RNNTracker.load("model.keras")
# Otherwise, if you want to instantiate a new model, ensure the input_shape matches your data.
# tracker = RNNTracker(input_shape=(50, 3))
tracker.model.load_weights("model.keras")  # Alternatively, load your model weights
tracker.compile(loss='mse', metrics=['accuracy'])


@app.route('/predict', methods=['POST'])
def predict():
    """
    Expects JSON payload of the form:
    {
       "points": [
           [feature1, feature2, feature3],
           [feature1, feature2, feature3],
           ... (total number of points should match the model's sequence length, e.g., 50)
       ]
    }
    """
    # Read JSON payload.
    data = request.get_json(force=True)

    if 'points' not in data:
        return jsonify({"error": "Missing 'points' in request"}), 400

    try:
        points = data['points']
        # Ensure points is a 2D array and reshape to add batch dimension.
        # For example, if your model expects a shape (sequence_length, features),
        # you need to wrap your data in an extra list/array to create a batch of size 1.
        input_array = np.array(points, dtype=np.float32)

        # Verify input shape matches expected shape (sequence_length, features)
        expected_seq_length, expected_features = tracker.input_shape
        if input_array.shape != (expected_seq_length, expected_features):
            return jsonify({
                "error": f"Expected input shape ({expected_seq_length}, {expected_features}), "
                         f"got {input_array.shape}"
            }), 400

        # Add batch dimension: (1, sequence_length, features)
        input_array = np.expand_dims(input_array, axis=0)

        # Get prediction from the tracker model.
        prediction = tracker.predict(input_array)

        # The prediction may be an array (for a single sample, shape (1,1)); convert to a list.
        prediction_value = prediction.flatten().tolist()

        return jsonify({"prediction": prediction_value})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    # Run the Flask app
    app.run(debug=True)
