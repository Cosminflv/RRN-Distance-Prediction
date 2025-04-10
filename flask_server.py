from flask import request, jsonify, Flask

from predict_distance import predict_distance
from rnn_model import RNNTracker
from trackpoint import deserialize_trackpoints, TrackPoint

app = Flask(__name__)

tracker = RNNTracker.load("model.keras")
tracker.compile(loss='mse', metrics=['accuracy'])

@app.route('/predict', methods=['POST'])
def predict():
    """
    Expects JSON payload as an array of track point objects:
    [
        {
            "latitude": 45.58652,
            "longitude": 24.76841,
            "elevation": 1416.56,
            "time": "2020-08-02T05:01:14+00:00"
        },
        ...
    ]
    """
    try:
        # Read and validate JSON payload
        data = request.get_json(force=True)

        if not isinstance(data, list):
            return jsonify({"error": "Expected array of track points"}), 400

        # Deserialize track points
        trackpoints = [TrackPoint.from_json(p) for p in data]

        # Get prediction from the tracker model
        dist = predict_distance(trackpoints, tracker)

        return jsonify({
            "prediction": dist,
            "points_processed": len(trackpoints)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    # Run the Flask app
    app.run(host='0.0.0.0', debug=True)
