import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Optional, List, Dict


@dataclass
class TrackPoint:
    latitude: float
    longitude: float
    elevation: float
    time: datetime

    def to_json(self) -> Dict:
        """Serialize to JSON-friendly dictionary"""
        data = asdict(self)
        # Convert datetime to ISO format string
        if self.time:
            data['time'] = self.time.isoformat()
        return data

    @classmethod
    def from_json(cls, data: Dict) -> 'TrackPoint':
        """Deserialize from dictionary"""
        # Convert ISO string back to datetime
        time_str = data.get('time')
        time = datetime.fromisoformat(time_str) if time_str else None

        return cls(
            latitude=data['latitude'],
            longitude=data['longitude'],
            elevation=data.get('elevation'),
            time=time
        )

    def __repr__(self):
        return (f"TrackPoint(lat={self.latitude:.6f}, lon={self.longitude:.6f}, "
                f"ele={self.elevation}, time={self.time})")


# Example usage
def create_trackpoints() -> List[TrackPoint]:
    """Example of creating trackpoints from data"""
    return [
        TrackPoint(
            latitude=40.7128,
            longitude=-74.0060,
            elevation=10.5,
            time=datetime.now()
        ),
        TrackPoint(
            latitude=40.7129,
            longitude=-74.0061,
            elevation=11.2,
            time=datetime.now()
        )
    ]


def serialize_trackpoints(points: List[TrackPoint]) -> str:
    """Serialize list of TrackPoints to JSON string"""
    return json.dumps([p.to_json() for p in points])


def deserialize_trackpoints(json_str: str) -> List[TrackPoint]:
    """Deserialize JSON string to list of TrackPoints"""
    return [TrackPoint.from_json(p) for p in json.loads(json_str)]


def write_trackpoints_to_file(trackpoints: List[TrackPoint], filename: str = "trackpoints.json"):
    """Writes trackpoints to a JSON file in the project root"""
    # Get JSON string from trackpoints
    json_data = serialize_trackpoints(trackpoints)

    # Write to file
    with open(filename, 'w') as f:
        f.write(json_data)

    print(f"Successfully wrote {len(trackpoints)} points to {os.path.abspath(filename)}")


def read_trackpoints_from_file(filename: str = "trackpoints.json") -> List[TrackPoint]:
    """Reads trackpoints from a JSON file in the project root"""
    with open(filename, 'r') as f:
        json_data = f.read()
    return deserialize_trackpoints(json_data)


# Usage example:
if __name__ == "__main__":
    # Create and serialize
    original_points = create_trackpoints()
    json_str = serialize_trackpoints(original_points)
    print(f"Serialized:\n{json_str}\n")

    # Deserialize and use
    reconstructed_points = deserialize_trackpoints(json_str)
    print("Deserialized points:")
    for p in reconstructed_points:
        print(p)

    # For your prediction function, convert to dict list:
    prediction_input = [p.to_json() for p in reconstructed_points]
    # Then use with your predict_distance function