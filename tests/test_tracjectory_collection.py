import pandas as pd
from puddy.trajectory_collection import TrajectoryCollection, ColumnConfig, NormalizedTrajectory

def test_load_from_file_dataframe():
    data = [
        {"lon": 10, "lat": 20, "alt": 1000, "identifier": "A"},
        {"lon": 11, "lat": 21, "alt": 1005, "identifier": "A"},
        {"lon": 30, "lat": 40, "alt": 1500, "identifier": "B"},
        {"lon": 31, "lat": 41, "alt": 1501, "identifier": "B"},
    ]
    df = pd.DataFrame(data)
    config = ColumnConfig.create_geo(lon_col="lon", lat_col="lat", alt_col="alt", identifier_col="identifier")
    collection = TrajectoryCollection()
    collection.load_from_file(df, config=config, min_points=2)
    assert len(collection.trajectories) == 2
    identifiers = {t.identifier for t in collection.trajectories}
    assert identifiers == {"A", "B"}

def test_min_points_filter(tmp_path):
    # Two trajectories: only one has enough points
    data = [
        {"lon": 1, "lat": 2, "alt": 10, "identifier": "A"},
        {"lon": 2, "lat": 3, "alt": 11, "identifier": "A"},
        {"lon": 3, "lat": 4, "alt": 12, "identifier": "B"}
    ]
    df = pd.DataFrame(data)
    path = tmp_path / "test.csv"
    df.to_csv(path, index=False)
    config = ColumnConfig.create_geo(lon_col="lon", lat_col="lat", alt_col="alt", identifier_col="identifier")
    collection = TrajectoryCollection()
    collection.load_from_file(str(path), config=config, min_points=2)
    # Only A should be included
    assert len(collection.trajectories) == 1
    assert collection.trajectories[0].identifier == "A"

def test_normalization_is_correct():
    # Check that normalization shifts the start point to (0, 0, 0)
    data = [
        {"lon": 0, "lat": 0, "alt": 0, "identifier": "T"},
        {"lon": 1, "lat": 1, "alt": 10, "identifier": "T"},
    ]
    df = pd.DataFrame(data)
    config = ColumnConfig.create_geo(lon_col="lon", lat_col="lat", alt_col="alt", identifier_col="identifier")
    collection = TrajectoryCollection()
    collection.load_from_file(df, config=config, min_points=2)
    traj = collection.trajectories[0]
    first_point = traj.points[0]
    assert first_point.x == 0.0
    assert first_point.y == 0.0
    assert first_point.z == 0.0
