import numpy as np
import pandas as pd
from puddy.trajectory_collection import TrajectoryCollection, ColumnConfig
from puddy.trajectory_analyzer import TrajectoryAnalyzer

def make_collection():
    data = [
        {"lon": 10, "lat": 10, "alt": 100, "identifier": "A"},
        {"lon": 11, "lat": 11, "alt": 110, "identifier": "A"},
        {"lon": 20, "lat": 20, "alt": 200, "identifier": "B"},
        {"lon": 21, "lat": 21, "alt": 210, "identifier": "B"},
    ]
    df = pd.DataFrame(data)
    config = ColumnConfig.create_geo(lon_col="lon", lat_col="lat", alt_col="alt", identifier_col="identifier")
    collection = TrajectoryCollection()
    collection.load_from_file(df, config=config, min_points=2)
    return collection

def test_feature_extraction():
    collection = make_collection()
    analyzer = TrajectoryAnalyzer(collection)
    features = analyzer.prepare_features()
    assert features.shape[0] == len(collection.trajectories)
    # Check some feature value range
    assert np.all(np.isfinite(features))

def test_anomaly_detector_and_scores():
    collection = make_collection()
    analyzer = TrajectoryAnalyzer(collection)
    analyzer.train_anomaly_detector(method="isolation_forest")
    scores = analyzer.get_normalcy_scores()
    assert len(scores) == len(collection.trajectories)
    # Should be a numpy array of floats
    assert isinstance(scores, np.ndarray)

def test_normalcy_dataframe():
    collection = make_collection()
    analyzer = TrajectoryAnalyzer(collection)
    analyzer.train_anomaly_detector()
    df = analyzer.get_normalcy_df()
    assert "normalcy_score" in df.columns
    assert df.shape[0] == len(collection.trajectories)

def test_find_anomalies_returns_subset():
    collection = make_collection()
    analyzer = TrajectoryAnalyzer(collection)
    analyzer.train_anomaly_detector()
    anomalies = analyzer.find_anomalies(threshold=0.5)
    assert isinstance(anomalies, list)
    for traj, score in anomalies:
        assert hasattr(traj, "identifier")
        assert isinstance(score, float) or isinstance(score, np.floating)
