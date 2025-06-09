from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.decomposition import PCA
from TrajectoryCollection import TrajectoryCollection, NormalizedTrajectory
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
import pandas as pd

class TrajectoryAnalyzer:
    def __init__(self, collection: TrajectoryCollection) -> None:
        self.collection: TrajectoryCollection = collection
        self.scaler: StandardScaler = StandardScaler()
        self.features: Optional[np.ndarray] = None
        self.model: Optional[Union[IsolationForest, LocalOutlierFactor]] = None

    def extract_features(self, trajectory: NormalizedTrajectory) -> Dict[str, float]:
        try:
            points = np.array([[p.x, p.y, p.z] for p in trajectory.points])
            features = {
                'total_distance': self._safe_calculate(self._total_distance, points),
                'bounding_box_volume': self._safe_calculate(self._bounding_box_volume, points),
                'mean_altitude': self._safe_calculate(lambda p: np.mean(p[:,2]), points),
                'altitude_range': self._safe_calculate(lambda p: np.ptp(p[:,2]), points),
                'path_linearity': self._safe_calculate(self._calculate_linearity, points),
                'total_turns': self._safe_calculate(self._calculate_turns, points),
                'aspect_ratio': self._safe_calculate(self._calculate_aspect_ratio, points)
            }
            return features
        except Exception as e:
            print(f'Error extracting features for trajectory: {e}')
            return {
                'total_distance': 0.0,
                'bounding_box_volume': 0.0,
                'mean_altitude': 0.0,
                'altitude_range': 0.0,
                'path_linearity': 0.0,
                'total_turns': 0.0,
                'aspect_ratio': 1.0
            }

    def _safe_calculate(self, func, points: np.ndarray) -> float:
        try:
            result = func(points)
            if np.isnan(result) or np.isinf(result):
                return 0.0
            return result
        except Exception as e:
            print(f'Unable to calculate {func}: {e}')
            return 0.0

    def _total_distance(self, points: np.ndarray) -> float:
        if len(points) < 2:
            return 0.0
        diff = np.diff(points, axis=0)
        return float(np.sum(np.linalg.norm(diff, axis=1)))
    
    def _bounding_box_volume(self, points: np.ndarray) -> float:
        if len(points) == 0:
            return 0.0
        ranges = np.ptp(points, axis=0)
        return float(np.prod(ranges)) if np.all(ranges > 0) else 0.0
    
    def _calculate_linearity(self, points: np.ndarray) -> float:
        if len(points) < 2:
            return 0.0
        try:
            pca = PCA(n_components=3)
            pca.fit(points)
            return float(pca.explained_variance_ratio_[0])
        except Exception:
            return 0.0
        
    def _calculate_turns(self, points: np.ndarray) -> float:
        if len(points) < 3:
            return 0.0
        vectors = np.diff(points, axis=0)
        norms = np.linalg.norm(vectors, axis=1)
        if np.any(norms == 0):
            return 0.0
        dot_products = np.sum(vectors[1:] * vectors[:-1], axis=1)
        cos_angles = dot_products / (norms[1:] * norms[:-1])
        cos_angles = np.clip(cos_angles, -1, 1)
        angles = np.arccos(cos_angles)
        return float(np.sum(angles > np.pi/4))
    
    def _calculate_aspect_ratio(self, points: np.ndarray) -> float:
        if len(points) == 0:
            return 1.0
        ranges = np.ptp(points, axis=0)
        min_range = float(np.min(ranges)) if np.min(ranges) > 0 else 1.0
        return float(np.max(ranges) / min_range) if min_range != 0 else 1.0

    def prepare_features(self) -> np.ndarray:
        all_features: List[List[float]] = []
        valid_trajectories: List[NormalizedTrajectory] = []

        for traj in self.collection.trajectories:
            features = self.extract_features(traj)
            feature_values = list(features.values())
            if not any(np.isnan(f) or np.isinf(f) for f in feature_values):
                all_features.append(feature_values)
                valid_trajectories.append(traj)
            
        if not all_features:
            raise ValueError('No valid features could be extracted from trajectories')
        
        self.collection.trajectories = valid_trajectories
        feature_array = np.array(all_features)
        self.features = self.scaler.fit_transform(feature_array)
        return self.features
    
    def train_anomaly_detector(self, method: str = 'isolation_forest') -> None:
        if self.features is None:
            self.prepare_features()

        if method == 'isolation_forest':
            self.model = IsolationForest(contamination=0.1, random_state=42)
            self.model.fit(self.features) # type: ignore
        elif method == 'lof':
            self.model = LocalOutlierFactor(contamination=0.1, novelty=True)
            self.model.fit(self.features) # type: ignore
        else:
            raise ValueError(f"Unknown method: {method}")

    def get_normalcy_scores(self) -> np.ndarray:
        if self.model is None:
            raise ValueError('Must train anomaly detector before getting scores')
        if isinstance(self.model, IsolationForest):
            return self.model.score_samples(self.features) # type: ignore
        else:
            return -self.model.decision_function(self.features) # type: ignore

    def find_anomalies(self, threshold: float = 0.2) -> List[Tuple[NormalizedTrajectory, float]]:
        scores = self.get_normalcy_scores()
        anomalies = [
            (traj, score)
            for traj, score in zip(self.collection.trajectories, scores)
            if score < threshold
        ]
        return anomalies
    
    def get_normalcy_df(self, id_column_name: str = 'identifier') -> pd.DataFrame:
        if self.model is None:
            raise ValueError('Must train anomaly detector before getting scores')
        scores = self.get_normalcy_scores()
        data = {
            id_column_name: [traj.identifier for traj in self.collection.trajectories],
            'normalcy_score': scores
        }
        df = pd.DataFrame(data)
        df = df.sort_values('normalcy_score', ascending=False)
        return df

def visualize_trajectories_sample(
        trajectories: List[NormalizedTrajectory],
        scores: np.ndarray,
        normal_sample: int = 10,
        show_all_anomalies: bool = True,
        threshold: float = 0.2
) -> Tuple[plt.Figure, plt.Axes]: # type: ignore
    plt.figure(figsize=(15, 10))
    ax = plt.subplot(111, projection='3d')
    normal_indices = np.where(scores >= threshold)[0]
    anomaly_indices = np.where(scores < threshold)[0]

    if len(normal_indices) > normal_sample:
        normal_indices = np.random.choice(normal_indices, normal_sample, replace=False)
    if show_all_anomalies:
        plot_indices = np.concatenate([normal_indices, anomaly_indices])
    else:
        anomaly_sample = min(len(anomaly_indices), normal_sample)
        sampled_anomaly_indices = np.random.choice(anomaly_indices, anomaly_sample, replace=False)
        plot_indices = np.concatenate([normal_indices, sampled_anomaly_indices])

    norm = Normalize(scores.min(), scores.max())
    cmap = plt.cm.viridis # type: ignore

    for idx in plot_indices:
        traj = trajectories[idx]
        score = scores[idx]
        points = np.array([[p.x, p.y, p.z] for p in traj.points])
        ax.plot(points[:,0], points[:,1], points[:,2], color=cmap(norm(score)), alpha=0.7, linewidth=2)

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    plt.colorbar(sm, ax=ax, label='Normalcy Score')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')  # type: ignore
    plt.title(f'Trajectories Colored by Normalcy Score\n'
              f'(Showing {len(normal_indices)} normal and {len(anomaly_indices)} anomalous trajectories)')
    ax.view_init(elev=20, azim=45) # type: ignore
    return plt.gcf(), ax
