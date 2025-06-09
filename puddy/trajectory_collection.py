from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, TextIO, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa

from puddy.readers import read_data


class TrajectoryType(Enum):
    GEOGRAPHIC = "geo"
    CARTESIAN = "xyz"
    OTHER = "other"


@dataclass
class ColumnConfig:
    x_col: str
    y_col: str
    z_col: str
    identifier_col: Optional[str] = None
    trajectory_type: TrajectoryType = TrajectoryType.OTHER

    @classmethod
    def create_geo(
        cls,
        lon_col: str = "lon",
        lat_col: str = "lat",
        alt_col: str = "alt",
        identifier_col: Optional[str] = None,
    ) -> "ColumnConfig":
        return cls(lon_col, lat_col, alt_col, identifier_col, TrajectoryType.GEOGRAPHIC)

    @classmethod
    def create_xyz(
        cls,
        x_col: str = "x",
        y_col: str = "y",
        z_col: str = "z",
        identifier_col: Optional[str] = None,
    ) -> "ColumnConfig":
        return cls(x_col, y_col, z_col, identifier_col, TrajectoryType.CARTESIAN)


@dataclass
class NormalizedPoint:
    x: float
    y: float
    z: float


@dataclass
class NormalizedTrajectory:
    points: List[NormalizedPoint]
    identifier: str = "ungrouped"

    @classmethod
    def from_df_group(
        cls, df: pd.DataFrame, config: ColumnConfig
    ) -> "NormalizedTrajectory":
        df = df.copy()
        df[config.x_col] = pd.to_numeric(df[config.x_col], errors="raise")
        df[config.y_col] = pd.to_numeric(df[config.y_col], errors="raise")
        df[config.z_col] = pd.to_numeric(df[config.z_col], errors="raise")

        if config.trajectory_type == TrajectoryType.GEOGRAPHIC:
            ref_lat: float = df[config.y_col].iloc[0]
            ref_lon: float = df[config.x_col].iloc[0]

            x = (df[config.x_col] - ref_lon) * 111320 * np.cos(ref_lat * np.pi / 180)
            y = (df[config.y_col] - ref_lat) * 110574
            z = df[config.z_col] - df[config.z_col].iloc[0]
        else:
            x = df[config.x_col] - df[config.x_col].iloc[0]
            y = df[config.y_col] - df[config.y_col].iloc[0]
            z = df[config.z_col] - df[config.z_col].iloc[0]

        normalized_points: List[NormalizedPoint] = [
            NormalizedPoint(float(x_), float(y_), float(z_))
            for x_, y_, z_ in zip(x, y, z)
        ]
        identifier: str = (
            str(df[config.identifier_col].iloc[0])
            if config.identifier_col
            else "ungrouped"
        )
        return cls(normalized_points, identifier)


class TrajectoryCollection:
    def __init__(self) -> None:
        self.trajectories: List[NormalizedTrajectory] = []
        self.config: Optional[ColumnConfig] = None

    def load_from_file(
        self,
        source: Union[
            str, TextIO, pd.DataFrame, pa.Table
        ],  # str, file-like, DataFrame, etc.
        config: Optional[ColumnConfig] = None,
        min_points: int = 20,
    ) -> None:
        if config is None:
            config = ColumnConfig.create_geo(identifier_col="identifier")
        self.config = config

        records = read_data(source)
        df = pd.DataFrame(records)

        self.trajectories = []
        if config.identifier_col:
            groups = (
                df.groupby(config.identifier_col)
                .filter(lambda x: len(x) >= min_points)
                .groupby(config.identifier_col)
            )
            for _, group in groups:
                normalized_traj = NormalizedTrajectory.from_df_group(group, config)
                self.trajectories.append(normalized_traj)
        else:
            if len(df) >= min_points:
                normalized_traj = NormalizedTrajectory.from_df_group(df, config)
                self.trajectories.append(normalized_traj)

        print(f"Loaded {len(self.trajectories)} trajectories")

    def visualize_sample(self, n: int = 5) -> None:
        plt.figure(figsize=(10, 10))
        ax = plt.subplot(111, projection="3d")
        if len(self.trajectories) == 0:
            print("No trajectories to plot.")
            return

        sample: List[NormalizedTrajectory] = (
            self.trajectories
            if len(self.trajectories) <= n
            else list(np.random.choice(self.trajectories, n, replace=False))  # type: ignore
        )
        for traj in sample:
            points = np.array([[p.x, p.y, p.z] for p in traj.points])
            ax.plot(points[:, 0], points[:, 1], points[:, 2], label=traj.identifier)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")  # type: ignore
        plt.legend()
        plt.show()
