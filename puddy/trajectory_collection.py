from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, TextIO, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa

from puddy.readers import read_data


class TrajectoryType(Enum):
    """Type of coordinate system for the trajectory."""

    GEOGRAPHIC = "geo"
    CARTESIAN = "xyz"
    OTHER = "other"


@dataclass
class ColumnConfig:
    """
    Configuration for identifying columns in the input data.

    Attributes:
        x_col: Name of the column containing X or longitude values.
        y_col: Name of the column containing Y or latitude values.
        z_col: Name of the column containing Z or altitude values.
        identifier_col: Optional; name of the trajectory/grouping identifier column.
        trajectory_type: The type of coordinates (geographic or cartesian).
    """

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
        """
        Returns a ColumnConfig for geographic coordinates.

        Args:
            lon_col: Name of longitude column.
            lat_col: Name of latitude column.
            alt_col: Name of altitude column.
            identifier_col: Optional identifier/grouping column.

        Returns:
            ColumnConfig: Configured for geographic data.
        """
        return cls(lon_col, lat_col, alt_col, identifier_col, TrajectoryType.GEOGRAPHIC)

    @classmethod
    def create_xyz(
        cls,
        x_col: str = "x",
        y_col: str = "y",
        z_col: str = "z",
        identifier_col: Optional[str] = None,
    ) -> "ColumnConfig":
        """
        Returns a ColumnConfig for cartesian coordinates.

        Args:
            x_col: Name of X coordinate column.
            y_col: Name of Y coordinate column.
            z_col: Name of Z coordinate column.
            identifier_col: Optional identifier/grouping column.

        Returns:
            ColumnConfig: Configured for cartesian data.
        """
        return cls(x_col, y_col, z_col, identifier_col, TrajectoryType.CARTESIAN)


@dataclass
class NormalizedPoint:
    """
    A single point in a normalized trajectory.

    Attributes:
        x: X coordinate (or normalized longitude).
        y: Y coordinate (or normalized latitude).
        z: Z coordinate (or normalized altitude).
    """

    x: float
    y: float
    z: float


@dataclass
class NormalizedTrajectory:
    """
    A trajectory normalized to a local origin.

    Attributes:
        points: List of normalized points for the trajectory.
        identifier: Trajectory or group identifier.
    """

    points: List[NormalizedPoint]
    identifier: str = "ungrouped"

    @classmethod
    def from_df_group(
        cls, df: pd.DataFrame, config: ColumnConfig
    ) -> "NormalizedTrajectory":
        """
        Create a normalized trajectory from a DataFrame group and column config.

        Args:
            df: DataFrame with columns for coordinates and optional identifier.
            config: ColumnConfig for extracting and normalizing coordinates.

        Returns:
            NormalizedTrajectory: The normalized trajectory object.
        """
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
    """
    A collection of normalized trajectories, with utilities for loading and visualization.
    """

    def __init__(self) -> None:
        """
        Initialize an empty collection.
        """
        self.trajectories: List[NormalizedTrajectory] = []
        self.config: Optional[ColumnConfig] = None

    def load_from_file(
        self,
        source: Union[str, TextIO, pd.DataFrame, pa.Table],
        config: Optional[ColumnConfig] = None,
        min_points: int = 20,
    ) -> None:
        """
        Loads trajectory data from any supported file or in-memory object,
        normalizes each group, and stores the results.

        Args:
            source: File path, file-like object, DataFrame, or Arrow Table.
            config: ColumnConfig specifying data columns and type.
            min_points: Minimum number of points required per trajectory/group.
        """
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
        """
        Plots a random sample of trajectories in 3D.

        Args:
            n: Number of trajectories to visualize (or all if fewer available).
        """
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
