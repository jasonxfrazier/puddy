# Puddy

**Puddy** is a Python toolkit for analyzing spatial trajectories from files in CSV, JSON, JSONL, or Parquet format.
It provides easy loading, normalization, feature extraction, anomaly detection, and rich 3D visualization for geospatial or cartesian trajectory data.

---

## Installation

Install via pip:

```sh
pip install puddy
```

Or for development (to auto-update as you change source):

```sh
pip install -e .
```

**PyPI:** [https://pypi.org/project/puddy/](https://pypi.org/project/puddy/)  

---

## What does Puddy do?

- **Load trajectories** from CSV, JSON, JSONL, Parquet/Arrow, or in-memory DataFrames
- **Normalize** and group trajectory data using customizable column configs
- **Extract features** automatically (distance, bounding box, linearity, turns, aspect ratio, etc)
- **Detect anomalies** using Isolation Forest or Local Outlier Factor
- **Visualize** trajectories in interactive 3D plots colored by "normalcy"

---

## Basic Usage

```python
from puddy import TrajectoryCollection, ColumnConfig, TrajectoryAnalyzer, visualize_trajectories_sample

# 1. Configure which columns to use for each axis and group
config = ColumnConfig.create_geo(
    lon_col="lon",
    lat_col="lat",
    alt_col="alt",
    identifier_col="id"
)

# 2. Load your data (CSV, JSONL, Parquet, DataFrame, etc.)
collection = TrajectoryCollection()
collection.load_from_file("trajectories.csv", config=config)

# 3. Visualize a sample
collection.visualize_sample(n=5)

# 4. Analyze features and anomalies
analyzer = TrajectoryAnalyzer(collection)
analyzer.train_anomaly_detector(method="isolation_forest")
scores = analyzer.get_normalcy_scores()
df = analyzer.get_normalcy_df()
print(df.head())

# 5. Visualize by normalcy
visualize_trajectories_sample(
    analyzer.collection.trajectories,
    scores,
    normal_sample=10,
    show_all_anomalies=True
)
```

---

## Features

- Robust file loading: works with large datasets, compressed files, and many formats
- Customizable coordinate systems and grouping
- Automatic feature engineering for trajectory shape/motion
- Built-in anomaly detection and scoring
- Publication-quality 3D plotting

---

## Documentation

For advanced usage, custom features, or troubleshooting, see the [Wiki](https://github.com/jasonxfrazier/puddy/wiki).

---

## License

MIT © Jason Frazier

