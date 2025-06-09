from puddy.trajectory_collection import TrajectoryCollection, ColumnConfig
from puddy.trajectory_analyzer import TrajectoryAnalyzer, visualize_trajectories_sample
import matplotlib.pyplot as plt
import numpy as np

def main() -> None:
    # Adjust column names to match your CSV headers
    geo_config: ColumnConfig = ColumnConfig.create_geo(
        lon_col='lon',
        lat_col='lat',
        alt_col='alt',
        identifier_col='identifier'
    )
    collection = TrajectoryCollection()
    collection.load_from_file('somegeodata.csv', config=geo_config)
    collection.visualize_sample(5)

    analyzer = TrajectoryAnalyzer(collection)
    analyzer.train_anomaly_detector(method='isolation_forest')
    scores: np.ndarray = analyzer.get_normalcy_scores()
    # You can access anomalies here if you want to print or process them, otherwise omit
    # anomalies = analyzer.find_anomalies(threshold=0.2)
    df = analyzer.get_normalcy_df()

    print(df.head())

    visualize_trajectories_sample(
        analyzer.collection.trajectories,
        scores,
        normal_sample=20,
        show_all_anomalies=True
    )
    plt.show()

if __name__ == "__main__":
    main()
