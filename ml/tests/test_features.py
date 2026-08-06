import numpy as np
import pandas as pd
from ml.src.features.feature_pipeline import FeatureEngineeringPipeline
from ml.src.features.weather_features import WeatherFeatureExtractor
from ml.src.features.utils import deg_to_rad, create_lags, create_rolling_stats


def test_feature_pipeline_output():
    pipeline = FeatureEngineeringPipeline()
    out_path = pipeline.run()
    assert out_path.exists()

    df = pd.read_csv(out_path)
    assert len(df) == 731
    assert df.isnull().sum().sum() == 0, "Feature dataset contains NaNs!"
    assert "pm25" in df.columns
    assert "wind_x" in df.columns
    assert "wind_y" in df.columns
    assert "pm25_lag_1d" in df.columns
    assert "pm25_roll_mean_7d" in df.columns


def test_wind_vector_calculations():
    df_raw = pd.DataFrame({
        "date": ["2023-01-01", "2023-01-02"],
        "wind_speed_kmh": [10.0, 20.0],
        "wind_direction_deg": [0.0, 90.0],
        "temperature_c": [20.0, 25.0],
        "humidity_pct": [50.0, 60.0],
        "precipitation_mm": [0.0, 5.0],
        "pressure_hpa": [1013.25, 1010.0]
    })
    extractor = WeatherFeatureExtractor()
    df_transformed = extractor.transform(df_raw)

    # Wind 0 deg (North): wind_x = -10*sin(0) = 0, wind_y = -10*cos(0) = -10
    assert np.isclose(df_transformed.loc[0, "wind_x"], 0.0, atol=1e-5)
    assert np.isclose(df_transformed.loc[0, "wind_y"], -10.0, atol=1e-5)

    # Wind 90 deg (East): wind_x = -20*sin(90) = -20, wind_y = -20*cos(90) = 0
    assert np.isclose(df_transformed.loc[1, "wind_x"], -20.0, atol=1e-5)
    assert np.isclose(df_transformed.loc[1, "wind_y"], 0.0, atol=1e-5)


def test_lags_and_rolling_no_future_leakage():
    df_raw = pd.DataFrame({
        "date": ["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04"],
        "pm25": [100.0, 200.0, 300.0, 400.0]
    })

    df_lag = create_lags(df_raw, cols=["pm25"], lags=[1, 2])
    assert pd.isna(df_lag.loc[0, "pm25_lag_1d"])
    assert df_lag.loc[1, "pm25_lag_1d"] == 100.0
    assert df_lag.loc[2, "pm25_lag_1d"] == 200.0

    df_roll = create_rolling_stats(df_raw, cols=["pm25"], windows=[2], funcs=["mean"])
    # 2023-01-03 roll mean (window=2 shifted by 1) should be mean([100, 200]) = 150.0 (NO current day 300 leakage!)
    assert df_roll.loc[2, "pm25_roll_mean_2d"] == 150.0
