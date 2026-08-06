#!/usr/bin/env python3
"""
Launcher script for atmosIQ Phase 2 Feature Engineering Pipeline.
"""
from ml.src.features.feature_pipeline import FeatureEngineeringPipeline

if __name__ == "__main__":
    pipeline = FeatureEngineeringPipeline()
    pipeline.run()
