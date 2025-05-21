import logging
from typing import Dict
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

def generate_features(df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """Generates features based on transformations defined in config."""
    try:
        df = df.copy()
        for name, params in config.get("calculate_norm_range", {}).items():
            df[name] = (df[params["max_col"]] - df[params["min_col"]]) / df[params["mean_col"]]
        for name, params in config.get("calculate_difference", {}).items():
            df[name] = df[params["max_col"]] - df[params["min_col"]]
        for name, col in config.get("log_transform", {}).items():
            df[name] = np.log(df[col] + 1e-8)
        for name, params in config.get("multiply", {}).items():
            df[name] = df[params["col_a"]] * df[params["col_b"]]
        logger.info("Feature generation complete with columns: %s", df.columns.tolist())
        return df
    except Exception as e:
        logger.error("Error generating features: %s", e)
        raise
