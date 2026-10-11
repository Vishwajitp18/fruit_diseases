
import numpy as np

from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder
)


# ==========================================
# ENVIRONMENTAL FEATURES
# ==========================================

NUMERICAL_FEATURES = [
    "temperature_c",
    "humidity_pct",
    "rainfall_mm",
    "soil_moisture_pct",
    "soil_ph"
]

CATEGORICAL_FEATURES = [
    "season"
]


# ==========================================
# PREPROCESSORS
# ==========================================

numerical_scaler = StandardScaler()

season_encoder = OneHotEncoder(
    sparse_output=False,
    handle_unknown="ignore"
)


# ==========================================
# FIT PREPROCESSORS
# ==========================================

def fit_preprocessors(train_df):
    """
    Fit the environmental preprocessing objects.

    IMPORTANT:
    Only the training dataset should be used
    to fit the scaler and encoder.
    """

    numerical_scaler.fit(
        train_df[NUMERICAL_FEATURES]
    )

    season_encoder.fit(
        train_df[CATEGORICAL_FEATURES]
    )


# ==========================================
# TRANSFORM ENVIRONMENTAL DATA
# ==========================================

def transform_environmental_data(df):
    """
    Transform environmental data using the
    already-fitted scaler and encoder.
    """

    # Scale numerical features
    numerical_data = numerical_scaler.transform(
        df[NUMERICAL_FEATURES]
    )

    # Encode season
    season_data = season_encoder.transform(
        df[CATEGORICAL_FEATURES]
    )

    # Combine numerical + categorical features
    combined_data = np.concatenate(
        [
            numerical_data,
            season_data
        ],
        axis=1
    )

    return combined_data
