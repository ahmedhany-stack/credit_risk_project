import sys
import numpy as np
import pandas as pd

from src.components.logger import logger
from src.components.constants import DATA_PATH
from src.exception import CustomException


def load_data(path: str) -> pd.DataFrame:
    """
    Load dataset from CSV file.
    """
    try:
        logger.info(f"Loading dataset from: {path}")

        df = pd.read_csv(path)

        logger.info(f"Dataset loaded successfully. Shape: {df.shape}")

        return df

    except Exception as e:
        logger.exception("Failed to load dataset.")
        raise CustomException(e, sys)


def drop_columns(
    df: pd.DataFrame,
    columns=("customer_id", "credit_risk")
) -> pd.DataFrame:
    """
    Drop unnecessary columns.
    """
    try:
        logger.info(f"Dropping columns: {list(columns)}")

        existing_columns = [col for col in columns if col in df.columns]

        df = df.drop(columns=existing_columns)

        logger.info("Columns dropped successfully.")

        return df

    except Exception as e:
        logger.exception("Failed to drop unnecessary columns.")
        raise CustomException(e, sys)


def get_numeric_columns(df: pd.DataFrame) -> list:
    """
    Return numerical columns only.
    """
    try:
        numeric_columns = df.select_dtypes(include=np.number).columns.tolist()

        logger.info(f"Found {len(numeric_columns)} numerical columns.")

        return numeric_columns

    except Exception as e:
        logger.exception("Failed to retrieve numerical columns.")
        raise CustomException(e, sys)


def handle_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle outliers using IQR Capping.
    Binary columns are ignored.
    """
    try:

        logger.info("Starting outlier treatment...")

        numeric_columns = get_numeric_columns(df)

        for column in numeric_columns:

            # Skip binary columns
            if df[column].nunique() <= 2:
                logger.info(f"Skipping binary column: {column}")
                continue

            q1 = df[column].quantile(0.25)
            q3 = df[column].quantile(0.75)

            iqr = q3 - q1

            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr

            outliers_before = (
                (df[column] < lower) |
                (df[column] > upper)
            ).sum()

            df[column] = np.clip(df[column], lower, upper)

            logger.info(
                f"{column}: {outliers_before} outliers capped."
            )

        logger.info("Outlier treatment completed successfully.")

        return df

    except Exception as e:
        logger.exception("Failed during outlier treatment.")
        raise CustomException(e, sys)


def calculate_skewness(df: pd.DataFrame) -> pd.Series:
    """
    Calculate skewness for numerical features.
    Binary columns are ignored.
    """
    try:

        logger.info("Calculating skewness...")

        numeric_columns = [
            col
            for col in get_numeric_columns(df)
            if df[col].nunique() > 2
        ]

        skewness = (
            df[numeric_columns]
            .skew()
            .sort_values(ascending=False)
        )

        logger.info("Skewness calculated successfully.")

        return skewness

    except Exception as e:
        logger.exception("Failed to calculate skewness.")
        raise CustomException(e, sys)


def validate_dataset(path: str):
    """
    Complete validation pipeline.
    """
    try:

        logger.info("=" * 60)
        logger.info("Validation pipeline started.")
        logger.info("=" * 60)

        df = load_data(path)

        df = drop_columns(df)

        df = handle_outliers(df)

        skewness = calculate_skewness(df)

        logger.info("Validation pipeline completed successfully.")
        logger.info("=" * 60)

        return df, skewness

    except Exception as e:
        logger.exception("Validation pipeline failed.")
        raise CustomException(e, sys)


if __name__ == "__main__":

    df, skewness = validate_dataset(DATA_PATH)

    print(skewness)