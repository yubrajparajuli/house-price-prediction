import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from logger import get_logger
from config import (
    DROP_COLS, MULTICOL_COLS, FEATURE_COLS,
    TEST_SIZE, RANDOM_STATE
)

logger = get_logger(__name__)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Starting data cleaning")

    # drop unnecessary columns
    df = df.drop(columns=DROP_COLS)
    logger.info(f"Dropped columns: {DROP_COLS}")

    # extract date features
    df['sale_year']  = df['date'].str[:4].astype(int)
    df['sale_month'] = df['date'].str[4:6].astype(int)
    df = df.drop(columns=['date'])
    logger.info("Extracted sale_year and sale_month from date")

    # remove bedrooms outlier
    df = df[df['bedrooms'] != 33]
    logger.info("Removed bedrooms == 33 outlier")

    # engineer house_age and is_renovated
    df['house_age']    = 2015 - df['yr_built']
    df['is_renovated'] = (df['yr_renovated'] != 0).astype(int)
    df = df.drop(columns=['yr_built', 'yr_renovated'])
    logger.info("Engineered house_age and is_renovated")

    # clean column names
    df.columns = df.columns.str.lower().str.strip().str.replace(' ', '_')
    logger.info("Cleaned column names")

    logger.info(f"Cleaning done — shape: {df.shape}")
    return df


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Starting feature engineering")

    # drop multicollinear columns
    df = df.drop(columns=MULTICOL_COLS)
    logger.info(f"Dropped multicollinear columns: {MULTICOL_COLS}")

    # new features
    df['total_rooms']       = df['bedrooms'] + df['bathrooms']
    df['living_area_ratio'] = df['sqft_living'] / df['sqft_living15']
    logger.info("Created total_rooms and living_area_ratio")

    # log price target
    df['log_price'] = np.log1p(df['price'])
    logger.info("Created log_price target")

    logger.info(f"Feature engineering done — shape: {df.shape}")
    return df


def split_data(df: pd.DataFrame):
    logger.info("Splitting data into train and test sets")

    X = df[FEATURE_COLS].values
    y = df['log_price'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    logger.info(f"X_train: {X_train.shape} | X_test: {X_test.shape}")
    return X_train, X_test, y_train, y_test


def standardize(X_train: np.ndarray, X_test: np.ndarray):
    logger.info("Standardizing features from scratch")

    train_mean = np.mean(X_train, axis=0)
    train_std  = np.std(X_train, axis=0)

    X_train_scaled = (X_train - train_mean) / train_std
    X_test_scaled  = (X_test  - train_mean) / train_std

    logger.info("Standardization done")
    return X_train_scaled, X_test_scaled, train_mean, train_std

def add_bias(X_train: np.ndarray, X_test: np.ndarray):
    X_train_final = np.c_[np.ones(X_train.shape[0]), X_train]
    X_test_final  = np.c_[np.ones(X_test.shape[0]),  X_test]
    logger.info(f"Bias added — X_train: {X_train_final.shape} | X_test: {X_test_final.shape}")
    return X_train_final, X_test_final