import numpy as np
from model import (
    LinearRegressionNormalEquation,
    LinearRegressionGradientDescent,
    LinearRegressionSklearn
)
from logger import get_logger
from config import LEARNING_RATE, N_ITERATIONS

logger = get_logger(__name__)


def train_all_models(
    X_train: np.ndarray,
    y_train: np.ndarray
) -> dict:
    logger.info("Starting training of all models")

    # normal equation
    ne_model = LinearRegressionNormalEquation()
    ne_model.fit(X_train, y_train)

    # gradient descent
    gd_model = LinearRegressionGradientDescent(
        lr=LEARNING_RATE,
        n_iterations=N_ITERATIONS
    )
    gd_model.fit(X_train, y_train)

    # sklearn
    sk_model = LinearRegressionSklearn()
    sk_model.fit(X_train, y_train)

    models = {
        'normal_equation' : ne_model,
        'gradient_descent': gd_model,
        'sklearn'         : sk_model
    }

    logger.info("All models trained successfully")
    return models


def get_predictions(
    models: dict,
    X_test: np.ndarray
) -> dict:
    logger.info("Generating predictions for all models")

    predictions = {
        name: model.predict(X_test)
        for name, model in models.items()
    }

    logger.info("Predictions generated successfully")
    return predictions