import pickle
import numpy as np
import pandas as pd
from logger import get_logger
from config import NE_MODEL_PATH, GD_MODEL_PATH, SK_MODEL_PATH, FEATURE_COLS
import os


logger = get_logger(__name__)


def save_model(model_data: dict, filepath: str) -> None:
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(model_data, f)
    logger.info(f"Model saved to {filepath}")


def load_model(filepath: str) -> dict:
    with open(filepath, 'rb') as f:
        model_data = pickle.load(f)
    logger.info(f"Model loaded from {filepath}")
    return model_data


def save_all_models(models: dict, train_mean: np.ndarray, train_std: np.ndarray) -> None:
    logger.info("Saving all models")

    ne_data = {
        'weights'     : models['normal_equation'].weights,
        'train_mean'  : train_mean,
        'train_std'   : train_std,
        'features'    : FEATURE_COLS
    }
    save_model(ne_data, NE_MODEL_PATH)

    gd_data = {
        'weights'     : models['gradient_descent'].weights,
        'train_mean'  : train_mean,
        'train_std'   : train_std,
        'features'    : FEATURE_COLS,
        'loss_history': models['gradient_descent'].loss_history
    }
    save_model(gd_data, GD_MODEL_PATH)

    sk_data = {
        'model'       : models['sklearn'].model,
        'train_mean'  : train_mean,
        'train_std'   : train_std,
        'features'    : FEATURE_COLS
    }
    save_model(sk_data, SK_MODEL_PATH)

    logger.info("All models saved successfully")


def predict_price(input_dict: dict, model_data: dict) -> float:
    input_df = pd.DataFrame([input_dict])
    input_df = input_df[model_data['features']]

    input_scaled = (input_df.values - model_data['train_mean']) / model_data['train_std']
    input_final  = np.c_[np.ones(1), input_scaled]

    log_price    = input_final @ model_data['weights']
    actual_price = np.expm1(log_price)

    logger.info(f"Predicted price: ${actual_price[0]:,.0f}")
    return actual_price[0]