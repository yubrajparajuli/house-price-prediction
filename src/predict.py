import numpy as np
import pandas as pd
from utils import load_model
from logger import get_logger
from config import NE_MODEL_PATH, GD_MODEL_PATH, SK_MODEL_PATH, FEATURE_COLS

logger = get_logger(__name__)


def predict_with_ne(input_dict: dict) -> float:
    logger.info("Predicting with Normal Equation model")
    model_data = load_model(NE_MODEL_PATH)
    price = _predict_scratch(input_dict, model_data)
    logger.info(f"Normal Equation prediction: ${price:,.0f}")
    return price


def predict_with_gd(input_dict: dict) -> float:
    logger.info("Predicting with Gradient Descent model")
    model_data = load_model(GD_MODEL_PATH)
    price = _predict_scratch(input_dict, model_data)
    logger.info(f"Gradient Descent prediction: ${price:,.0f}")
    return price


def predict_with_sklearn(input_dict: dict) -> float:
    logger.info("Predicting with Sklearn model")
    model_data = load_model(SK_MODEL_PATH)
    price = _predict_sklearn(input_dict, model_data)
    logger.info(f"Sklearn prediction: ${price:,.0f}")
    return price


def predict_all(input_dict: dict) -> dict:
    logger.info("Predicting with all models")

    results = {
        'normal_equation' : round(predict_with_ne(input_dict), 2),
        'gradient_descent': round(predict_with_gd(input_dict), 2),
        'sklearn'         : round(predict_with_sklearn(input_dict), 2)
    }

    logger.info(f"Predictions: {results}")
    return results


def _predict_scratch(input_dict: dict, model_data: dict) -> float:
    input_df     = pd.DataFrame([input_dict])[model_data['features']]
    input_scaled = (input_df.values - model_data['train_mean']) / model_data['train_std']
    input_final  = np.c_[np.ones(1), input_scaled]
    log_price    = input_final @ model_data['weights']
    return float(np.expm1(log_price[0]))


def _predict_sklearn(input_dict: dict, model_data: dict) -> float:
    input_df     = pd.DataFrame([input_dict])[model_data['features']]
    input_scaled = (input_df.values - model_data['train_mean']) / model_data['train_std']
    input_final  = np.c_[np.ones(1), input_scaled]
    log_price    = model_data['model'].predict(input_final)
    return float(np.expm1(log_price[0]))


if __name__ == '__main__':
    sample_house = {
        'bedrooms'         : 3,
        'bathrooms'        : 2.0,
        'sqft_living'      : 1800,
        'sqft_lot'         : 7000,
        'floors'           : 1.0,
        'waterfront'       : 0,
        'view'             : 0,
        'condition'        : 3,
        'grade'            : 7,
        'sqft_basement'    : 0,
        'lat'              : 47.5,
        'long'             : -122.2,
        'sqft_living15'    : 1800,
        'sale_year'        : 2015,
        'sale_month'       : 6,
        'house_age'        : 30,
        'is_renovated'     : 0,
        'total_rooms'      : 5.0,
        'living_area_ratio': 1.0
    }

    results = predict_all(sample_house)
    print(f"\nNormal Equation  : ${results['normal_equation']:,.0f}")
    print(f"Gradient Descent : ${results['gradient_descent']:,.0f}")
    print(f"Sklearn          : ${results['sklearn']:,.0f}")