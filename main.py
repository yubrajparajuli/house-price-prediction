import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_loader import load_data, data_info
from preprocessor import clean_data, feature_engineering, split_data, standardize, add_bias
from eda import run_eda
from trainer import train_all_models, get_predictions
from evaluator import (
    evaluate_all,
    plot_actual_vs_predicted,
    plot_residuals,
    plot_loss_curve,
    plot_feature_importance
)
from utils import save_all_models
from logger import get_logger
from config import DATA_FILE, FEATURE_COLS

logger = get_logger(__name__)


def main():
    logger.info("Starting House Price Prediction Pipeline")

    # step 1: load data
    logger.info("Step 1: Loading Data")
    df = load_data(DATA_FILE)
    data_info(df)

    # step 2: clean data
    logger.info("Step 2: Cleaning Data")
    df = clean_data(df)

    # step 3: eda
    logger.info("Step 3: Running EDA")
    run_eda(df)

    # step 4: feature engineering
    logger.info("Step 4: Feature Engineering")
    df = feature_engineering(df)

    # step 5: split data
    logger.info("Step 5: Splitting Data")
    X_train, X_test, y_train, y_test = split_data(df)

    # step 6: standardize
    logger.info("Step 6: Standardizing Features")
    X_train_scaled, X_test_scaled, train_mean, train_std = standardize(X_train, X_test)

    # step 7: add bias
    logger.info("Step 7: Adding Bias Term")
    X_train_final, X_test_final = add_bias(X_train_scaled, X_test_scaled)

    # step 8: train models
    logger.info("Step 8: Training Models")
    models = train_all_models(X_train_final, y_train)

    # step 9: get predictions
    logger.info("Step 9: Generating Predictions")
    predictions = get_predictions(models, X_test_final)

    # step 10: evaluate
    logger.info("Step 10: Evaluating Models")
    results_df = evaluate_all(predictions, y_test)
    print("\nModel Comparison:")
    print(results_df)

    # step 11: plots
    logger.info("Step 11: Generating Plots")
    plot_actual_vs_predicted(y_test, predictions)
    plot_residuals(y_test, predictions)
    plot_loss_curve(models['gradient_descent'].loss_history)
    plot_feature_importance(
        models['normal_equation'].weights,
        FEATURE_COLS
    )

    # step 12: save models
    logger.info("Step 12: Saving Models")
    save_all_models(models, train_mean, train_std)

    logger.info("Pipeline Complete")


if __name__ == '__main__':
    main()