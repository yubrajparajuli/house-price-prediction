import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from logger import get_logger
from config import SRC_FIGURES_DIR as FIGURES_DIR
import os

logger = get_logger(__name__)

os.makedirs(FIGURES_DIR, exist_ok=True)


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.mean((y_true - y_pred) ** 2)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.sqrt(mse(y_true, y_pred))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.mean(np.abs(y_true - y_pred))


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)


def evaluate_all(
    predictions: dict,
    y_test: np.ndarray
) -> pd.DataFrame:
    logger.info("Evaluating all models")

    results = {}
    for name, y_pred in predictions.items():
        results[name] = {
            'MSE' : round(mse(y_test, y_pred), 6),
            'RMSE': round(rmse(y_test, y_pred), 6),
            'MAE' : round(mae(y_test, y_pred), 6),
            'R2'  : round(r2(y_test, y_pred), 6)
        }

    results_df = pd.DataFrame(results).T
    logger.info(f"\n{results_df}")
    return results_df


def plot_actual_vs_predicted(
    y_test: np.ndarray,
    predictions: dict
) -> None:
    logger.info("Plotting actual vs predicted")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    colors = ['steelblue', 'salmon', 'green']

    for ax, (name, y_pred), color in zip(axes, predictions.items(), colors):
        ax.scatter(y_test, y_pred, alpha=0.3, s=10, color=color)
        ax.plot([y_test.min(), y_test.max()],
                [y_test.min(), y_test.max()], 'r--', lw=2)
        ax.set_title(f'{name} — Actual vs Predicted')
        ax.set_xlabel('Actual Log Price')
        ax.set_ylabel('Predicted Log Price')

    plt.tight_layout()
    plt.savefig(f'{FIGURES_DIR}/actual_vs_predicted.png', dpi=150, bbox_inches='tight')
    plt.show()
    logger.info("Saved actual_vs_predicted.png")


def plot_residuals(
    y_test: np.ndarray,
    predictions: dict
) -> None:
    logger.info("Plotting residuals")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    colors = ['steelblue', 'salmon', 'green']

    for ax, (name, y_pred), color in zip(axes, predictions.items(), colors):
        residuals = y_test - y_pred
        ax.scatter(y_pred, residuals, alpha=0.3, s=10, color=color)
        ax.axhline(y=0, color='red', linestyle='--', lw=2)
        ax.set_title(f'{name} — Residuals')
        ax.set_xlabel('Predicted Log Price')
        ax.set_ylabel('Residuals')

    plt.tight_layout()
    plt.savefig(f'{FIGURES_DIR}/residuals.png', dpi=150, bbox_inches='tight')
    plt.show()
    logger.info("Saved residuals.png")


def plot_loss_curve(loss_history: list) -> None:
    logger.info("Plotting loss curve")

    plt.figure(figsize=(10, 5))
    plt.plot(loss_history, color='steelblue')
    plt.title('Gradient Descent Loss Curve')
    plt.xlabel('Iteration')
    plt.ylabel('Loss (MSE)')
    plt.tight_layout()
    plt.savefig(f'{FIGURES_DIR}/loss_curve.png', dpi=150, bbox_inches='tight')
    plt.show()
    logger.info("Saved loss_curve.png")


def plot_feature_importance(
    weights: np.ndarray,
    feature_names: list
) -> None:
    logger.info("Plotting feature importance")

    coef_df = pd.DataFrame({
        'feature'    : feature_names,
        'coefficient': weights[1:]
    })
    coef_df = coef_df.reindex(
        coef_df['coefficient'].abs().sort_values(ascending=False).index
    )

    plt.figure(figsize=(12, 7))
    colors = ['steelblue' if c > 0 else 'salmon' for c in coef_df['coefficient']]
    plt.barh(coef_df['feature'], coef_df['coefficient'], color=colors)
    plt.axvline(x=0, color='black', linestyle='--', lw=1)
    plt.title('Feature Importance — Model Coefficients')
    plt.xlabel('Coefficient Value')
    plt.tight_layout()
    plt.savefig(f'{FIGURES_DIR}/feature_importance.png', dpi=150, bbox_inches='tight')
    plt.show()
    logger.info("Saved feature_importance.png")