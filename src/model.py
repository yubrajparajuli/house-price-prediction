import numpy as np
from sklearn.linear_model import LinearRegression as SklearnLR
from logger import get_logger
from config import LEARNING_RATE, N_ITERATIONS

logger = get_logger(__name__)


class LinearRegressionNormalEquation:
    """
    Linear Regression using Normal Equation.
    w = (X^T X)^-1 X^T y
    """

    def __init__(self):
        self.weights = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        logger.info("Training Linear Regression — Normal Equation")
        self.weights = np.linalg.inv(X.T @ X) @ X.T @ y
        logger.info(f"Training done — weights shape: {self.weights.shape}")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return X @ self.weights


class LinearRegressionGradientDescent:
    """
    Linear Regression using Gradient Descent.
    w = w - lr * (1/n) * X^T (Xw - y)
    """

    def __init__(
        self,
        lr: float = LEARNING_RATE,
        n_iterations: int = N_ITERATIONS
    ):
        self.lr           = lr
        self.n_iterations = n_iterations
        self.weights      = None
        self.loss_history = []

    def fit(self, X: np.ndarray, y: np.ndarray):
        logger.info("Training Linear Regression — Gradient Descent")
        logger.info(f"lr: {self.lr} | iterations: {self.n_iterations}")

        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)

        for i in range(self.n_iterations):
            y_pred   = X @ self.weights
            error    = y_pred - y
            gradient = (1 / n_samples) * X.T @ error
            self.weights -= self.lr * gradient

            loss = (1 / (2 * n_samples)) * np.sum(error ** 2)
            self.loss_history.append(loss)

            if i % 100 == 0:
                logger.info(f"Iteration {i:4d} | Loss: {loss:.6f}")

        logger.info("Training done")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return X @ self.weights


class LinearRegressionSklearn:
    """
    Linear Regression using Sklearn for comparison.
    """

    def __init__(self):
        self.model   = SklearnLR()
        self.weights = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        logger.info("Training Linear Regression — Sklearn")
        self.model.fit(X, y)
        self.weights = np.r_[self.model.intercept_, self.model.coef_[1:]]
        logger.info("Training done")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)