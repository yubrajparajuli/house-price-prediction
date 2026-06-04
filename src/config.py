import os

# paths
BASE_DIR        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR        = os.path.join(BASE_DIR, 'data')
MODELS_DIR      = os.path.join(BASE_DIR, 'models')
REPORTS_DIR     = os.path.join(BASE_DIR, 'reports')
FIGURES_DIR     = os.path.join(REPORTS_DIR, 'figures')

# src pipeline outputs — separate from notebook outputs
SRC_MODELS_DIR  = os.path.join(BASE_DIR, 'src_models')
SRC_FIGURES_DIR = os.path.join(BASE_DIR, 'src_reports', 'figures')

# data
DATA_FILE       = os.path.join(DATA_DIR, 'kc_house_data.csv')

# model paths
NE_MODEL_PATH   = os.path.join(SRC_MODELS_DIR, 'ne_model.pkl')
GD_MODEL_PATH   = os.path.join(SRC_MODELS_DIR, 'gd_model.pkl')
SK_MODEL_PATH   = os.path.join(SRC_MODELS_DIR, 'sk_model.pkl')

# preprocessing
TEST_SIZE       = 0.2
RANDOM_STATE    = 42
TARGET_COL      = 'log_price'

# gradient descent hyperparameters
LEARNING_RATE   = 0.01
N_ITERATIONS    = 1000

# features to drop
DROP_COLS       = ['id', 'zipcode']
MULTICOL_COLS   = ['sqft_above', 'sqft_lot15']

# feature columns
FEATURE_COLS    = [
    'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot',
    'floors', 'waterfront', 'view', 'condition', 'grade',
    'sqft_basement', 'lat', 'long', 'sqft_living15',
    'sale_year', 'sale_month', 'house_age', 'is_renovated',
    'total_rooms', 'living_area_ratio'
]