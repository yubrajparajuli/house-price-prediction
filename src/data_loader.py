import pandas as pd
from logger import get_logger
from config import DATA_FILE

logger = get_logger(__name__)


def load_data(filepath: str = DATA_FILE) -> pd.DataFrame:
    logger.info(f"Loading data from {filepath}")
    df = pd.read_csv(filepath)
    logger.info(f"Data loaded successfully — shape: {df.shape}")
    return df


def data_info(df: pd.DataFrame) -> None:
    logger.info("Dataset Info")
    print(f"Shape             : {df.shape}")
    print(f"\nData Types:\n{df.dtypes}")
    print(f"\nMissing Values:\n{df.isnull().sum()}")
    print(f"\nDuplicate Rows    : {df.duplicated().sum()}")
    print(f"\nUnique Values per Column:")
    for col in df.columns:
        print(f"  {col}: {df[col].nunique()} unique")
    print(f"\nBasic Statistics:\n{df.describe()}")