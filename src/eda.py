import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from logger import get_logger
from config import SRC_FIGURES_DIR as FIGURES_DIR

logger = get_logger(__name__)

os.makedirs(FIGURES_DIR, exist_ok=True)


def plot_price_distribution(df: pd.DataFrame) -> None:
    logger.info("Plotting price distribution")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(df['price'], bins=50, color='steelblue', edgecolor='white')
    axes[0].set_title('Price Distribution')
    axes[0].set_xlabel('Price')
    axes[0].set_ylabel('Frequency')

    axes[1].hist(np.log1p(df['price']), bins=50, color='salmon', edgecolor='white')
    axes[1].set_title('Log Price Distribution')
    axes[1].set_xlabel('Log Price')
    axes[1].set_ylabel('Frequency')

    plt.tight_layout()
    plt.savefig(f'{FIGURES_DIR}/price_distribution.png', dpi=150, bbox_inches='tight')
    plt.show()
    logger.info("Saved price_distribution.png")


def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    logger.info("Plotting correlation heatmap")

    plt.figure(figsize=(16, 10))
    corr = df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f',
                cmap='coolwarm', linewidths=0.5)
    plt.title('Correlation Heatmap')
    plt.tight_layout()
    plt.savefig(f'{FIGURES_DIR}/correlation_heatmap.png', dpi=150, bbox_inches='tight')
    plt.show()
    logger.info("Saved correlation_heatmap.png")


def plot_price_vs_features(df: pd.DataFrame) -> None:
    logger.info("Plotting price vs features")

    top_features = ['sqft_living', 'grade', 'sqft_above',
                    'sqft_living15', 'bathrooms', 'bedrooms']

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for i, col in enumerate(top_features):
        axes[i].scatter(df[col], df['price'], alpha=0.3, color='steelblue', s=10)
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('Price')
        axes[i].set_title(f'Price vs {col}')

    plt.tight_layout()
    plt.savefig(f'{FIGURES_DIR}/price_vs_features.png', dpi=150, bbox_inches='tight')
    plt.show()
    logger.info("Saved price_vs_features.png")


def plot_boxplots(df: pd.DataFrame) -> None:
    logger.info("Plotting boxplots")

    cols = ['price', 'sqft_living', 'sqft_lot',
            'bedrooms', 'bathrooms', 'house_age']

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for i, col in enumerate(cols):
        axes[i].boxplot(df[col])
        axes[i].set_title(col)
        axes[i].set_ylabel(col)

    plt.tight_layout()
    plt.savefig(f'{FIGURES_DIR}/boxplots.png', dpi=150, bbox_inches='tight')
    plt.show()
    logger.info("Saved boxplots.png")


def plot_price_by_categories(df: pd.DataFrame) -> None:
    logger.info("Plotting price by categories")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].boxplot([df[df['waterfront']==0]['price'],
                     df[df['waterfront']==1]['price']],
                    labels=['No Waterfront', 'Waterfront'])
    axes[0].set_title('Price by Waterfront')
    axes[0].set_ylabel('Price')

    conditions = sorted(df['condition'].unique())
    axes[1].boxplot([df[df['condition']==c]['price'] for c in conditions],
                    labels=conditions)
    axes[1].set_title('Price by Condition')
    axes[1].set_ylabel('Price')

    grades = sorted(df['grade'].unique())
    axes[2].boxplot([df[df['grade']==g]['price'] for g in grades],
                    labels=grades)
    axes[2].set_title('Price by Grade')
    axes[2].set_ylabel('Price')

    plt.tight_layout()
    plt.savefig(f'{FIGURES_DIR}/price_by_categories.png', dpi=150, bbox_inches='tight')
    plt.show()
    logger.info("Saved price_by_categories.png")


def plot_geographic_map(df: pd.DataFrame) -> None:
    logger.info("Plotting geographic price map")

    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(df['long'], df['lat'],
                          c=df['price'],
                          cmap='RdYlGn_r',
                          alpha=0.4,
                          s=5)
    plt.colorbar(scatter, label='Price')
    plt.title('Geographic Price Map — King County Seattle')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.tight_layout()
    plt.savefig(f'{FIGURES_DIR}/geographic_price_map.png', dpi=150, bbox_inches='tight')
    plt.show()
    logger.info("Saved geographic_price_map.png")


def plot_pairplot(df: pd.DataFrame) -> None:
    logger.info("Plotting pairplot")

    top_cols = ['price', 'sqft_living', 'grade', 'bathrooms', 'sqft_living15']
    sns.pairplot(df[top_cols], plot_kws={'alpha': 0.3, 's': 10})
    plt.suptitle('Pairplot — Top Features vs Price', y=1.02)
    plt.savefig(f'{FIGURES_DIR}/pairplot.png', dpi=150, bbox_inches='tight')
    plt.show()
    logger.info("Saved pairplot.png")


def run_eda(df: pd.DataFrame) -> None:
    logger.info("Running full EDA")
    plot_price_distribution(df)
    plot_correlation_heatmap(df)
    plot_price_vs_features(df)
    plot_boxplots(df)
    plot_price_by_categories(df)
    plot_geographic_map(df)
    plot_pairplot(df)
    logger.info("EDA complete")