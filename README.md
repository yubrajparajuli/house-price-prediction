# House Price Prediction — Linear Regression from Scratch

A complete end-to-end machine learning project predicting house prices in King County, Seattle using Linear Regression built from scratch with NumPy.

## Live Demo

- Streamlit App: [House Price Predictor](https://yubrajparajuli-house-price-prediction-appstreamlit-app-cvdjo3.streamlit.app/)
- Docker Hub: [yubraj101/house-price-prediction](https://hub.docker.com/r/yubraj101/house-price-prediction)
- Medium Blog: Coming Soon

## Models Implemented

| Model            | Description                                     |
| ---------------- | ----------------------------------------------- |
| Normal Equation  | Closed-form solution: w = (XᵀX)⁻¹Xᵀy            |
| Gradient Descent | Iterative optimization from scratch using NumPy |
| Sklearn          | Reference implementation for comparison         |

## Results

| Model            | R²     | RMSE   | MAE    |
| ---------------- | ------ | ------ | ------ |
| Normal Equation  | 0.7719 | 0.2554 | 0.1962 |
| Gradient Descent | 0.7711 | 0.2558 | 0.1964 |
| Sklearn          | 0.7719 | 0.2553 | 0.1962 |

## Setup

### Using uv

```bash
git clone https://github.com/yubrajparajuli/house-price-prediction.git
cd house-price-prediction

uv venv
source .venv/bin/activate  # Mac/Linux
.venv\Scripts\activate     # Windows

uv add numpy pandas matplotlib seaborn scikit-learn streamlit jupyter ipykernel

# download dataset from Kaggle
# https://www.kaggle.com/datasets/harlfoxem/housesalesprediction
# place kc_house_data.csv in data/ folder

python main.py

streamlit run app/streamlit_app.py
```

### Using Docker

```bash
docker pull yubraj101/house-price-prediction
docker run -p 8501:8501 yubraj101/house-price-prediction
```

Then open http://localhost:8501

## Dataset

King County House Sales dataset from Kaggle.
21,613 house sales in Seattle, WA between 2014-2015.
Features include sqft, bedrooms, bathrooms, grade, condition, location and more.

Link: https://www.kaggle.com/datasets/harlfoxem/housesalesprediction

## Tech Stack

| Tool                 | Purpose                         |
| -------------------- | ------------------------------- |
| Python 3.11          | Core language                   |
| NumPy                | Model from scratch              |
| Pandas               | Data manipulation               |
| Matplotlib / Seaborn | Visualization                   |
| Scikit-learn         | Train/test split and comparison |
| Streamlit            | Web application                 |
| Docker               | Containerization                |
| uv                   | Package management              |

## Blog

Read the full walkthrough on Medium:
[Building Linear Regression from Scratch on King County House Sales — Coming Soon](#)

## Author

**Yubraj Parajuli**

---

Built as part of a linear regression learning project.
