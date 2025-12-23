# 📱 Smartphone Price Prediction

Machine learning project to classify smartphones as expensive or non-expensive based on technical specifications.

## 🎯 Project Overview

This project implements a complete machine learning pipeline with advanced feature selection to predict smartphone price categories with 92%+ accuracy.

## 📊 Key Results

- **Best Model**: RandomForest
- **Accuracy**: 92.5%
- **Features**: Reduced from 35 to 20 (43% reduction)
- **Models Compared**: 3 (RandomForest, Logistic Regression, SVC)

## 🚀 Features

- Advanced feature selection (variance threshold, correlation analysis, importance ranking)
- Three model comparison (RandomForest, Logistic Regression, SVC)
- Comprehensive visualizations (feature importance, confusion matrices, performance comparison)
- Production-ready preprocessing pipeline
- Stratified train-test split for balanced evaluation

## 🛠️ Technologies Used

- Python 3.x
- Scikit-learn
- Pandas
- NumPy
- Matplotlib & Seaborn
- Joblib

## 📁 Project Structure
```
├── train.csv                    # Training dataset
├── test.csv                     # Test dataset
├── main.py                      # Main ML pipeline
├── Models/                      # Saved models
│   ├── best_model_randomforest.pkl
│   ├── logistic_regression_model.pkl
│   ├── svc_model.pkl
│   └── selected_features.csv
├── plots/                       # Visualizations
│   ├── feature_importance.png
│   ├── model_comparison.png
│   ├── confusion_matrices.png
│   └── feature_selection_impact.png
└── README.md
```

## 🔧 Installation
```bash
pip install numpy pandas scikit-learn matplotlib seaborn joblib
```

## 📈 Results

### Model Performance

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| RandomForest | 92.5% | 0.89 | 0.92 | 0.90 |
| SVC | 90.2% | 0.87 | 0.88 | 0.88 |
| Logistic Regression | 85.0% | 0.83 | 0.84 | 0.84 |

### Top Features

1. rating (15.6%)
2. RAM Size GB (14.2%)
3. battery_capacity (9.8%)
4. primary_rear_camera_mp (8.7%)
5. Storage Size GB (7.5%)

## 🎓 Key Learnings

- Feature selection significantly improves model interpretability
- Stratified sampling is crucial for imbalanced datasets
- Pipeline architecture prevents data leakage
- RandomForest excels at capturing non-linear relationships

## 👨‍💻 Author

[Omar Ganoub] (https://www.linkedin.com/in/omarganoub)
