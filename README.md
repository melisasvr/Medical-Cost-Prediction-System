# 🏥 Medical Cost Prediction System

- An interactive machine learning web application that predicts healthcare costs based on patient demographics and health factors.
- Built with Python, Scikit-learn, and Streamlit.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.0+-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.0+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Models](#models)
- [Key Findings](#key-findings)
- [Technologies Used](#technologies-used)
- [Future Improvements](#future-improvements)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## 🎯 Overview
- This project uses machine learning to predict annual medical costs for individuals based on various factors including age, BMI, smoking status, and more.
- The interactive web application allows users to:
- Explore healthcare cost data through interactive visualizations
- Compare multiple machine learning models
- Make real-time predictions for individual patients
- Understand key cost drivers through feature importance analysis

- **Key Insight:** Smoking status is the strongest predictor of medical costs, with smokers paying 3-5x more on average than non-smokers.

## ✨ Features

### 📊 Data Overview
- Interactive data exploration with statistics and visualizations
- Distribution analysis of medical charges
- Correlation analysis between features and costs
- Upload custom CSV datasets or use sample data

### 🤖 Model Performance
- Comparison of 5+ machine learning algorithms
- Performance metrics: R², RMSE, MAE
- Actual vs Predicted visualization
- Automatic best model selection

### 🎯 Cost Prediction
- Real-time individual cost predictions
- Interactive sliders and dropdowns for patient data input
- Monthly and annual cost breakdown
- Risk factor identification and analysis
- Comparison with average costs

### 📈 Feature Insights
- Feature importance rankings
- Visual analysis of cost drivers
- Top 10 most influential features
- Model explainability metrics


## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/medical-cost-prediction.git
cd medical-cost-prediction
```

### Step 2: Create Virtual Environment (Recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Run the Application
```bash
streamlit run app.py
```

The app will automatically open in your browser at `http://localhost:8501`

## 📦 Requirements

Create a `requirements.txt` file with:

```
streamlit==1.28.0
pandas==2.0.3
numpy==1.24.3
matplotlib==3.7.2
seaborn==0.12.2
scikit-learn==1.3.0
plotly==5.16.1
xgboost==1.7.6
lightgbm==4.0.0
```

## 💻 Usage

### Using Sample Data

1. Run the app with `streamlit run app.py`
2. Select "Use Sample Data" in the sidebar
3. Explore the tabs to analyze data and models

### Using Your Own Data

1. Prepare a CSV file with the following columns:
   - `age`: Patient age (18-64)
   - `sex`: Gender (male/female)
   - `bmi`: Body Mass Index (15-50)
   - `children`: Number of children (0-5)
   - `smoker`: Smoking status (yes/no)
   - `region`: Geographic region (northeast/northwest/southeast/southwest)
   - `charges`: Annual medical costs (target variable)

2. Select "Upload CSV File" in the sidebar
3. Upload your CSV file
4. The app will automatically process and analyze your data

### Making Predictions

1. Navigate to the "🎯 Make Prediction" tab
2. Adjust the sliders and dropdowns for patient information:
   - Age (18-64)
   - BMI (15.0-50.0)
   - Sex, Smoker status, Children, Region
3. Click "🔮 Predict Cost"
4. View predicted annual cost, monthly cost, and risk factors

## 📁 Project Structure

```
medical-cost-prediction/
│
├── app.py                      # Main Streamlit application
├── medical_cost_analysis.py    # Python analysis script
├── requirements.txt            # Project dependencies
├── README.md                   # Project documentation
│
├── data/                       # Data directory (optional)
│   └── insurance.csv          # Sample dataset
│
├── models/                     # Saved models (optional)
│   └── best_model.pkl
│
└── screenshots/               # App screenshots for README
    ├── data_overview.png
    ├── model_performance.png
    ├── prediction.png
    └── feature_importance.png
```

## 🤖 Models

The application trains and compares the following machine learning models:

| Model | Type | Best For |
|-------|------|----------|
| **Linear Regression** | Baseline | Understanding linear relationships |
| **Ridge Regression** | Regularized Linear | Preventing overfitting |
| **Lasso Regression** | Regularized Linear | Feature selection |
| **ElasticNet** | Regularized Linear | Combining Ridge + Lasso |
| **Random Forest** | Ensemble | Handling non-linear relationships |
| **Gradient Boosting** | Ensemble | High accuracy predictions |
| **AdaBoost** | Ensemble | Boosting weak learners |
| **Support Vector Regression** | Kernel-based | Complex patterns |
| **Neural Network** | Deep Learning | Non-linear patterns |
| **XGBoost** ⭐ | Gradient Boosting | Best overall performance |
| **LightGBM** | Gradient Boosting | Fast training, high accuracy |
| **CatBoost** | Gradient Boosting | Categorical data handling |

**Best Model:** XGBoost typically achieves R² > 0.90 with RMSE < $4,000

## 🔍 Key Findings

### Top 5 Cost Drivers (Feature Importance)

1. **Smoker Status** (50%+) - Smoking increases costs by 3-5x
2. **Age** (15-20%) - Older patients have higher costs
3. **BMI** (10-15%) - Higher BMI correlates with higher costs
4. **Smoker × BMI Interaction** (8-10%) - Combined effect is significant
5. **Age × BMI Interaction** (5-8%) - Older patients with high BMI face compounding costs

### Model Performance Summary

- **R² Score:** 90.3% (can explain 90% of cost variance)
- **RMSE:** $3,774 average prediction error
- **MAE:** $2,234 typical deviation

### Cost Statistics

- **Average Annual Cost:** $13,270
- **Non-smoker Average:** $8,434
- **Smoker Average:** $32,050
- **Cost Increase from Smoking:** +280%

## 🛠️ Technologies Used

### Core Libraries
- **Python 3.8+** - Programming language
- **Pandas** - Data manipulation
- **NumPy** - Numerical computing
- **Scikit-learn** - Machine learning algorithms

### Advanced ML
- **XGBoost** - Gradient boosting framework
- **LightGBM** - High-performance gradient boosting
- **CatBoost** - Categorical feature handling

### Visualization
- **Plotly** - Interactive charts
- **Matplotlib** - Static visualizations
- **Seaborn** - Statistical graphics

### Web Framework
- **Streamlit** - Interactive web application

## 🚀 Future Improvements

- [ ] Add SHAP values for individual prediction explanations
- [ ] Implement batch prediction (upload multiple patients)
- [ ] Add model retraining capability
- [ ] Include more advanced ensemble methods
- [ ] Add PDF report generation
- [ ] Implement A/B testing for model comparison
- [ ] Add database integration for storing predictions
- [ ] Create API endpoints with FastAPI
- [ ] Add user authentication
- [ ] Implement model monitoring and drift detection

## 📈 Model Optimization

The project includes hyperparameter tuning using GridSearchCV for:
- Number of estimators
- Learning rate
- Maximum depth
- Regularization parameters

This ensures optimal model performance on the given dataset.

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a new branch (`git checkout -b feature/improvement`)
3. Make your changes
4. Commit your changes (`git commit -am 'Add new feature'`)
5. Push to the branch (`git push origin feature/improvement`)
6. Create a Pull Request

## 📄 License

- This project is licensed under the MIT License
```
MIT License

Copyright (c) 2026 Melisa Sever

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
## 👤 Author
**Melisa Sever**

## 🙏 Acknowledgments
- Dataset inspiration from Kaggle's Medical Cost Personal Dataset
- Streamlit community for excellent documentation
- Scikit-learn team for comprehensive ML library

## 📊 Sample Results

```
Model Performance:
├── XGBoost:           R² = 0.9034, RMSE = $3,774
├── LightGBM:          R² = 0.8989, RMSE = $3,856
├── Gradient Boosting: R² = 0.8834, RMSE = $4,145
└── Random Forest:     R² = 0.8678, RMSE = $4,412

Prediction Example:
├── Patient: 30 years old, BMI 25, Non-smoker
├── Predicted Cost: $2,728/year ($227/month)
└── vs Average: -77.2% (much lower than average)
```

---

⭐ **If you found this project helpful, please give it a star!** ⭐

**Made with ❤️ using Python and Machine Learning**
