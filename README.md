# Sector Rotation Trading Strategy with Machine Learning

This repository contains the code and documentation for a machine learning-based investment strategy focused on sector rotation in the U.S. market, aiming to consistently outperform S&P 500 index returns.

## Project Overview

This project applies advanced machine learning techniques to predict the relative performance of different economic sectors through sector ETFs. The strategy incorporates:

- Macroeconomic variables 
- Fama-French 5-factor model
- Technical indicators
- Multiple normalization approaches
- Various dimensionality reduction techniques

The model is designed to select the top-performing sectors and implement rotational trading strategies to achieve alpha generation.

## Key Features

- **Data Processing Pipeline**: Collection and preprocessing of market data, macroeconomic indicators, and factor exposures
- **Feature Engineering**: Creation of technical indicators, momentum metrics, and relative performance measures
- **ML Model Implementation**: LightGBM gradient boosting implementation with cross-validation
- **Robust Validation**: Walk-forward validation methodology to prevent overfitting
- **Model Interpretability**: SHAP analysis and feature importance visualization
- **Backtesting Framework**: Vectorized backtesting implementation for strategy evaluation
- **Performance Metrics**: Information Coefficient (IC) analysis and quintile-based performance evaluation

## Strategy Variants

The repository implements six model variants based on two normalization approaches and three feature selection methods:

1. **Normalization Methods**:
   - Historical volatility normalization
   - Cross-sectional normalization (market effect neutralization)

2. **Feature Selection Methods**:
   - Full feature set
   - Correlation-based feature selection
   - Principal Component Analysis (PCA)

## Trading Strategies

Three trading strategy implementations are included:
- Long-only (buying top-predicted sectors)
- Short-only (selling bottom-predicted sectors)
- Long-short (combination approach)

## Results

![image](https://github.com/user-attachments/assets/00671011-1711-40df-a781-fe52c82c4e0c)

The results demonstrate that:
- Long-only strategies generally match or outperform the benchmark
- Short strategies underperform in the analyzed period
- Different normalization and dimensionality reduction techniques yield varying results across market regimes
- The model can effectively identify relative sector performance

## Requirements

- Python 3.8+
- Key libraries: pandas, numpy, scikit-learn, LightGBM, matplotlib, SHAP, Alphalens

## Installation

```bash
git clone https://github.com/JArmunia/FamaFrenchTrading.git
cd FamaFrenchTrading
pip install -r requirements.txt
```

## Usage
See the example notebooks for detailed implementation of:

- Data preparation
- Model training
- Strategy evaluation
- Performance visualization

## Future Work
Potential extensions of this work include:

- Hybrid model combining strengths of normalized and neutralized approaches
- Real-time adaptation using only data available at prediction time
- Incorporation of transaction costs into strategy evaluation
- More sophisticated market effect neutralization techniques
- Exploration of deep learning techniques

## Citation
If you use this code for academic or research purposes, please cite:
```
CopyArmunia Hinojosa, J. (2024). Machine Learning Applied to Investment Management: 
Sector Rotation Strategy to Outperform the S&P 500. Master's Thesis, 
Universitat Oberta de Catalunya.
```

## License
This project is licensed under the MIT License - see the LICENSE file for details.


