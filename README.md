# Retrieval-Augmented Generation for Large Language Models: Enhancing Applied Economic Reasoning and Forecasting

## Overview

This repository contains the implementation and research for an MIT MEng thesis exploring the application of Retrieval-Augmented Generation (RAG) in economic forecasting and reasoning. The project combines traditional economic forecasting methods with state-of-the-art language models to enhance predictive accuracy and interpretability.

Paper: [insert link here]

## Project Structure

- `scripts/`: Core source code for the RAG implementation
- `data/`: Economic datasets and processed data
- `results/`: Experimental results and model evaluations

## Models and Components

The project implements and evaluates several forecasting approaches:

### Traditional Models

- Linear Regression
- ARIMA (AutoRegressive Integrated Moving Average)
- BVAR (Bayesian Vector Autoregression)

### Advanced Models

- TimeGPT: Time series forecasting with GPT architecture
- TimesFM: Time series forecasting with foundation models
- LagLlama: Custom implementation using LLaMA architecture
- Moirai: Custom forecasting implementation

## Key Features

- Economic time series preprocessing and transformation
- Multiple forecasting horizons (h=1, h=6, h=12)
- Comprehensive model performance tracking
- RMSE, R², and MAPE metric evaluation
- Specialized handling for various economic indicators:
  - GDP
  - Imports/Exports
  - Gross Private Domestic Investment
  - Real Personal Income
  - Industrial Production
  - Commercial and Industrial Loans
  - Total Nonrevolving Credit

## Requirements

```python
# Key dependencies in requirements.txt
pandas
numpy
matplotlib
seaborn
scipy
torch
scikit-learn
statsmodels
pymc
arviz
```

## Usage

1. Set up a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

2. Data Preparation:

- Place your economic time series data in the `data/` directory
- Use the preprocessing utilities in `forecasting_utils.py`

3. Model Training and Evaluation:

- Jupyter notebooks are provided for different models:
  - `forecasting.ipynb`: General forecasting experiments + TimeGPT
  - `rag_evaluation.ipynb`: RAG+HyDE model evaluation for Multiple-Choice Test Bank

## Results

The project includes comprehensive evaluations across different forecasting horizons and models. Results are stored in the `results/` directory, with detailed performance metrics for each model.

## Author

Sebastian Quintero  
MIT B.S. '24, M.Eng. '25

## License

All rights reserved.
