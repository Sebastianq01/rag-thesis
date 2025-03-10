import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.api as sm
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA
import warnings

class TimeSeriesPreprocessor:
    def __init__(self):
        self.scalers = {}  # Store scalers for each variable
        
    def transform_and_standardize(self, df):
        """
        Apply transformations and standardization to the dataset.
        Should be used consistently across all models.
        """
        # Create a copy of the dataframe
        df_copy = df.copy()
        
        # Set observation_date as index if it exists
        if 'observation_date' in df_copy.columns:
            df_copy.set_index('observation_date', inplace=True)
        
        df_transformed = pd.DataFrame(index=df_copy.index)
    
        
        # Keep observation_date if it exists
        if 'observation_date' in df_copy.columns:
            df_transformed['observation_date'] = df_copy['observation_date']
        
         # Group variables by transformation type
        log_vars = [
            'GDP',              # Gross Domestic Product
            'XTIMVA01USM664S',  # Imports
            'XTEXVA01USM664S',  # Exports
            'GPDI',             # Gross Private Domestic Investment
            'RPI',              # Real Personal Income
            'INDPRO',           # Industrial Production
            'BUSLOANS',         # Commercial and Industrial Loans
            'NONREVSL'          # Total Nonrevolving Credit
        ]
        
        diff_vars = [
            'UNRATE',           # Unemployment Rate
            'DFF',              # Federal Funds Rate
            'GS1',              # 1-Year Treasury Rate
            'AAA',              # AAA Corporate Bond Rate
            'IRLTLT01USM156N'   # Long-term Interest Rate
        ]
        
        level_vars = [
            'M1V',              # Velocity of M1 Money Stock
            'CUMFNS',           # Capacity Utilization
            'CLF16OV',          # Civilian Labor Force
            'B235RC1Q027SBEA',  # Personal current taxes
            'ROWFDIQ027S'       # Rest of world direct investment
        ]
        
        inflation_vars = [
            'CPILFESL',         # Core CPI
            'PCEPI'             # PCE Price Index
        ]
        
        # 1. Apply transformations
        # Log transformation
        for var in log_vars:
            if var in df_copy.columns:
                transformed_name = f'{var}_log'
                df_transformed[transformed_name] = np.log(df_copy[var])
                # Standardize
                self.scalers[transformed_name] = StandardScaler()
                df_transformed[transformed_name] = self.scalers[transformed_name].fit_transform(
                    df_transformed[transformed_name].values.reshape(-1, 1)
                )
        
        # First difference
        for var in diff_vars:
            if var in df_copy.columns:
                transformed_name = f'{var}_diff'
                df_transformed[transformed_name] = df_copy[var].diff()
                # Standardize
                self.scalers[transformed_name] = StandardScaler()
                df_transformed[transformed_name] = self.scalers[transformed_name].fit_transform(
                    df_transformed[transformed_name].values.reshape(-1, 1)
                )
        
        # Level variables
        for var in level_vars:
            if var in df_copy.columns:
                # Standardize directly
                self.scalers[var] = StandardScaler()
                df_transformed[var] = self.scalers[var].fit_transform(
                    df_copy[var].values.reshape(-1, 1)
                )
        
        # Inflation rates
        for var in inflation_vars:
            if var in df_copy.columns:
                transformed_name = f'{var}_inflation'
                df_transformed[transformed_name] = df_copy[var].pct_change()
                # Standardize
                self.scalers[transformed_name] = StandardScaler()
                df_transformed[transformed_name] = self.scalers[transformed_name].fit_transform(
                    df_transformed[transformed_name].values.reshape(-1, 1)
                )
        
        # Ensure data has datetime index
        if not isinstance(df_transformed.index, pd.DatetimeIndex):
            df_transformed = df_transformed.set_index(pd.date_range(start='1960-01-01', 
                                                                periods=len(df_transformed), 
                                                                freq='Q'))
        
        # Remove NaN values from transformations
        df_transformed = df_transformed.dropna()
        
        return df_transformed
    
    def inverse_transform(self, data, variable):
        """
        Convert standardized values back to transformed scale
        """
        return self.scalers[variable].inverse_transform(data)
    

def calculate_mape(y_true, y_pred):
    """Calculate Mean Absolute Percentage Error"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


class ModelPerformanceTracker:
    def __init__(self):
        self.metrics = {
            'Linear_Regression': {'RMSE': [], 'R2': [], 'MAPE': []},
            'BVAR': {'RMSE': [], 'R2': [], 'MAPE': []},
            'ARIMA': {'RMSE': [], 'R2': [], 'MAPE': []},
            'TimeGPT': {'RMSE': [], 'R2': [], 'MAPE': []},
            'TimesFM': {'RMSE': [], 'R2': [], 'MAPE': []},
            'LagLlama': {'RMSE': [], 'R2': [], 'MAPE': []},
            'Moirai': {'RMSE': [], 'R2': [], 'MAPE': []}
        }
        
    def add_metrics(self, model_name, y_true, y_pred):
        """
        Add metrics to the tracker
        Args:
            model_name (str): Name of the model
            y_true (np.array): True values
            y_pred (np.array): Predicted values
        """
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        mape = calculate_mape(y_true, y_pred)
        
        self.metrics[model_name]['RMSE'].append(rmse)
        self.metrics[model_name]['R2'].append(r2)
        self.metrics[model_name]['MAPE'].append(mape)
        
        print(f"\n{model_name} Metrics:")
        print(f"RMSE: {rmse:.4f}")
        print(f"R²: {r2:.4f}")
        print(f"MAPE: {mape:.4f}%")
    
    def get_summary(self):
        """
        Get a summary of the metrics for all models
        Returns:
            pd.DataFrame: Summary of metrics for all models
        """
        return pd.DataFrame({
            model: {
                'Final_RMSE': metrics['RMSE'][-1] if metrics['RMSE'] else None,
                'Final_R2': metrics['R2'][-1] if metrics['R2'] else None,
                'Final_MAPE': metrics['MAPE'][-1] if metrics['MAPE'] else None
            }
            for model, metrics in self.metrics.items()
        }).T
    

def test_train_split(df_transformed, test_size=0.3):
    """
    Prepare data for BVAR analysis using pre-transformed and standardized data.
    
    Args:
        df_transformed (pd.DataFrame): Pre-transformed and standardized dataset
        target_var (str): Name of target variable (including transformation suffix)
        test_size (float): Proportion of data to use for testing
    
    Returns:
        tuple: (train_data, test_data, variable_names)
    """
    # Ensure data has datetime index
    if not isinstance(df_transformed.index, pd.DatetimeIndex):
        df_transformed = df_transformed.set_index(pd.date_range(start='1960-01-01', 
                                                            periods=len(df_transformed), 
                                                            freq='QE'))
    # Select all numeric columns except observation_date
    numeric_data = df_transformed.select_dtypes(include=[np.number])
    
    # Remove any remaining NaN values
    numeric_data = numeric_data.dropna()
    
    # Split into train and test
    train_size = int(len(numeric_data) * (1 - test_size))
    train_data = numeric_data[:train_size]
    test_data = numeric_data[train_size:]
    
    return train_data, test_data, list(numeric_data.columns)


def run_ar_benchmark(train_data, test_data, target_var, ar_order=1):
    """
    Runs an AR(p) model on the transformed dataset and returns performance metrics.
    """

    
    # Ensure the index has frequency information
    if not isinstance(train_data.index, pd.DatetimeIndex) or train_data.index.freq is None:
        train_data.index = pd.date_range(
            start=train_data.index[0],
            periods=len(train_data),  # Use periods instead of end date
            freq='QE'
        )
    
    # Fit AR(p) model
    model = AutoReg(train_data[target_var], lags=ar_order)
    model_fit = model.fit()
    
    # Make predictions
    predictions = model_fit.predict(start=len(train_data), end=len(train_data)+len(test_data)-1)
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(test_data[target_var], predictions))
    mape = calculate_mape(test_data[target_var], predictions)

    return rmse, mape


def fit_evaluate_bvar(train_data, test_data, target_var, horizon, lag_order=6):
    """
    Fit Bayesian VAR model and evaluate forecasts using pre-standardized data.
    
    Args:
        train_data (pd.DataFrame): Training data (pre-standardized)
        test_data (pd.DataFrame): Test data (pre-standardized)
        target_var (str): Name of variable to forecast
        horizon (int): Forecast horizon
        lag_order (int): Number of lags to include
    
    Returns:
        tuple: (RMSE, MAPE) for the forecasts
    """
    try:
        # Ensure the index has frequency information
        if not isinstance(train_data.index, pd.DatetimeIndex) or train_data.index.freq is None:
            train_data.index = pd.date_range(
                start=train_data.index[0],
                periods=len(train_data),  # Use periods instead of end date
                freq='QE'
        )
        # Convert data to numpy arrays
        train_array = np.asarray(train_data)
        test_array = np.asarray(test_data)
        
        # Set up BVAR priors (Minnesota prior)
        k = train_array.shape[1]  # Number of variables
        
        # Hyperparameters for Minnesota prior
        # Tighter priors since data is standardized
        lambda1 = 0.05  # Overall tightness (reduced from 0.1)
        lambda2 = 0.99  # Cross-variable weighting
        lambda3 = 1     # Lag decay
        
        # Create prior variance matrix for VAR coefficients
        prior_variance = np.zeros((k * lag_order, k))
        for i in range(k):
            for j in range(k):
                for l in range(lag_order):
                    if i == j:
                        # Own lags - since data is standardized, use same prior for all variables
                        prior_variance[i + l*k, j] = lambda1 / ((l + 1) ** lambda3)
                    else:
                        # Cross-variable lags
                        prior_variance[i + l*k, j] = (lambda1 * lambda2) / ((l + 1) ** lambda3)
        
        # Fit VAR model
        model = sm.tsa.VAR(train_array)
        results = model.fit(maxlags=lag_order, ic=None)
        
        # Initialize storage for forecasts and actuals
        forecasts = []
        actuals = []
        
        # Generate forecasts using rolling window approach
        for i in range(len(test_data) - horizon + 1):
            end_idx = i + horizon
            
            # Generate forecast using last lag_order observations
            forecast = results.forecast(y=train_array[-lag_order:], steps=horizon)
            
            # Store the h-step ahead forecast and actual
            if end_idx <= len(test_data):
                forecasts.append(forecast[horizon-1])
                actuals.append(test_array[end_idx-1])
        
        # Convert to numpy arrays
        forecasts = np.array(forecasts)
        actuals = np.array(actuals)
        
        # Get the index of target variable
        target_idx = train_data.columns.get_loc(target_var)
        
        # Calculate metrics on standardized scale
        rmse = np.sqrt(mean_squared_error(actuals[:, target_idx], forecasts[:, target_idx]))
        mape = calculate_mape(actuals[:, target_idx], forecasts[:, target_idx])
        
        return rmse, mape
    
    except Exception as e:
        print(f"Error in BVAR estimation for {target_var}: {str(e)}")
        return np.nan, np.nan
    

def fit_evaluate_arima(var, train_data, test_data, h, order=(5,1,1)):
    """
    Fit and evaluate ARIMA model
    
    Args:
        var (str): Variable name to model
        train_data (pd.DataFrame): Training data
        test_data (pd.DataFrame): Test data
        h (int): Forecast horizon
        order (tuple): ARIMA order (p,d,q)
    """

    # Ensure the index has frequency information
    if not isinstance(train_data.index, pd.DatetimeIndex) or train_data.index.freq is None:
        train_data.index = pd.date_range(
            start=train_data.index[0],
            periods=len(train_data),  # Use periods instead of end date
            freq='QE'
        )
    
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        model = ARIMA(train_data[var], order=order)
        fitted_model = model.fit()
    
    # Generate forecasts
    forecasts = fitted_model.forecast(steps=h)
    
    # Calculate metrics
    actuals = test_data[var][:h]
    arima_rmse = np.sqrt(mean_squared_error(actuals, forecasts))
    arima_mape = calculate_mape(actuals, forecasts)
    
    return arima_rmse, arima_mape