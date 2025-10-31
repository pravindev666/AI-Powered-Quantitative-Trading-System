"""Time series transformers and datasets for machine learning models"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Union, Tuple
from datetime import datetime, timedelta
import torch
from torch.utils.data import Dataset

class TimeSeriesTransformer:
    """Transform time series data for ML models"""
    
    def __init__(self, sequence_length: int = 10):
        """Initialize transformer with sequence length"""
        self.sequence_length = sequence_length
        
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform time series data into sequences
        
        Args:
            data: DataFrame with time series data
            
        Returns:
            DataFrame with transformed sequences
        """
        return data.copy()
        
    def inverse_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Inverse transform predictions back to original scale
        
        Args:
            data: DataFrame with predictions
            
        Returns:
            DataFrame with inverse transformed data
        """
        return data.copy()

class TimeSeriesDataset(Dataset):
    """PyTorch dataset for time series data"""
    
    def __init__(self, features: pd.DataFrame, targets: pd.Series):
        """Initialize dataset
        
        Args:
            features: DataFrame of input features
            targets: Series of target values
        """
        self.features = torch.FloatTensor(features.values)
        self.targets = torch.FloatTensor(targets.values)
        
    def __len__(self) -> int:
        """Get dataset length"""
        return len(self.features)
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get item by index"""
        return self.features[idx], self.targets[idx]
