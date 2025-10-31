import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Union
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif, SelectKBest
from sklearn.ensemble import IsolationForest
from sklearn.mixture import GaussianMixture
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import logging

logger = logging.getLogger(__name__)

class AlphaFactorDataset(Dataset):
    """PyTorch dataset for alpha factors."""
    
    def __init__(self, features: pd.DataFrame, labels: pd.Series):
        self.features = torch.FloatTensor(features.values)
        self.labels = torch.FloatTensor(labels.values)
        
    def __len__(self):
        return len(self.features)
        
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

class AlphaFactorNet(nn.Module):
    """Neural network for alpha factor combination."""
    
    def __init__(self, input_dim: int, hidden_dims: List[int], dropout: float = 0.2):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.LeakyReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
            
        # Final layer
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)

class AlphaFactorProcessor:
    """Process and enhance alpha factors using machine learning."""
    
    def __init__(self, lookback_window: int = 252):
        self.lookback_window = lookback_window
        self.scaler = RobustScaler()
        self.pca = PCA(n_components=0.95)  # Keep 95% variance
        self.feature_selector = None
        self.alpha_net = None
        self.anomaly_detector = IsolationForest(contamination=0.1)
        self.regime_detector = GaussianMixture(n_components=3)
        
    def preprocess_factors(self, factors: pd.DataFrame) -> pd.DataFrame:
        """Preprocess alpha factors."""
        # Remove highly correlated factors
        factors = self._remove_correlated_factors(factors)
        
        # Scale factors
        scaled_factors = pd.DataFrame(
            self.scaler.fit_transform(factors),
            index=factors.index,
            columns=factors.columns
        )
        
        # Detect and remove anomalies
        anomalies = self.anomaly_detector.fit_predict(scaled_factors)
        scaled_factors = scaled_factors[anomalies == 1]
        
        return scaled_factors
        
    def enhance_factors(self, factors: pd.DataFrame, returns: pd.Series) -> pd.DataFrame:
        """Enhance alpha factors using ML techniques."""
        # Preprocess factors
        processed_factors = self.preprocess_factors(factors)
        
        # Apply PCA
        pca_factors = pd.DataFrame(
            self.pca.fit_transform(processed_factors),
            index=processed_factors.index,
            columns=[f'PC_{i+1}' for i in range(self.pca.n_components_)]
        )
        
        # Select best features
        if self.feature_selector is None:
            self.feature_selector = SelectKBest(
                score_func=mutual_info_classif,
                k=min(10, len(pca_factors.columns))
            )
            
        selected_factors = pd.DataFrame(
            self.feature_selector.fit_transform(pca_factors, (returns > 0).astype(int)),
            index=pca_factors.index,
            columns=pca_factors.columns[self.feature_selector.get_support()]
        )
        
        return selected_factors
        
    def train_alpha_combiner(self, factors: pd.DataFrame, returns: pd.Series,
                           epochs: int = 100, batch_size: int = 32) -> Dict:
        """Train neural network to combine alpha factors."""
        # Prepare data
        dataset = AlphaFactorDataset(factors, returns)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Initialize network
        input_dim = factors.shape[1]
        hidden_dims = [32, 16, 8]
        self.alpha_net = AlphaFactorNet(input_dim, hidden_dims)
        
        # Training setup
        criterion = nn.MSELoss()
        optimizer = optim.AdamW(self.alpha_net.parameters(), lr=0.001)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
        
        # Training loop
        train_losses = []
        best_loss = float('inf')
        
        for epoch in range(epochs):
            epoch_loss = 0
            for batch_x, batch_y in dataloader:
                optimizer.zero_grad()
                output = self.alpha_net(batch_x)
                loss = criterion(output.squeeze(), batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                
            avg_loss = epoch_loss / len(dataloader)
            train_losses.append(avg_loss)
            scheduler.step(avg_loss)
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(self.alpha_net.state_dict(), 'best_alpha_net.pth')
                
        return {
            'train_losses': train_losses,
            'best_loss': best_loss
        }
        
    def combine_factors(self, factors: pd.DataFrame) -> pd.Series:
        """Combine alpha factors using trained neural network."""
        if self.alpha_net is None:
            raise ValueError("Alpha combiner not trained yet")
            
        self.alpha_net.eval()
        with torch.no_grad():
            combined = self.alpha_net(torch.FloatTensor(factors.values))
            
        return pd.Series(combined.numpy().squeeze(), index=factors.index)
        
    def detect_regime(self, factors: pd.DataFrame) -> Dict:
        """Detect market regime based on alpha factor patterns."""
        regime_labels = self.regime_detector.fit_predict(factors)
        regime_probs = self.regime_detector.predict_proba(factors)
        
        regimes = pd.Series(regime_labels, index=factors.index)
        probabilities = pd.DataFrame(regime_probs, index=factors.index)
        
        return {
            'regimes': regimes,
            'probabilities': probabilities,
            'means': self.regime_detector.means_,
            'covariances': self.regime_detector.covariances_
        }
        
    def _remove_correlated_factors(self, factors: pd.DataFrame,
                                 threshold: float = 0.7) -> pd.DataFrame:
        """Remove highly correlated factors."""
        corr_matrix = factors.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
        
        return factors.drop(columns=to_drop)
        
    def get_factor_importance(self) -> pd.Series:
        """Get importance scores for each factor."""
        if self.feature_selector is None:
            raise ValueError("Feature selector not fitted yet")
            
        importance = pd.Series(
            self.feature_selector.scores_,
            index=self.feature_selector.feature_names_in_
        )
        
        return importance.sort_values(ascending=False)
        
    def get_regime_characteristics(self) -> Dict:
        """Get characteristics of each market regime."""
        if self.regime_detector is None:
            raise ValueError("Regime detector not fitted yet")
            
        n_regimes = self.regime_detector.n_components
        characteristics = {}
        
        for i in range(n_regimes):
            characteristics[f'regime_{i}'] = {
                'mean': self.regime_detector.means_[i],
                'covariance': self.regime_detector.covariances_[i],
                'weight': self.regime_detector.weights_[i]
            }
            
        return characteristics

class SignalOptimizer:
    """Optimize trading signals using machine learning."""
    
    def __init__(self):
        self.alpha_processor = AlphaFactorProcessor()
        self.current_regime = None
        self.signal_history = []
        
    def process_signal(self, raw_signal: float,
                      factors: pd.DataFrame,
                      returns: pd.Series) -> Dict:
        """Process and enhance trading signal."""
        # Enhance factors
        enhanced_factors = self.alpha_processor.enhance_factors(factors, returns)
        
        # Detect regime
        regime_info = self.alpha_processor.detect_regime(enhanced_factors)
        self.current_regime = regime_info['regimes'].iloc[-1]
        
        # Combine factors
        alpha_signal = self.alpha_processor.combine_factors(enhanced_factors)
        
        # Blend signals
        final_signal = self._blend_signals(raw_signal, alpha_signal.iloc[-1])
        
        # Store signal info
        self.signal_history.append({
            'timestamp': factors.index[-1],
            'raw_signal': raw_signal,
            'alpha_signal': alpha_signal.iloc[-1],
            'final_signal': final_signal,
            'regime': self.current_regime
        })
        
        return {
            'signal': final_signal,
            'regime': self.current_regime,
            'regime_probs': regime_info['probabilities'].iloc[-1],
            'alpha_contribution': alpha_signal.iloc[-1],
            'raw_contribution': raw_signal
        }
        
    def _blend_signals(self, raw_signal: float, alpha_signal: float,
                      alpha_weight: float = 0.3) -> float:
        """Blend raw and alpha signals."""
        return (1 - alpha_weight) * raw_signal + alpha_weight * alpha_signal
        
    def get_signal_metrics(self) -> Dict:
        """Calculate signal quality metrics."""
        if not self.signal_history:
            return None
            
        history = pd.DataFrame(self.signal_history)
        
        metrics = {
            'signal_stability': np.std([s['final_signal'] for s in self.signal_history]),
            'regime_distribution': history['regime'].value_counts().to_dict(),
            'alpha_correlation': np.corrcoef(
                history['raw_signal'],
                history['alpha_signal']
            )[0,1],
            'enhancement_ratio': np.mean(np.abs(history['final_signal']) / 
                                      np.abs(history['raw_signal']))
        }
        
        return metrics
        
    def get_regime_signals(self) -> Dict:
        """Get signal statistics by regime."""
        if not self.signal_history:
            return None
            
        history = pd.DataFrame(self.signal_history)
        regime_stats = {}
        
        for regime in history['regime'].unique():
            regime_data = history[history['regime'] == regime]
            regime_stats[f'regime_{regime}'] = {
                'avg_signal': regime_data['final_signal'].mean(),
                'signal_vol': regime_data['final_signal'].std(),
                'raw_alpha_corr': np.corrcoef(
                    regime_data['raw_signal'],
                    regime_data['alpha_signal']
                )[0,1],
                'count': len(regime_data)
            }
            
        return regime_stats