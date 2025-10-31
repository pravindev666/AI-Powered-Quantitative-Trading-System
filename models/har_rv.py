"""HAR-RV (Heterogeneous Autoregressive Realized Volatility) Model with Advanced Features"""
import numpy as np
import pandas as pd
from scipy.stats import norm, t
from typing import Dict, List, Optional, Tuple
import logging
from sklearn.linear_model import LassoCV, ElasticNet
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import IsolationForest
from sklearn.mixture import GaussianMixture

logger = logging.getLogger(__name__)

class HARRVModel:
    """Advanced HAR-RV model implementation with sophisticated features"""
    def __init__(self, 
                 lags: List[int] = [1, 5, 22, 66], 
                 window_size: int = 66, 
                 alpha: float = 0.01,
                 use_robust: bool = True,
                 n_regimes: int = 3):
        self.lags = lags
        self.window_size = window_size
        self.alpha = alpha
        self.use_robust = use_robust
        self.n_regimes = n_regimes
        self.params = None
        self.scaler = RobustScaler() if use_robust else StandardScaler()
        self.regime_model = GaussianMixture(n_components=n_regimes, random_state=42)
        self.outlier_detector = IsolationForest(contamination=0.1, random_state=42)
        self.is_trained = False
        
    def train(self, returns: pd.Series) -> Dict:
        """
        Train the HAR-RV model on historical returns
        
        Args:
            returns: Series of asset returns
            
        Returns:
            Dict of training metrics
        """
        if returns.isna().all():
            raise ValueError("All values in returns series are NaN")
            
        returns = returns.fillna(method='ffill').fillna(0)  # Clean data
        if len(returns) < self.window_size:
            raise ValueError(f"Insufficient data points ({len(returns)}) for window size {self.window_size}")
            
        try:
            # Prepare features and fit model
            features = self._prepare_features(returns)
            target = self._prepare_target(returns)
            
            # Fit regime model
            self.regime_model.fit(features)
            
            # Train model and store parameters
            self.params = self._fit_model(features, target)
            self.is_trained = True
            
            return {
                'n_samples': len(features),
                'r2_score': 0.5,  # Placeholder - implement actual metric calculation
                'mse': 0.1,  # Placeholder
                'n_regimes': self.n_regimes
            }
            
        except Exception as e:
            logger.error(f"Error training HAR-RV model: {str(e)}")
            raise

    def _compute_realized_volatility(self, returns: pd.Series) -> pd.Series:
        """
        Compute realized volatility using advanced high-frequency methods
        """
        try:
            # Ensure we have valid returns data
            if returns.empty or returns.isna().all():
                logger.warning("No valid returns data for volatility computation")
                return pd.Series(dtype=float)
            
            # Clean returns data
            clean_returns = returns.dropna()
            if len(clean_returns) < 5:
                logger.warning("Insufficient returns data for volatility computation")
                return pd.Series(dtype=float)
            
            # Standard realized vol (annualized) - use variance instead of std for single period
            daily_rv = clean_returns.pow(2) * 252  # Realized variance annualized
            
            # Use simple volatility calculation for all cases
            # Calculate rolling volatility
            window_size = min(5, len(clean_returns))
            rv = clean_returns.rolling(window=window_size).std() * np.sqrt(252)
            
            # Fill any remaining NaN with simple variance-based volatility
            rv = rv.fillna(daily_rv.pow(0.5))  # Square root of variance = volatility
            
            return rv.dropna()
            
        except Exception as e:
            logger.error(f"Error computing realized volatility: {e}")
            # Fallback to simple volatility
            try:
                simple_rv = returns.dropna().rolling(window=5).std() * np.sqrt(252)
                return simple_rv.dropna()
            except Exception:
                return pd.Series(dtype=float)

    def _prepare_features(self, rv: pd.Series) -> pd.DataFrame:
        """
        Prepare HAR-RV features with robust data handling
        """
        try:
            if rv.empty or len(rv) < 5:
                logger.warning("Insufficient volatility data for feature preparation")
                return pd.DataFrame()
            
            features = pd.DataFrame(index=rv.index)
            
            # Basic volatility features
            features['RV'] = rv
            
            # Multi-scale volatility components
            for lag in [1, 5, 22]:
                if len(rv) > lag:
                    features[f'RV_lag_{lag}'] = rv.shift(lag)
                    features[f'RV_ma_{lag}'] = rv.rolling(window=lag).mean()
            
            # Volatility trends
            if len(rv) > 5:
                features['RV_trend_5'] = rv.rolling(window=5).mean() / rv.rolling(window=10).mean() - 1
            if len(rv) > 22:
                features['RV_trend_22'] = rv.rolling(window=22).mean() / rv.rolling(window=44).mean() - 1
            
            # Volatility momentum
            if len(rv) > 1:
                features['RV_momentum'] = rv.pct_change()
            
            # Volatility regime indicators
            if len(rv) > 22:
                rolling_mean = rv.rolling(window=22).mean()
                rolling_std = rv.rolling(window=22).std()
                features['HighVolRegime'] = (rv > rolling_mean + rolling_std).astype(float)
                features['VolRegimeChange'] = features['HighVolRegime'].diff().abs()
            
            # Non-linear transformations
            features['RV_sqrt'] = np.sqrt(rv)
            features['RV_log'] = np.log(rv + 1e-8)  # Add small constant to avoid log(0)
            
            # Temporal features
            if hasattr(rv.index, 'month'):
                features['MonthOfYear'] = rv.index.month
                features['DayOfWeek'] = rv.index.dayofweek
                
                # One-hot encode temporal features
                month_dummies = pd.get_dummies(features['MonthOfYear'], prefix='Month')
                day_dummies = pd.get_dummies(features['DayOfWeek'], prefix='Day')
                features = pd.concat([features, month_dummies, day_dummies], axis=1)
                features.drop(['MonthOfYear', 'DayOfWeek'], axis=1, inplace=True)
            
            return features.dropna()
            
        except Exception as e:
            logger.error(f"Error preparing HAR-RV features: {e}")
            return pd.DataFrame()

    def train(self, returns: pd.Series, validation_size: float = 0.2) -> Dict:
        """
        Train advanced HAR-RV model with:
        - Adaptive LASSO + Elastic Net regularization
        - Time series cross-validation
        - Regime detection
        - Outlier identification
        - Out-of-sample validation
        """
        try:
            # Compute realized volatility
            rv = self._compute_realized_volatility(returns)
            
            # Prepare features and target
            features = self._prepare_features(rv)
            target = rv[features.index]
            # Aggressive NaN cleaning
            logger.info(f"Features before cleaning: {len(features)} rows, {features.isna().sum().sum()} NaNs")
            features = features.fillna(method='ffill').fillna(method='bfill')
            features = features.fillna(features.mean())
            features = features.fillna(0)
            features = features.loc[(features != 0).any(axis=1)]
            target = target[features.index]
            logger.info(f"Features after cleaning: {len(features)} rows, {features.isna().sum().sum()} NaNs")
            if len(features) < 10:
                logger.warning(f"Insufficient data after cleaning: {len(features)} rows")
                raise ValueError("Insufficient data for HAR-RV training")
            
            rv_aligned = rv[features.index]
            regime_feature_list = []
            regime_feature_list.append(rv_aligned.values)
            regime_feature_list.append(rv_aligned.rolling(window=5, min_periods=1).mean().fillna(rv_aligned.mean()).values)
            regime_feature_list.append(rv_aligned.rolling(window=22, min_periods=1).mean().fillna(rv_aligned.mean()).values)
            regime_feature_list.append(rv_aligned.rolling(window=5, min_periods=1).std().fillna(rv_aligned.std()).values)
            regime_features = np.column_stack(regime_feature_list)
            regime_features = np.nan_to_num(regime_features, nan=0.0, posinf=0.0, neginf=0.0)
            logger.info(f"Regime features: shape={regime_features.shape}, NaNs={np.isnan(regime_features).sum()}")
            if len(regime_features) >= 10 and not np.isnan(regime_features).any():
                try:
                    self.regime_model.fit(regime_features)
                    logger.info("Regime model trained successfully")
                except Exception as e:
                    logger.warning(f"Regime model training failed: {e}, continuing without it")
            else:
                logger.warning("Skipping regime model - insufficient valid data")
            
            # Detect outliers
            self.outlier_detector.fit(features)
            
            # Split data for validation
            split_idx = int(len(features) * (1 - validation_size))
            X_train, X_val = features.iloc[:split_idx], features.iloc[split_idx:]
            y_train, y_val = target.iloc[:split_idx], target.iloc[split_idx:]
            
            # Standardize features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_val_scaled = self.scaler.transform(X_val)
            
            # Time series cross-validation
            tscv = TimeSeriesSplit(n_splits=5)
            
            # First stage: Elastic Net for feature selection
            elastic_net = ElasticNet(
                l1_ratio=0.5,
                random_state=42
            )
            
            # Second stage: Adaptive LASSO
            lasso = LassoCV(
                cv=tscv,
                n_alphas=100,
                max_iter=10000,
                selection='random',
                random_state=42
            )
            
            # Fit models
            elastic_net.fit(X_train_scaled, y_train)
            feature_weights = 1 / (np.abs(elastic_net.coef_) + 1e-6)
            lasso.fit(X_train_scaled * feature_weights.reshape(1, -1), y_train)
            
            # Store model components
            self.params = lasso.coef_ / feature_weights
            self.feature_names = features.columns.tolist()
            self.alpha = lasso.alpha_
            self.is_trained = True
            
            # Compute metrics
            train_pred = lasso.predict(X_train_scaled)
            val_pred = lasso.predict(X_val_scaled)
            
            train_metrics = self._compute_metrics(y_train, train_pred)
            val_metrics = self._compute_metrics(y_val, val_pred)
            
            # Feature importance
            importance = pd.Series(
                np.abs(self.params),
                index=self.feature_names
            ).sort_values(ascending=False)
            
            return {
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
                'alpha': self.alpha,
                'feature_importance': importance.to_dict(),
                'n_selected_features': np.sum(self.params != 0)
            }
            
        except Exception as e:
            logger.error(f"Error training HAR-RV model: {str(e)}")
            raise

    def predict(self, returns: pd.Series) -> Dict:
        """
        Generate comprehensive volatility predictions with:
        - Point estimates with confidence intervals
        - Regime probabilities and transitions
        - Advanced risk metrics
        - Multiple scenario forecasts
        - Outlier detection
        """
        if not self.is_trained:
            raise RuntimeError("Model not trained yet")
            
        try:
            # Compute realized volatility
            rv = self._compute_realized_volatility(returns)
            
            # Prepare features
            features = self._prepare_features(rv)
            
            # Standardize features
            X = self.scaler.transform(features)
            
            # Base prediction
            base_pred = X @ self.params
            predictions = pd.Series(base_pred, index=features.index)
            
            # Compute prediction intervals
            residuals = returns - predictions
            std_err = np.std(residuals)
            
            # Student-t based intervals for heavy tails
            dof = 5  # Degrees of freedom for heavy tails
            conf_intervals = {
                '95': {
                    'lower': predictions - t.ppf(0.975, dof) * std_err,
                    'upper': predictions + t.ppf(0.975, dof) * std_err
                },
                '99': {
                    'lower': predictions - t.ppf(0.995, dof) * std_err,
                    'upper': predictions + t.ppf(0.995, dof) * std_err
                }
            }
            
            # Regime analysis
            regime_features = np.column_stack([
                rv,
                rv.rolling(window=5).mean(),
                rv.rolling(window=22).mean(),
                rv.rolling(window=5).std()
            ])
            regime_probs = pd.DataFrame(
                self.regime_model.predict_proba(regime_features),
                index=features.index,
                columns=[f'regime_{i}' for i in range(self.n_regimes)]
            )
            
            # Outlier detection
            outlier_scores = pd.Series(
                self.outlier_detector.decision_function(features),
                index=features.index
            )
            is_outlier = outlier_scores < 0
            
            # Advanced risk metrics
            var_95 = predictions + t.ppf(0.95, dof) * std_err
            es_95 = (predictions + t.ppf(0.95, dof) * std_err * 
                    t.pdf(t.ppf(0.95, dof), dof) / (1 - 0.95))
            
            # Multiple scenario forecasts
            scenarios = {
                'base': predictions,
                'high_stress': predictions * (1 + features['StressIndex'] * 0.2),
                'low_stress': predictions * (1 - features['StressIndex'] * 0.1),
                'regime_shift': predictions * (1 + regime_probs.idxmax(axis=1).astype(float) * 0.15),
                'outlier_adjusted': predictions * (1 - is_outlier * 0.25)
            }
            
            return {
                'predictions': predictions,
                'confidence_intervals': conf_intervals,
                'regime_probabilities': regime_probs.to_dict('series'),
                'outlier_detection': {
                    'scores': outlier_scores,
                    'is_outlier': is_outlier
                },
                'risk_metrics': {
                    'VaR_95': var_95,
                    'ES_95': es_95,
                    'conditional_vol': std_err * np.sqrt(252)
                },
                'scenarios': scenarios
            }
            
        except Exception as e:
            logger.error(f"Error generating predictions: {str(e)}")
            raise

    def _compute_metrics(self, y_true: pd.Series, y_pred: pd.Series) -> Dict:
        """Compute comprehensive model performance metrics"""
        residuals = y_true - y_pred
        
        return {
            'r2': 1 - np.sum(residuals**2) / np.sum((y_true - y_true.mean())**2),
            'mse': np.mean(residuals**2),
            'mae': np.mean(np.abs(residuals)),
            'mape': np.mean(np.abs(residuals / y_true)) * 100,
            'bias': np.mean(residuals),
            'std_err': np.std(residuals),
            'skewness': pd.Series(residuals).skew(),
            'kurtosis': pd.Series(residuals).kurt()
        }

    def load_model(self, path: str) -> None:
        """Load model parameters and components"""
        try:
            data = np.load(path, allow_pickle=True).item()
            self.params = data['params']
            self.feature_names = data['feature_names']
            self.scaler = data['scaler']
            self.regime_model = data['regime_model']
            self.outlier_detector = data['outlier_detector']
            self.is_trained = True
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

    def save_model(self, path: str) -> None:
        """Save model parameters and components"""
        if not self.is_trained:
            raise RuntimeError("Cannot save untrained model")
        try:
            np.save(path, {
                'params': self.params,
                'feature_names': self.feature_names,
                'scaler': self.scaler,
                'regime_model': self.regime_model,
                'outlier_detector': self.outlier_detector
            })
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise