import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectFromModel, RFE, RFECV, mutual_info_classif
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.model_selection import TimeSeriesSplit
from scipy.stats import spearmanr
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

class AdvancedFeatureEngineer:
    """Advanced feature engineering and selection pipeline"""
    
    def __init__(self, n_poly_degree=2, n_top_features=30):
        self.n_poly_degree = n_poly_degree
        self.n_top_features = n_top_features
        self.selected_features = None
        self.poly = PolynomialFeatures(degree=n_poly_degree, include_bias=False)
        self.base_estimator = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=42
        )
        
    def _create_time_based_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features"""
        # Time components
        df['DayOfWeek'] = df.index.dayofweek
        df['MonthStart'] = df.index.is_month_start.astype(int)
        df['MonthEnd'] = df.index.is_month_end.astype(int)
        df['QuarterStart'] = df.index.is_quarter_start.astype(int)
        df['QuarterEnd'] = df.index.is_quarter_end.astype(int)
        
        # Cyclical encoding of time features
        df['DayOfWeek_sin'] = np.sin(2 * np.pi * df['DayOfWeek']/7)
        df['DayOfWeek_cos'] = np.cos(2 * np.pi * df['DayOfWeek']/7)
        
        return df
        
    def _create_technical_interactions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create meaningful technical indicator interactions"""
        if all(x in df.columns for x in ['RSI', 'BB_Width']):
            df['RSI_BB'] = df['RSI'] * df['BB_Width']
            
        if all(x in df.columns for x in ['ADX', 'ATR']):
            df['ADX_ATR'] = df['ADX'] * df['ATR']
            
        if all(x in df.columns for x in ['Volume_Ratio', 'ROC']):
            df['Vol_ROC'] = df['Volume_Ratio'] * df['ROC']
            
        if 'IndiaVIX' in df.columns and 'ATR' in df.columns:
            df['VIX_ATR_Ratio'] = df['IndiaVIX'] / df['ATR']
            
        return df
        
    def _create_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create rolling window features"""
        windows = [5, 10, 20]
        
        for col in ['RSI', 'ATR', 'ROC', 'IndiaVIX']:
            if col in df.columns:
                for window in windows:
                    df[f'{col}_MA_{window}'] = df[col].rolling(window).mean()
                    df[f'{col}_STD_{window}'] = df[col].rolling(window).std()
                    
        if 'Close' in df.columns:
            for window in windows:
                df[f'Returns_{window}'] = df['Close'].pct_change(window)
                df[f'Volatility_{window}'] = df['Close'].pct_change().rolling(window).std()
                
        return df
        
    def _remove_highly_correlated(self, df: pd.DataFrame, threshold=0.95) -> List[str]:
        """Remove highly correlated features"""
        corr_matrix = df.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
        return [col for col in df.columns if col not in to_drop]
        
    def _select_features_mi(self, X: pd.DataFrame, y: pd.Series) -> List[str]:
        """Select features using mutual information"""
        mi_scores = mutual_info_classif(X, y)
        mi_series = pd.Series(mi_scores, index=X.columns)
        top_mi = mi_series.nlargest(self.n_top_features)
        return list(top_mi.index)
        
    def _select_features_rfe(self, X: pd.DataFrame, y: pd.Series) -> List[str]:
        """Select features using recursive feature elimination with cross-validation"""
        tscv = TimeSeriesSplit(n_splits=5)
        rfecv = RFECV(
            estimator=self.base_estimator,
            step=1,
            cv=tscv,
            scoring=make_scorer(accuracy_score),
            min_features_to_select=10
        )
        rfecv.fit(X, y)
        return list(X.columns[rfecv.support_])
        
    def _create_polynomial_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create polynomial features for selected numeric columns"""
        numeric_cols = X.select_dtypes(include=['float64', 'int64']).columns
        if len(numeric_cols) > 0:
            poly_features = self.poly.fit_transform(X[numeric_cols])
            feature_names = self.poly.get_feature_names_out(numeric_cols)
            poly_df = pd.DataFrame(poly_features, columns=feature_names, index=X.index)
            return pd.concat([X.drop(numeric_cols, axis=1), poly_df], axis=1)
        return X
        
    def fit_transform(self, df: pd.DataFrame, y: pd.Series = None) -> pd.DataFrame:
        """Main method to engineer and select features"""
        print("\n📊 Advanced Feature Engineering Pipeline")
        print("=" * 60)
        
        # 1. Create time-based features
        print("   ⚙️ Creating time-based features...")
        df = self._create_time_based_features(df)
        
        # 2. Create technical interactions
        print("   ⚙️ Creating technical interactions...")
        df = self._create_technical_interactions(df)
        
        # 3. Create rolling features
        print("   ⚙️ Creating rolling window features...")
        df = self._create_rolling_features(df)
        
        # 4. Remove highly correlated features
        print("   ⚙️ Removing highly correlated features...")
        uncorrelated = self._remove_highly_correlated(df.fillna(0))
        df = df[uncorrelated]
        
        if y is not None:
            # 5. Feature selection using multiple methods
            print("   ⚙️ Performing feature selection...")
            
            # Mutual Information selection
            mi_features = self._select_features_mi(df.fillna(0), y)
            
            # RFE selection
            rfe_features = self._select_features_rfe(df.fillna(0), y)
            
            # Combine selected features
            selected = list(set(mi_features) | set(rfe_features))
            df = df[selected]
            
            # 6. Create polynomial features for final set
            print("   ⚙️ Creating polynomial features...")
            df = self._create_polynomial_features(df)
            
            self.selected_features = df.columns.tolist()
            print(f"   ✅ Selected {len(self.selected_features)} features")
            
        return df
        
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform new data using the fitted feature engineering pipeline"""
        if self.selected_features is None:
            raise ValueError("Must call fit_transform before transform")
            
        df = self._create_time_based_features(df)
        df = self._create_technical_interactions(df)
        df = self._create_rolling_features(df)
        
        if self.selected_features:
            # Only keep selected features in the same order
            available_features = [f for f in self.selected_features if f in df.columns]
            df = df[available_features]
            
        return df