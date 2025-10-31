"""
Temporal Fusion Transformer for Multi-horizon Forecasting
"""

import os
import json
import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Callable, Any, Union
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.metrics import QuantileLoss
from pytorch_forecasting.data import GroupNormalizer
from .config import TFT_CONFIG, MODEL_DIR
from .config import TFT_CONFIG, MODEL_DIR
from pytorch_forecasting.data.encoders import (
    NaNLabelEncoder, EncoderNormalizer, TorchNormalizer, 
    GroupNormalizer, MultiNormalizer
)
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
import logging
import safetensors.torch
from pathlib import Path

logger = logging.getLogger(__name__)

def safe_load_model(model_path: str) -> Dict:
    """Safely load a PyTorch model using safetensors"""
    import os
    try:
        model_path = Path(model_path)
        if model_path.suffix == '.safetensors':
            return safetensors.torch.load_file(str(model_path))

        # Only allow conversion when torch >= 2.6 due to security restrictions
        torch_version = tuple(int(x) for x in torch.__version__.split('+')[0].split('.')[:2])
        if torch_version >= (2, 6):
            # Convert to safetensors format first
            state_dict = torch.load(model_path, map_location='cpu')
            safe_path = model_path.with_suffix('.safetensors')
            safetensors.torch.save_file(state_dict, str(safe_path))
            try:
                os.remove(model_path)  # Remove the original unsafe file
            except Exception:
                pass
            return safetensors.torch.load_file(str(safe_path))
        else:
            logger.error(
                f"Refusing to call torch.load: installed torch {torch.__version__} < 2.6. "
                "Provide a `.safetensors` file or upgrade torch to >=2.6 to enable conversion."
            )
            return {}
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return {}

class TemporalFusionTransformerModel:
    """
    Temporal Fusion Transformer for probabilistic multi-horizon forecasting
    Produces quantile predictions (5%, 50%, 95%) instead of binary classification
    """
    
    def __init__(
        self,
        max_encoder_length: int = None,
        max_prediction_length: int = None,
        learning_rate: float = None,
        hidden_size: int = None,
        attention_head_size: int = None,
        dropout: float = None,
        hidden_continuous_size: int = None,
        output_size: int = None,
        config: Dict[str, Any] = None
    ):
        """
        Initialize TFT model with configuration
        
        Args:
            max_encoder_length: Length of encoder sequence
            max_prediction_length: Length of prediction sequence
            learning_rate: Learning rate for training
            hidden_size: Hidden layer size
            attention_head_size: Number of attention heads
            dropout: Dropout rate
            hidden_continuous_size: Hidden continuous layer size
            output_size: Number of output quantiles
            config: Optional configuration dictionary. If None, uses TFT_CONFIG from config.py
        """
        # Use provided config or default TFT_CONFIG
        base_config = TFT_CONFIG.copy()
        
        # Override with any provided parameters
        if max_encoder_length is not None:
            base_config['max_encoder_length'] = max_encoder_length
        if max_prediction_length is not None:
            base_config['max_prediction_length'] = max_prediction_length
        if learning_rate is not None:
            base_config['learning_rate'] = learning_rate
        if hidden_size is not None:
            base_config['hidden_size'] = hidden_size
        if attention_head_size is not None:
            base_config['attention_head_size'] = attention_head_size
        if dropout is not None:
            base_config['dropout'] = dropout
        if hidden_continuous_size is not None:
            base_config['hidden_continuous_size'] = hidden_continuous_size
        if output_size is not None:
            base_config['output_size'] = output_size
            
        # Update with any additional config
        if config is not None:
            base_config.update(config)
        
        self.config = base_config
        
        # Set model parameters from config
        self.max_encoder_length = self.config['max_encoder_length']
        self.max_prediction_length = self.config['max_prediction_length']
        self.learning_rate = self.config['learning_rate']
        self.hidden_size = self.config['hidden_size']
        self.attention_head_size = self.config['attention_head_size']
        self.dropout = self.config['dropout']
        self.hidden_continuous_size = self.config['hidden_continuous_size']
        self.output_size = self.config['output_size']
        
        self.model = None
        self.training_data = None
        self.validation_data = None
        
    def prepare_data(self, df: pd.DataFrame) -> Tuple[TimeSeriesDataSet, TimeSeriesDataSet]:
        """
        Prepare data for TFT model
        """
        try:
            # Add time index and group identifier
            df = df.copy()
            
            # Create integer time index
            if isinstance(df.index, pd.DatetimeIndex):
                df['time_idx'] = ((pd.to_datetime(df.index) - pd.to_datetime(df.index.min())).days).astype('int32')
            else:
                df['time_idx'] = np.arange(len(df)).astype('int32')
                
            df['series'] = 'NIFTY'  # Single time series identifier
            
            # Clean and preprocess the data
            df = df.copy()
            if not isinstance(df, pd.DataFrame):
                df = pd.DataFrame(df)
                
            # Handle duplicate columns by keeping only the first occurrence
            df = df.loc[:, ~df.columns.duplicated()]
                
            # Handle target column first if it exists
            if 'target' in df.columns:
                if isinstance(df['target'], pd.DataFrame):
                    df['target'] = df['target'].iloc[:, 0]
                df['target'] = pd.to_numeric(df['target'].values, errors='coerce')
                
            # Handle Returns data
            if 'Returns' not in df.columns and isinstance(df.index, pd.DatetimeIndex):
                df['Returns'] = df['Close'].pct_change()
            else:
                # Handle Returns column based on its type
                returns_col = df['Returns']
                if isinstance(returns_col, pd.DataFrame):
                    returns_col = returns_col.iloc[:, 0]  # Take first column
                elif isinstance(returns_col, pd.Series):
                    returns_col = returns_col.values  # Convert to numpy array
                    
                # Convert to numeric values
                df['Returns'] = pd.to_numeric(returns_col, errors='coerce')
                
            # Handle missing values
            df['Returns'] = df['Returns'].ffill().fillna(0)
            
            # Define variable groups
            static_categoricals = ['series']
            static_reals = []
            time_varying_known_reals = ['time_idx']
            time_varying_unknown_reals = [
                'Close', 'High', 'Low', 'Volume', 'IndiaVIX', 'Returns'
            ]
            
            # Ensure numeric columns have correct types and format
            float_columns = ['Close', 'High', 'Low', 'Volume', 'IndiaVIX', 'Returns', 'target']
            for col in float_columns:
                if col in df.columns:
                    if isinstance(df[col], pd.DataFrame):
                        # Convert DataFrame column to numpy array then back to Series
                        df[col] = pd.Series(df[col].iloc[:, 0].to_numpy(), index=df.index, dtype='float32')
                    elif isinstance(df[col], pd.Series):
                        # Convert Series to numpy array then back to ensure clean data
                        df[col] = pd.Series(df[col].to_numpy(), index=df.index, dtype='float32')
                    else:
                        # Convert other types through numpy array to Series
                        df[col] = pd.Series(np.array(df[col]), index=df.index, dtype='float32')
            
            # Ensure time_idx is integer
            if 'time_idx' in df.columns:
                df['time_idx'] = pd.to_numeric(df['time_idx'], errors='coerce').astype('int32')
            
            # Create training dataset
            training_cutoff = df['time_idx'].max() - self.max_prediction_length
            df_train = df[lambda x: x.time_idx <= training_cutoff].copy()
            
            # Normalize column types: ensure no column is a DataFrame and all columns are Series
            for col in df_train.columns:
                # Preserve 'series' as a string categorical
                if col == 'series':
                    df_train[col] = df_train[col].astype(str)
                    continue

                # Get the column value
                val = df_train[col]
                
                # If a column contains a DataFrame (e.g., nested structure), take the first column
                if isinstance(val, pd.DataFrame):
                    val = val.iloc[:, 0]
                
                # If a column is an ndarray with ndim > 1, reduce to first column
                if isinstance(val, np.ndarray) and getattr(val, 'ndim', 1) > 1:
                    val = val[:, 0]
                
                # Ensure we have a pandas Series (1-D) for each column
                if not isinstance(val, pd.Series):
                    try:
                        val = pd.Series(val, index=df_train.index)
                    except Exception:
                        # As a last resort, convert element-wise
                        try:
                            val = pd.Series([v if not isinstance(v, (list, np.ndarray, pd.DataFrame)) else (v[0] if len(v) else np.nan) for v in val], index=df_train.index)
                        except Exception:
                            val = pd.Series([np.nan] * len(df_train), index=df_train.index)

                # Assign the Series to the column
                df_train[col] = val
                
                # Coerce numeric types explicitly
                if col != 'series':  # Skip string columns
                    try:
                        if pd.api.types.is_integer_dtype(val.dtype):
                            df_train[col] = pd.to_numeric(df_train[col], errors='coerce').astype('int32')
                        elif pd.api.types.is_float_dtype(val.dtype) or pd.api.types.is_object_dtype(val.dtype):
                            # Convert objects/floats to float32
                            if val.dtype == object:
                                # try to convert, but leave as-is if non-numeric
                                try:
                                    arr = pd.to_numeric(df_train[col], errors='coerce').to_numpy(dtype='float32')
                                    df_train[col] = pd.Series(arr, index=df_train.index)
                                except Exception:
                                    pass
                            else:
                                try:
                                    arr = df_train[col].to_numpy(dtype='float32')
                                    df_train[col] = pd.Series(arr, index=df_train.index)
                                except Exception:
                                    pass
                    except Exception:
                        pass
            
            training = TimeSeriesDataSet(
                df_train,
                time_idx='time_idx',
                target='target',  # Target variable (can be Returns or a derived target)
                group_ids=['series'],
                min_encoder_length=self.max_encoder_length // 2,
                max_encoder_length=self.max_encoder_length,
                min_prediction_length=1,
                max_prediction_length=self.max_prediction_length,
                static_categoricals=static_categoricals,
                static_reals=static_reals,
                time_varying_known_reals=time_varying_known_reals,
                time_varying_unknown_reals=time_varying_unknown_reals,
                target_normalizer=TorchNormalizer(method='robust', center=True),
                add_relative_time_idx=True,
                add_target_scales=True,
                add_encoder_length=True
            )
            
            # Create validation set
            validation = TimeSeriesDataSet.from_dataset(
                training,
                df,
                min_prediction_idx=training_cutoff + 1,
                stop_randomization=True
            )
            
            self.training_data = training
            self.validation_data = validation
            
            logger.info("Data preparation completed successfully")
            return training, validation
            
        except Exception as e:
            logger.error(f"Error preparing data: {str(e)}")
            raise
            
    def train(self, df: pd.DataFrame, max_epochs: int = 30) -> Dict:
        """
        Train the TFT model
        
        Args:
            df: DataFrame with features
            max_epochs: Maximum number of training epochs
            
        Returns:
            Dictionary with training metrics
        """
        try:
            logger.info("Starting TFT model training")
            
            # Prepare data
            training, validation = self.prepare_data(df)
            
            # Create dataloaders
            train_dataloader = training.to_dataloader(
                train=True,
                batch_size=128,
                num_workers=0,
                shuffle=True
            )
            
            val_dataloader = validation.to_dataloader(
                train=False,
                batch_size=128 * 2,
                num_workers=0,
                shuffle=False
            )
            
            # Create model
            self.model = TemporalFusionTransformer.from_dataset(
                training,
                learning_rate=self.learning_rate,
                hidden_size=self.hidden_size,
                attention_head_size=self.attention_head_size,
                dropout=self.dropout,
                hidden_continuous_size=self.hidden_continuous_size,
                loss=QuantileLoss(),
                output_size=self.output_size,
                log_interval=10,
                reduce_on_plateau_patience=4
            )
            
            # Configure callbacks
            early_stop_callback = EarlyStopping(
                monitor="val_loss",
                min_delta=1e-4,
                patience=10,
                verbose=False,
                mode="min"
            )
            
            lr_monitor = LearningRateMonitor()
            
            # Configure trainer
            trainer = pl.Trainer(
                max_epochs=max_epochs,
                accelerator="auto",
                enable_model_summary=True,
                gradient_clip_val=0.1,
                callbacks=[early_stop_callback, lr_monitor],
                limit_train_batches=50,
                limit_val_batches=30,
                logger=True
            )
            
            # Train
            trainer.fit(
                self.model,
                train_dataloaders=train_dataloader,
                val_dataloaders=val_dataloader
            )
            
            # Get metrics
            metrics = {
                "train_loss": float(trainer.callback_metrics.get("train_loss", 0)),
                "val_loss": float(trainer.callback_metrics.get("val_loss", 0)),
                "epochs": trainer.current_epoch,
                "early_stopped": trainer.should_stop
            }
            
            logger.info(f"Training completed: {metrics}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            raise
            
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate predictions with uncertainty quantiles
        
        Args:
            df: DataFrame with features
            
        Returns:
            DataFrame with predictions and quantiles
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
            
        if self.training_data is None:
            raise ValueError("Training data not available. Model may not have been properly trained.")
            
        try:
            # Prepare prediction data
            df = df.copy()
            
            # Handle duplicate columns by keeping only the first occurrence
            df = df.loc[:, ~df.columns.duplicated()]
            
            df['time_idx'] = (pd.to_datetime(df.index) - pd.to_datetime(df.index.min())).days
            df['series'] = 'NIFTY'
            
            # Create prediction dataset
            pred_data = TimeSeriesDataSet.from_dataset(
                self.training_data,
                df,
                stop_randomization=True
            )
            
            pred_dataloader = pred_data.to_dataloader(
                train=False,
                batch_size=128,
                num_workers=0
            )
            
            # Make predictions
            raw_predictions = self.model.predict(
                pred_dataloader,
                mode="quantiles",
                return_x=False
            )
            
            # Extract quantiles
            predictions = raw_predictions.numpy()
            # predictions may come back as 1D, 2D, or 3D (n_series, prediction_length, quantiles)
            if predictions.ndim == 1:
                predictions = predictions.reshape(-1, 1)
            elif predictions.ndim == 3:
                # flatten series*time axis into rows x quantiles
                n, seq_len, q = predictions.shape
                predictions = predictions.reshape(n * seq_len, q)
            n_quantiles = predictions.shape[1]
            mid_idx = n_quantiles // 2 if n_quantiles > 1 else 0
            result = pd.DataFrame(
                {
                    'q05': predictions[:, 0],
                    'q50': predictions[:, mid_idx],
                    'q95': predictions[:, -1]
                },
                index=df.index[-len(predictions):]
            )
            
            logger.info(f"Generated predictions for {len(result)} time points")
            return result
            
        except Exception as e:
            logger.error(f"Error generating predictions: {str(e)}")
            raise
            
            return quantile_dict
            
        except Exception as e:
            print(f"   ⚠️ TFT prediction failed: {e}")
            return None
    
    def save_model(self, path: str) -> None:
        """
        Save model securely using safetensors format along with its configuration
        
        Args:
            path: Path to save the model
        """
        if self.model is None:
            raise ValueError("No model to save")
            
        try:
            path = Path(path)
            if not path.parent.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                
            # Ensure safetensors extension
            if path.suffix != '.safetensors':
                path = path.with_suffix('.safetensors')
                
            # Save model configuration
            config_path = path.with_suffix('.json')
            model_config = {
                'user_config': self.config.copy(),  # Save the full user configuration
                'model_hparams': {},  # Add empty dict to be populated below
                'dataset': {}
            }
            
            # Save model hyperparameters
            if self.model is not None and hasattr(self.model, 'hparams'):
                hparams = self.model.hparams
                model_config['model_hparams'] = {
                    name: getattr(hparams, name) 
                    for name in ['hidden_size', 'attention_head_size', 'dropout', 
                               'hidden_continuous_size', 'output_size', 'static_categoricals',
                               'static_reals', 'time_varying_known_reals', 
                               'time_varying_unknown_reals']
                    if hasattr(hparams, name)
                }
                model_config['model_hparams'].update({
                    'max_encoder_length': self.max_encoder_length,
                    'max_prediction_length': self.max_prediction_length
                })

            # If we have a prepared training dataset, save its configuration to allow exact reconstruction
            if getattr(self, 'training_data', None) is not None:
                ds = self.training_data
                # Attempt to read commonly used attributes; fall back silently if missing
                dataset_info = {}
                for attr in [
                    'static_categoricals', 'static_reals',
                    'time_varying_known_reals', 'time_varying_unknown_reals',
                    'min_encoder_length', 'max_encoder_length',
                    'min_prediction_length', 'max_prediction_length'
                ]:
                    val = getattr(ds, attr, None)
                    # Convert numpy arrays and other sequences to lists for JSON
                    if isinstance(val, (list, tuple)):
                        dataset_info[attr] = list(val)
                    elif isinstance(val, (np.ndarray,)):
                        dataset_info[attr] = val.tolist()
                    elif val is not None:
                        try:
                            dataset_info[attr] = list(val)
                        except Exception:
                            dataset_info[attr] = val

                model_config['dataset'] = dataset_info
            
            with open(config_path, 'w') as f:
                json.dump(model_config, f)
                
            # Get state dict and create a deep copy to handle shared tensors
            state_dict = self.model.state_dict()
            state_dict_copy = {k: v.clone().detach() for k, v in state_dict.items()}
            
            # Save using safetensors
            safetensors.torch.save_file(state_dict_copy, str(path))
            logger.info(f"Model saved securely to {path}")
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise
            
    def load_model(self, path: str, progress_callback: Callable[[str], None] = None) -> None:
        """
        Load model securely using safetensors format
        
        Args:
            path: Path to the model file
            progress_callback: Optional callback function for progress updates
        """
        try:
            if progress_callback:
                progress_callback(f"Loading TFT model from {path}")
                
            # Load configuration first
            config_path = Path(path).with_suffix('.json')
            saved_config = {}
            if config_path.exists():
                with open(config_path, 'r') as f:
                    saved_config = json.load(f)
                # Update instance configuration from saved config
                self.config.update(saved_config.get('user_config', {}))
                if 'model_hparams' in saved_config:
                    self.max_encoder_length = saved_config['model_hparams'].get('max_encoder_length', self.config['max_encoder_length'])
                    self.max_prediction_length = saved_config['model_hparams'].get('max_prediction_length', self.config['max_prediction_length'])
            
            # Load state dict securely
            state_dict = safe_load_model(path)
            if progress_callback:
                progress_callback("Verifying model state")
                
            if not state_dict:
                raise ValueError("Failed to load model state dict")
                
            # Initialize model if needed
            if self.model is None:
                # Create a minimal example dataset with all required columns and sufficient length
                example_data = pd.DataFrame({
                    'time_idx': list(range(32)),  # Ensure enough data points
                    'target': [0.0] * 32,
                    'series': ['NIFTY'] * 32,
                    'Close': [0.0] * 32,
                    'High': [0.0] * 32,
                    'Low': [0.0] * 32,
                    'Volume': [0.0] * 32,
                    'IndiaVIX': [0.0] * 32,
                    'Returns': [0.0] * 32
                })
                
                # Build dataset configuration from saved JSON (prefer saved dataset info)
                dataset_info = saved_config.get('dataset', {})
                # prefer values from dataset_info, then instance config
                static_categoricals = dataset_info.get('static_categoricals', self.config.get('static_categoricals', ['series']))
                static_reals = dataset_info.get('static_reals', self.config.get('static_reals', []))
                time_varying_known_reals = dataset_info.get('time_varying_known_reals', self.config.get('time_varying_known_reals', ['time_idx']))
                time_varying_unknown_reals = dataset_info.get('time_varying_unknown_reals', self.config.get('time_varying_unknown_reals', ['Close', 'High', 'Low', 'Volume', 'IndiaVIX', 'Returns']))

                min_encoder_length = dataset_info.get('min_encoder_length', max(2, int(self.max_encoder_length // 2)))
                max_encoder_length = dataset_info.get('max_encoder_length', self.max_encoder_length)
                min_prediction_length = dataset_info.get('min_prediction_length', 1)
                max_prediction_length = dataset_info.get('max_prediction_length', self.max_prediction_length if self.max_prediction_length is not None else 1)

                # Ensure our example data contains the exact columns required by the original dataset
                cols = set(['time_idx', 'target', 'series'])
                cols.update(static_categoricals or [])
                cols.update(static_reals or [])
                cols.update(time_varying_known_reals or [])
                cols.update(time_varying_unknown_reals or [])

                # build example_data with zeros for all required columns
                example_len = max( max_encoder_length, 32 )
                example_data = pd.DataFrame(index=range(example_len))
                example_data['time_idx'] = list(range(example_len))
                example_data['target'] = 0.0
                example_data['series'] = ['NIFTY'] * example_len
                for c in (time_varying_known_reals or []):
                    if c not in example_data.columns:
                        example_data[c] = 0.0
                for c in (time_varying_unknown_reals or []):
                    if c not in example_data.columns:
                        example_data[c] = 0.0
                for c in (static_reals or []):
                    if c not in example_data.columns:
                        example_data[c] = 0.0
                for c in (static_categoricals or []):
                    if c not in example_data.columns:
                        example_data[c] = ['NIFTY'] * example_len

                train_config = {
                    'time_idx': 'time_idx',
                    'target': 'target',
                    'group_ids': ['series'],
                    'static_categoricals': static_categoricals,
                    'static_reals': static_reals,
                    'time_varying_known_reals': time_varying_known_reals,
                    'time_varying_unknown_reals': time_varying_unknown_reals,
                    'min_encoder_length': min_encoder_length,
                    'max_encoder_length': max_encoder_length,
                    'min_prediction_length': min_prediction_length,
                    'max_prediction_length': max_prediction_length,
                    'add_relative_time_idx': True,
                    'add_target_scales': True,
                    'add_encoder_length': True
                }

                # Instantiate dataset and model so that model architecture matches saved state
                example_dataset = TimeSeriesDataSet(example_data, **train_config)
                self.model = TemporalFusionTransformer.from_dataset(
                    example_dataset,
                    hidden_size=self.config.get('hidden_size', 32),
                    attention_head_size=self.config.get('attention_head_size', 4),
                    dropout=self.config.get('dropout', 0.1),
                    hidden_continuous_size=self.config.get('hidden_continuous_size', 16),
                    output_size=self.config.get('output_size', 7),  # Number of quantiles
                    loss=QuantileLoss(),
                    log_interval=self.config.get('log_interval', 10),
                    reduce_on_plateau_patience=self.config.get('reduce_on_plateau_patience', 4)
                )
            
            # Load state dict
            self.model.load_state_dict(state_dict, strict=False)  # Use strict=False to allow partial loading
            logger.info(f"Model loaded securely from {path}")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
            
    def get_attention_weights(self):
        """Extract attention weights for interpretation"""
        if self.model is None:
            return None
            
        try:
            interpretation = self.model.interpret_output(
                self.training_data, attention_prediction_horizon=0
            )
            return interpretation['attention_weights'].numpy()
        except Exception as e:
            logger.error(f"Attention extraction failed: {e}")
            return None