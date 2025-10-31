"""Time Series Model Implementation"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np
from torch.utils.data import Dataset, DataLoader
from pytorch_forecasting.metrics import SMAPE, MAE, RMSE
import warnings
warnings.filterwarnings('ignore')

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y, sequence_length):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        self.sequence_length = sequence_length
        
    def __len__(self):
        return len(self.X) - self.sequence_length
    
    def __getitem__(self, idx):
        X_seq = self.X[idx:idx + self.sequence_length]
        y_target = self.y[idx + self.sequence_length - 1]
        return X_seq, y_target

class TimeSeriesTransformer(pl.LightningModule):
    def __init__(
        self, 
        input_dim,
        d_model=64,
        nhead=4,
        num_layers=2,
        dropout=0.1,
        learning_rate=1e-4
    ):
        super().__init__()
        
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # Transformer encoder
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu'
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers,
            num_layers=num_layers
        )
        
        # Output layer
        self.output_layer = nn.Linear(d_model, 1)
        
    def forward(self, x):
        # Shape: [batch, seq_len, features]
        x = self.input_projection(x)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Transformer expects [seq_len, batch, features]
        x = x.permute(1, 0, 2)
        
        # Pass through transformer
        x = self.transformer_encoder(x)
        
        # Back to [batch, seq_len, features]
        x = x.permute(1, 0, 2)
        
        # Get predictions for each time step
        x = self.output_layer(x)
        return x
    
    def training_step(self, batch, batch_idx):
        X, y = batch
        y_hat = self(X).squeeze(-1)
        loss = nn.MSELoss()(y_hat, y)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        X, y = batch
        y_hat = self(X).squeeze(-1)
        loss = nn.MSELoss()(y_hat, y)
        self.log('val_loss', loss)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_loss'
        }

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)