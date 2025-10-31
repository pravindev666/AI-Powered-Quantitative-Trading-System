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
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers,
            num_layers=num_layers
        )
        
        # Output heads
        self.classification_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )
        
        # Loss functions
        self.classification_loss = nn.BCELoss()
        
        # Metrics
        self.train_accuracy = pl.metrics.Accuracy(task='binary')
        self.val_accuracy = pl.metrics.Accuracy(task='binary')
        
    def forward(self, x):
        # Project input to d_model dimensions
        x = self.input_projection(x)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Apply transformer encoder
        transformer_out = self.transformer_encoder(x)
        
        # Take the last sequence element for prediction
        last_hidden = transformer_out[:, -1, :]
        
        # Get predictions
        y_hat = self.classification_head(last_hidden)
        return y_hat
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.classification_loss(y_hat, y.unsqueeze(-1))
        self.train_accuracy(y_hat, y.int())
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', self.train_accuracy, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.classification_loss(y_hat, y.unsqueeze(-1))
        self.val_accuracy(y_hat, y.int())
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', self.val_accuracy, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=0.01
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.1,
            patience=5,
            verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss"
            }
        }

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        pe = pe.transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)