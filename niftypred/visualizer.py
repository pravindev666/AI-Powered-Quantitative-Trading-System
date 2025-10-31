"""Data visualization utilities."""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from typing import Dict, List, Optional, Tuple
import numpy as np

class Visualizer:
    """Visualization tools for trading system."""
    
    def __init__(self):
        self.color_scheme = {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e',
            'success': '#2ca02c',
            'danger': '#d62728',
            'info': '#17becf'
        }
    
    def plot_market_data(self, data: pd.DataFrame, indicators: Optional[Dict] = None) -> go.Figure:
        """Plot market data with technical indicators."""
        fig = make_subplots(rows=2, cols=1, 
                          shared_xaxes=True,
                          vertical_spacing=0.03,
                          row_heights=[0.7, 0.3])
        
        # Price candlestick
        fig.add_trace(
            go.Candlestick(
                x=data.index,
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'],
                name='OHLC'
            ),
            row=1, col=1
        )
        
        # Volume
        fig.add_trace(
            go.Bar(
                x=data.index,
                y=data['Volume'],
                name='Volume',
                opacity=0.7
            ),
            row=2, col=1
        )
        
        # Add indicators if provided
        if indicators:
            for name, values in indicators.items():
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=values,
                        name=name,
                        line=dict(width=1)
                    ),
                    row=1, col=1
                )
        
        fig.update_layout(
            title='Market Data Analysis',
            yaxis_title='Price',
            yaxis2_title='Volume',
            xaxis_rangeslider_visible=False
        )
        
        return fig
    
    def plot_sentiment_analysis(self, sentiment_data: Dict) -> go.Figure:
        """Plot sentiment analysis results."""
        dates = [pd.to_datetime(item['timestamp']) for item in sentiment_data['history']]
        sentiments = [item['sentiment'] for item in sentiment_data['history']]
        confidences = [item['confidence'] for item in sentiment_data['history']]
        
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=sentiments,
                name="Sentiment",
                line=dict(color=self.color_scheme['primary'])
            )
        )
        
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=confidences,
                name="Confidence",
                line=dict(color=self.color_scheme['secondary']),
                opacity=0.7
            ),
            secondary_y=True
        )
        
        fig.update_layout(
            title='Sentiment Analysis Over Time',
            yaxis_title='Sentiment Score',
            yaxis2_title='Confidence'
        )
        
        return fig
    
    def plot_prediction_intervals(self, 
                                predictions: pd.DataFrame,
                                actual: Optional[pd.Series] = None) -> go.Figure:
        """Plot prediction intervals with actual values if available."""
        fig = go.Figure()
        
        # Plot prediction intervals
        fig.add_trace(
            go.Scatter(
                x=predictions.index,
                y=predictions['q95'],
                name='Upper 95%',
                line=dict(width=0),
                showlegend=False
            )
        )
        
        fig.add_trace(
            go.Scatter(
                x=predictions.index,
                y=predictions['q05'],
                name='Lower 95%',
                fill='tonexty',
                fillcolor='rgba(31, 119, 180, 0.2)',
                line=dict(width=0),
                showlegend=False
            )
        )
        
        # Plot median prediction
        fig.add_trace(
            go.Scatter(
                x=predictions.index,
                y=predictions['q50'],
                name='Median Prediction',
                line=dict(color=self.color_scheme['primary'])
            )
        )
        
        # Add actual values if provided
        if actual is not None:
            fig.add_trace(
                go.Scatter(
                    x=actual.index,
                    y=actual,
                    name='Actual',
                    line=dict(color=self.color_scheme['secondary'])
                )
            )
        
        fig.update_layout(
            title='Prediction Intervals',
            yaxis_title='Value',
            showlegend=True
        )
        
        return fig
    
    def plot_performance_metrics(self, metrics: Dict) -> go.Figure:
        """Plot trading performance metrics."""
        fig = make_subplots(rows=2, cols=2,
                          subplot_titles=('Cumulative Returns', 'Drawdown',
                                        'Monthly Returns', 'Rolling Sharpe Ratio'))
        
        # Cumulative returns
        fig.add_trace(
            go.Scatter(
                x=metrics['cumulative_returns'].index,
                y=metrics['cumulative_returns'].values,
                name='Cumulative Returns',
                line=dict(color=self.color_scheme['primary'])
            ),
            row=1, col=1
        )
        
        # Drawdown
        fig.add_trace(
            go.Scatter(
                x=metrics['drawdown'].index,
                y=metrics['drawdown'].values,
                name='Drawdown',
                fill='tozeroy',
                line=dict(color=self.color_scheme['danger'])
            ),
            row=1, col=2
        )
        
        # Monthly returns heatmap
        monthly_returns = metrics['monthly_returns'].pivot_table(
            index=metrics['monthly_returns'].index.year,
            columns=metrics['monthly_returns'].index.month,
            values='returns'
        )
        
        fig.add_trace(
            go.Heatmap(
                z=monthly_returns.values,
                x=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
                y=monthly_returns.index,
                name='Monthly Returns'
            ),
            row=2, col=1
        )
        
        # Rolling Sharpe ratio
        fig.add_trace(
            go.Scatter(
                x=metrics['rolling_sharpe'].index,
                y=metrics['rolling_sharpe'].values,
                name='Rolling Sharpe',
                line=dict(color=self.color_scheme['info'])
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            height=800,
            title='Trading Performance Metrics',
            showlegend=True
        )
        
        return fig
    
    def plot_risk_analysis(self, risk_metrics: Dict) -> go.Figure:
        """Plot risk analysis metrics."""
        fig = make_subplots(rows=2, cols=2,
                          subplot_titles=('Value at Risk', 'Risk Contribution',
                                        'Risk-Return Scatter', 'Correlation Matrix'))
        
        # Value at Risk
        fig.add_trace(
            go.Scatter(
                x=risk_metrics['var_series'].index,
                y=risk_metrics['var_series'].values,
                name='VaR (95%)',
                line=dict(color=self.color_scheme['danger'])
            ),
            row=1, col=1
        )
        
        # Risk Contribution
        fig.add_trace(
            go.Bar(
                x=list(risk_metrics['risk_contribution'].keys()),
                y=list(risk_metrics['risk_contribution'].values()),
                name='Risk Contribution',
                marker_color=self.color_scheme['primary']
            ),
            row=1, col=2
        )
        
        # Risk-Return Scatter
        fig.add_trace(
            go.Scatter(
                x=risk_metrics['volatility'],
                y=risk_metrics['returns'],
                mode='markers+text',
                text=risk_metrics['assets'],
                textposition='top center',
                name='Risk-Return',
                marker=dict(color=self.color_scheme['info'])
            ),
            row=2, col=1
        )
        
        # Correlation Matrix
        fig.add_trace(
            go.Heatmap(
                z=risk_metrics['correlation'].values,
                x=risk_metrics['correlation'].index,
                y=risk_metrics['correlation'].columns,
                name='Correlation'
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            height=800,
            title='Risk Analysis Dashboard',
            showlegend=True
        )
        
        return fig