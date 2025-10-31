"""Data visualization utilities"""
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from typing import Dict, Optional

class Visualizer:
    """Data visualization tools"""
    
    def __init__(self):
        self.theme = "plotly_white"
        self.height = 600

    def plot_price_and_signals(self, data: pd.DataFrame, signals: Dict) -> go.Figure:
        """Plot price chart with technical indicators and signals"""
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            row_heights=[0.7, 0.3]
        )
        
        # Candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=data.index,
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'],
                name='NIFTY'
            ),
            row=1, col=1
        )
        
        # Add Moving Averages
        if 'SMA20' in data.columns:
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['SMA20'],
                    name='SMA20',
                    line=dict(color='blue', width=1)
                ),
                row=1, col=1
            )
        
        if 'SMA50' in data.columns:
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['SMA50'],
                    name='SMA50',
                    line=dict(color='red', width=1)
                ),
                row=1, col=1
            )
        
        # Add RSI
        if 'RSI' in data.columns:
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data['RSI'],
                    name='RSI',
                    line=dict(color='purple', width=1)
                ),
                row=2, col=1
            )
            
            # Add RSI levels
            fig.add_hline(y=70, line_dash="dot", line_color="red", row=2, col=1)
            fig.add_hline(y=30, line_dash="dot", line_color="green", row=2, col=1)
        
        # Update layout
        fig.update_layout(
            title='Nifty Price Chart with Indicators',
            yaxis_title='Price',
            yaxis2_title='RSI',
            xaxis_rangeslider_visible=False,
            height=self.height,
            template=self.theme
        )
        
        return fig

    def plot_volatility_forecast(self, data: pd.DataFrame, forecasts: pd.Series) -> go.Figure:
        """Plot volatility forecasts"""
        fig = go.Figure()
        
        # Historical volatility
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data['IndiaVIX'],
                name='India VIX',
                line=dict(color='blue', width=1)
            )
        )
        
        # Forecasted volatility
        if not forecasts.empty:
            fig.add_trace(
                go.Scatter(
                    x=forecasts.index,
                    y=forecasts,
                    name='Forecast',
                    line=dict(color='red', width=1, dash='dot')
                )
            )
        
        fig.update_layout(
            title='Volatility Forecast',
            yaxis_title='Volatility (%)',
            height=self.height,
            template=self.theme,
            showlegend=True
        )
        
        return fig

    def plot_signals_dashboard(self, signals: Dict) -> go.Figure:
        """Create a dashboard of trading signals"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Technical Signals',
                'Sentiment Analysis',
                'Volatility',
                'Combined Signal'
            )
        )
        
        # Technical Signals
        if 'technical' in signals:
            tech = signals['technical']
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number",
                    value=tech.get('combined', 0) * 100,
                    title={'text': "Technical Score"},
                    gauge={'axis': {'range': [0, 100]}},
                ),
                row=1, col=1
            )
        
        # Sentiment
        if 'sentiment' in signals:
            sent = signals['sentiment']
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number",
                    value=sent.get('score', 0) * 100,
                    title={'text': "Sentiment Score"},
                    gauge={'axis': {'range': [0, 100]}},
                ),
                row=1, col=2
            )
        
        # Volatility
        if 'volatility' in signals:
            vol = signals['volatility']
            fig.add_trace(
                go.Indicator(
                    mode="gauge+number",
                    value=vol.get('current', 0),
                    title={'text': "India VIX"},
                    gauge={'axis': {'range': [0, 40]}},
                ),
                row=2, col=1
            )
        
        # Combined Signal
        prob = signals.get('probability', 0.5)
        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=prob * 100,
                title={'text': "Signal Strength"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'steps': [
                        {'range': [0, 40], 'color': "red"},
                        {'range': [40, 60], 'color': "yellow"},
                        {'range': [60, 100], 'color': "green"}
                    ]
                },
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            height=800,
            template=self.theme,
            showlegend=False,
            title_text="Trading Signals Dashboard"
        )
        
        return fig