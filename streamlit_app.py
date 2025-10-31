"""
Enhanced Streamlit Web UI for Nifty Prediction System
Run with: streamlit run streamlit_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import time
from app import EnhancedNiftyPredictionSystem

logger = logging.getLogger('nifty_predictor')

# Initialize session state
if 'timing_info' not in st.session_state:
    st.session_state.timing_info = {}
if 'last_prediction' not in st.session_state:
    st.session_state.last_prediction = None
if 'progress' not in st.session_state:
    st.session_state.progress = 0
if 'predictor' not in st.session_state:
    st.session_state.process_start_time = time.time()
    st.session_state.predictor = EnhancedNiftyPredictionSystem()
    st.session_state.status = 'System initialized'
    
# Initialize configuration state
if 'config' not in st.session_state:
    st.session_state.config = {
        'historical_days': 90,
        'force_refresh': False,
        'enable_tft': True,
        'enable_sentiment': True
    }

def update_progress(message: str):
    """Update progress bar and status message"""
    st.session_state.status = message
    if 'info_placeholder' in st.session_state:
        st.session_state.info_placeholder.info(f"🔄 {message}")
    
# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('nifty_prediction.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

# Page configuration
st.set_page_config(
    page_title="Nifty Prediction System",
    page_icon="📈",
    layout="wide"
)

# Main layout
st.title("🚀 Enhanced Nifty Option Prediction System v1.0.0 by Pravin A Mathew")

# Sidebar configuration
st.sidebar.header("⚙️ Configuration")

# Input fields
spot_price = st.sidebar.number_input(
    "Spot Price (0 = auto-fetch)",
    min_value=0.0,
    max_value=50000.0,
    value=0.0,
    step=50.0
)

# Training parameters
st.session_state.config['historical_days'] = st.sidebar.slider(
    "Historical Days for Training",
    min_value=30,
    max_value=365,
    value=st.session_state.config['historical_days'],
    help="Number of historical days to use for model training"
)

st.session_state.config['force_refresh'] = st.sidebar.checkbox(
    "Force Refresh Data",
    value=st.session_state.config['force_refresh'],
    help="Force refresh data from source instead of using cache"
)

# Expiry date selection (for option strategies)
st.sidebar.subheader("📅 Options Configuration")
next_friday = datetime.now() + timedelta(days=(4 - datetime.now().weekday()) % 7)
expiry_date = st.sidebar.date_input(
    "Expiry Date",
    value=next_friday,
    help="Target expiry date for option strategies"
)

# Model selection
st.sidebar.subheader("🤖 Model Selection")
st.session_state.config['enable_tft'] = st.sidebar.checkbox(
    "Enable TFT Model", 
    value=st.session_state.config['enable_tft'],
    help="Temporal Fusion Transformer for multi-horizon forecasting"
)
st.session_state.config['enable_har_rv'] = st.sidebar.checkbox(
    "Enable HAR-RV Model", 
    value=True,
    help="Heterogeneous Autoregressive Realized Volatility model"
)
st.session_state.config['enable_sentiment'] = st.sidebar.checkbox(
    "Enable Sentiment Analysis", 
    value=st.session_state.config['enable_sentiment'],
    help="News sentiment analysis for market mood"
)

# Main content area - Step by step flow
st.header("📊 Step 1: Market Data & Analysis")
    
# Data loading section
if st.button("🔄 Load Market Data", type="primary"):
    with st.spinner("Loading market data..."):
        try:
            update_progress("Fetching market data...")
            st.session_state.predictor.load_data(
                days=st.session_state.config['historical_days'],
                force_refresh=st.session_state.config['force_refresh']
            )
            update_progress("Calculating technical indicators...")
            st.session_state.predictor.calculate_indicators()
            update_progress("Generating signals...")
            st.session_state.predictor.generate_signals()
            st.success("✅ Market data loaded successfully!")
            
        except Exception as e:
            st.error(f"❌ Error loading data: {str(e)}")
            logger.error(f"Data loading error: {str(e)}")

# Display market data if available
if st.session_state.predictor.data is not None:
    st.subheader("📈 Current Market Data")
    
    # Latest market data
    latest_data = st.session_state.predictor.data.iloc[-1]
    current_price = latest_data['Close']
    current_vix = latest_data.get('IndiaVIX', 0)
    
    col1_1, col1_2, col1_3, col1_4 = st.columns(4)
    
    with col1_1:
        st.metric("Nifty 50", f"₹{current_price:,.2f}")
    with col1_2:
        st.metric("India VIX", f"{current_vix:.2f}")
    with col1_3:
        change = latest_data.get('Returns', 0) * 100
        st.metric("Daily Change", f"{change:+.2f}%")
    with col1_4:
        volume = latest_data.get('Volume', 0)
        st.metric("Volume", f"{volume:,.0f}")
    
    # Show technical indicators
    st.subheader("🔧 Technical Indicators")
    if hasattr(st.session_state.predictor, 'data') and st.session_state.predictor.data is not None:
        # Get indicator columns (exclude basic OHLCV data)
        basic_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Returns', 'IndiaVIX']
        indicator_cols = [col for col in st.session_state.predictor.data.columns 
                        if col not in basic_cols]
        
        if indicator_cols:
            # Show latest indicator values
            latest_indicators = st.session_state.predictor.data[indicator_cols].iloc[-1]
            
            # Create a nice display
            indicator_data = []
            for col in indicator_cols:
                value = latest_indicators[col]
                if pd.notna(value):
                    indicator_data.append({
                        'Indicator': col,
                        'Value': f"{value:.4f}" if isinstance(value, (int, float)) else str(value),
                        'Status': '✅' if pd.notna(value) else '❌'
                    })
            
            if indicator_data:
                indicator_df = pd.DataFrame(indicator_data)
                st.dataframe(indicator_df, width='stretch')
            else:
                st.info("No technical indicators calculated yet.")
        else:
            st.info("No technical indicators found. Click 'Load Market Data' to calculate them.")

# Step 2: Model Training Section
st.header("🤖 Step 2: Model Training & Performance")

if st.button("🏋️ Train Models"):
    if st.session_state.predictor.data is None:
        st.warning("⚠️ Please load market data first!")
    else:
        with st.spinner("Training models..."):
            try:
                update_progress("Training models...")
                st.session_state.predictor.train_ml_models()
                
                st.success("✅ Models trained successfully!")
                
                # Display model information
                st.subheader("📊 Model Status")
                
                # TFT Model
                if st.session_state.predictor.tft_model:
                    st.write("**TFT Model:** ✅ Available")
                    if hasattr(st.session_state.predictor.tft_model, 'is_trained'):
                        st.write(f"- Trained: {'✅ Yes' if st.session_state.predictor.tft_model.is_trained else '❌ No'}")
                else:
                    st.write("**TFT Model:** ❌ Not Available")
                
                # HAR-RV Model
                if st.session_state.predictor.har_rv_model:
                    st.write("**HAR-RV Model:** ✅ Available")
                    if hasattr(st.session_state.predictor.har_rv_model, 'is_trained'):
                        st.write(f"- Trained: {'✅ Yes' if st.session_state.predictor.har_rv_model.is_trained else '❌ No'}")
                else:
                    st.write("**HAR-RV Model:** ❌ Not Available")
                
                # Data summary
                if st.session_state.predictor.data is not None:
                    st.subheader("📈 Data Summary")
                    st.write(f"- **Data Points:** {len(st.session_state.predictor.data)}")
                    st.write(f"- **Date Range:** {st.session_state.predictor.data.index[0].strftime('%Y-%m-%d')} to {st.session_state.predictor.data.index[-1].strftime('%Y-%m-%d')}")
                    st.write(f"- **Features:** {len(st.session_state.predictor.data.columns)}")
                    
            except Exception as e:
                st.error(f"❌ Error training models: {str(e)}")
                logger.error(f"Training error: {str(e)}")

# Step 3: Prediction & Strategy
st.header("🎯 Step 3: Prediction & Strategy")

# Generate prediction button
if st.button("🔮 Generate Prediction", type="primary"):
    if st.session_state.predictor.data is None:
        st.warning("⚠️ Please load market data first!")
    else:
        with st.spinner("Generating prediction..."):
            try:
                # Update predictor with current settings
                st.session_state.predictor.spot_price = spot_price if spot_price > 0 else None
                st.session_state.predictor.expiry_date = str(expiry_date)
                
                prediction = st.session_state.predictor.predict_next_day()
                st.session_state.last_prediction = prediction
                
                # Display prediction results
                st.subheader("📊 Prediction Results")
                
                # Main prediction metrics
                pred_col1, pred_col2 = st.columns(2)
                
                with pred_col1:
                    st.metric(
                        "Trend",
                        f"{prediction['trend']} {prediction['trend_emoji']}"
                    )
                
                with pred_col2:
                    st.metric(
                        "Confidence",
                        f"{prediction['confidence']:.1f}%"
                    )
                    
                # Detailed analysis
                st.subheader("📈 Detailed Analysis")
                
                # Technical Signals
                if 'technical' in prediction.get('signals', {}):
                    tech_signals = prediction['signals']['technical']
                    st.write("**Technical Signals:**")
                    for signal, value in tech_signals.items():
                        st.write(f"- {signal}: {value}")
                
                # Sentiment Analysis
                if 'sentiment' in prediction.get('signals', {}):
                    sent_signals = prediction['signals']['sentiment']
                    st.write("**Market Sentiment:**")
                    st.write(f"- Score: {sent_signals.get('score', 'N/A')}")
                    st.write(f"- Confidence: {sent_signals.get('confidence', 'N/A')}")
                
                # Volatility Forecast
                if 'volatility' in prediction.get('signals', {}):
                    vol_signals = prediction['signals']['volatility']
                    st.write("**Volatility Forecast:**")
                    st.write(f"- Forecast: {vol_signals.get('forecast', 'N/A')}")
                
                # Option Strategy
                if 'option_strategy' in prediction:
                    st.subheader("📋 Recommended Option Strategy")
                    strategy = prediction['option_strategy']
                    st.write(f"**Strategy:** {strategy.get('strategy', 'N/A')}")
                    st.write(f"**Description:** {strategy.get('description', 'N/A')}")
                    
                    # Display strikes
                    if 'strikes' in strategy:
                        st.write("**Strikes:**")
                        strikes = strategy['strikes']
                        for strike_type, strike_price in strikes.items():
                            if isinstance(strike_price, (int, float)):
                                st.write(f"- {strike_type.replace('_', ' ').title()}: ₹{strike_price:,.0f}")
                            else:
                                st.write(f"- {strike_type.replace('_', ' ').title()}: {strike_price}")
                    
                    # Display strategy details
                    st.write(f"**Premium:** {strategy.get('premium', 'N/A')}")
                    st.write(f"**Max Profit:** {strategy.get('max_profit', 'N/A')}")
                    st.write(f"**Max Loss:** {strategy.get('max_loss', 'N/A')}")
                    st.write(f"**Breakeven:** {strategy.get('breakeven', 'N/A')}")
                
                # Position Sizing
                if 'position_size' in prediction:
                    st.subheader("💰 Position Sizing")
                    pos_size = prediction['position_size']
                    st.write(f"**Value:** ₹{pos_size.get('value', 0):,.2f}")
                    st.write(f"**Units:** {pos_size.get('units', 0):,.0f}")
                    st.write(f"**Capital %:** {pos_size.get('capital_pct', 0):.2f}%")
                    st.caption("💡 Position sizing shows how much to invest based on risk and confidence")
                
                st.success("✅ Prediction generated successfully!")
                
                # Store prediction in session state for backtest
                if st.session_state.last_prediction is not None:
                    st.session_state.prediction_for_backtest = st.session_state.last_prediction
            
            except Exception as e:
                st.error(f"❌ Error generating prediction: {str(e)}")
                logger.error(f"Prediction error: {str(e)}")

# Backtest Section (outside the button, persistent display)
if st.session_state.get('last_prediction') is not None:
    st.subheader("📊 Strategy Backtest")
    
    if st.button("🔄 Run Backtest"):
        if st.session_state.predictor.data is None:
            st.warning("⚠️ Please load market data first!")
        else:
            with st.spinner("Running backtest..."):
                try:
                    backtest_results = st.session_state.predictor.run_backtest()
                    
                    # Display backtest results
                    st.success("✅ Backtest completed!")
                    
                    # Show performance metrics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        total_return = (backtest_results['cum_strategy'].iloc[-1] - 1) * 100
                        st.metric("Strategy Return", f"{total_return:.2f}%")
                    with col2:
                        buyhold_return = (backtest_results['cum_buyhold'].iloc[-1] - 1) * 100
                        st.metric("Buy & Hold Return", f"{buyhold_return:.2f}%")
                    with col3:
                        alpha = total_return - buyhold_return
                        st.metric("Alpha", f"{alpha:.2f}%")
                    with col4:
                        sharpe = backtest_results['sharpe'].iloc[-1] if 'sharpe' in backtest_results.columns else 0
                        st.metric("Sharpe Ratio", f"{sharpe:.2f}")
                    
                    # Plot performance - show Iron Condor strategy value
                    if 'strategy_value' in backtest_results.columns:
                        st.line_chart(backtest_results[['strategy_value']])
                        st.caption("📈 Iron Condor Strategy Performance - This shows how the Iron Condor strategy would perform based on historical price movements within/outside the profit range")
                    else:
                        st.line_chart(backtest_results[['cum_strategy', 'cum_buyhold']])
                        st.caption("📈 Strategy vs Buy & Hold Performance")

                except Exception as e:
                    st.error(f"❌ Backtest failed: {str(e)}")

# Step 4: Data Visualization
if st.session_state.predictor.data is not None:
    st.header("📊 Data Visualization")
    
    # Create tabs for different visualizations - only Nifty 50 and India VIX
    viz_tab1, viz_tab2 = st.tabs(["📈 Nifty 50", "📊 India VIX"])
    
    with viz_tab1:
        st.subheader("📈 Nifty 50 Price Chart")
        price_data = st.session_state.predictor.data[['Close']].copy()
        st.line_chart(price_data)
        
        # Show statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Current Price", f"₹{price_data['Close'].iloc[-1]:,.2f}")
        with col2:
            price_change = price_data['Close'].pct_change().iloc[-1] * 100
            st.metric("Daily Change", f"{price_change:+.2f}%")
        with col3:
            price_range = price_data['Close'].max() - price_data['Close'].min()
            st.metric("Price Range", f"₹{price_range:,.2f}")
    
    with viz_tab2:
        st.subheader("📊 India VIX (Volatility Index)")
        if 'IndiaVIX' in st.session_state.predictor.data.columns:
            vix_data = st.session_state.predictor.data[['IndiaVIX']].copy()
            st.line_chart(vix_data)
            
            # Show statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Current VIX", f"{vix_data['IndiaVIX'].iloc[-1]:.2f}")
            with col2:
                avg_vix = vix_data['IndiaVIX'].mean()
                st.metric("Average VIX", f"{avg_vix:.2f}")
            with col3:
                st.caption("⚠️ Higher VIX = Higher market volatility")
        else:
            st.info("VIX data not available")

# Footer
st.markdown("---")
st.markdown("**Enhanced Nifty Option Prediction System v1.0.0 by Pravin A Mathew** | Built with Streamlit")