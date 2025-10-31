"""Results output and persistence module for the prediction system"""
import json
from datetime import datetime
from pathlib import Path
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class ResultsManager:
    """Handles saving and formatting prediction results"""
    
    def __init__(self, base_dir='predictions'):
        """Initialize with base directory for outputs"""
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._setup_logging()
    
    def _setup_logging(self):
        """Configure logging for the results manager"""
        log_dir = self.base_dir / 'logs'
        log_dir.mkdir(exist_ok=True)
        
        log_file = log_dir / f'results_{datetime.now().strftime("%Y%m%d")}.log'
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        )
        logger.addHandler(file_handler)
    
    def _get_timestamp(self):
        """Get formatted timestamp for filenames"""
        return datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def save_summary_csv(self, prediction):
        """Save prediction summary to CSV"""
        try:
            timestamp = self._get_timestamp()
            summary_csv = self.base_dir / f"summary_{timestamp}.csv"
            
            # Extract prediction data safely with defaults
            summary_data = {
                'Timestamp': [timestamp],
                'Trend': [prediction.get('trend', 'UNKNOWN')],
                'Confidence': [prediction.get('confidence', 0.0)],
                'Technical_Score': [prediction.get('technical_score', 0.0)],
                'Current_Price': [prediction.get('current_price', 0.0)],
                'ATR': [prediction.get('atr', 0.0)],
                'India_VIX': [prediction.get('market_conditions', {}).get('india_vix', 0.0)],
                'ADX': [prediction.get('market_conditions', {}).get('adx', 0.0)],
                'RSI': [prediction.get('market_conditions', {}).get('rsi', 0.0)],
                'Regime': [prediction.get('regime_detection', {}).get('regime', 'UNKNOWN')],
                'Strategy': [prediction.get('option_strategy', {}).get('name', 'N/A')],
                'Position_Size': [prediction.get('position_size', {}).get('recommended_size', 0.0) * 100]
            }
            
            # Add ML predictions if available
            if 'ml_predictions' in prediction:
                ml_pred = prediction['ml_predictions']
                summary_data.update({
                    'ML_Prediction': ['Bullish' if ml_pred.get('ensemble_prediction', 0) > 0 else 'Bearish'],
                    'ML_Confidence': [ml_pred.get('ensemble_confidence', 0.0) * 100]
                })
            
            # Create and save DataFrame
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_csv(summary_csv, index=False)
            
            logger.info(f"Summary saved to: {summary_csv}")
            print(f"\n📊 Summary saved to: {summary_csv}")
            return summary_csv
            
        except Exception as e:
            logger.error(f"Failed to save summary CSV: {e}")
            print(f"\n⚠️ Summary CSV save failed: {e}")
            return None
    
    def save_prediction_json(self, prediction, expiry_date=None):
        """Save full prediction details to JSON"""
        try:
            timestamp = self._get_timestamp()
            filename = self.base_dir / f"prediction_{timestamp}.json"
            
            # Prepare serializable prediction data
            pred_to_save = {
                'timestamp': timestamp,
                'trend': prediction.get('trend', 'UNKNOWN'),
                'confidence': float(prediction.get('confidence', 0.0)),
                'current_price': float(prediction.get('current_price', 0.0)),
                'expiry_date': expiry_date,
                'technical_score': float(prediction.get('technical_score', 0.0)),
                'market_conditions': prediction.get('market_conditions', {}),
                'regime_detection': prediction.get('regime_detection', {}),
                'option_strategy': {
                    'name': prediction.get('option_strategy', {}).get('name', 'N/A'),
                    'strikes': prediction.get('option_strategy', {}).get('strikes', {}),
                    'explanation': prediction.get('option_strategy', {}).get('explanation', '')
                },
                'position_size': prediction.get('position_size', {
                    'recommended_size': 0.0,
                    'volatility_factor': 1.0
                })
            }
            
            # Add ML predictions if available
            if 'ml_predictions' in prediction:
                pred_to_save['ml_predictions'] = {
                    'direction': 'Bullish' if prediction['ml_predictions'].get('ensemble_prediction', 0) > 0 else 'Bearish',
                    'confidence': float(prediction['ml_predictions'].get('ensemble_confidence', 0.0))
                }
            
            # Save with pretty formatting
            with open(filename, 'w') as f:
                json.dump(pred_to_save, f, indent=2, default=str)
            
            logger.info(f"Full prediction saved to: {filename}")
            print(f"   ✅ Prediction saved to: {filename}")
            return filename
            
        except Exception as e:
            logger.error(f"Failed to save prediction JSON: {e}")
            print(f"\n⚠️ Prediction save failed: {e}")
            return None
    
    def print_prediction_summary(self, prediction):
        """Display formatted prediction summary"""
        print("\n" + "=" * 60)
        print("✅ ANALYSIS COMPLETE!")
        print("=" * 60)
        
        # Core prediction
        print(f"\n   Trend: {prediction.get('trend', 'UNKNOWN')} {prediction.get('trend_emoji', '')}")
        print(f"   Confidence: {prediction.get('confidence', 0.0):.1f}%")
        print(f"   Technical Score: {prediction.get('technical_score', 0.0):.2f}")
        
        # Market conditions
        market_conditions = prediction.get('market_conditions', {})
        print("\n   Market Conditions:")
        print(f"      VIX: {market_conditions.get('india_vix', 0.0):.1f}")
        print(f"      ADX: {market_conditions.get('adx', 0.0):.1f}")
        print(f"      RSI: {market_conditions.get('rsi', 0.0):.1f}")
        
        # Option strategy
        strategy = prediction.get('option_strategy', {})
        print(f"\n   Strategy: {strategy.get('name', 'N/A')}")
        if 'explanation' in strategy:
            print(f"      → {strategy['explanation']}")
        
        # ML predictions if available
        if 'ml_predictions' in prediction:
            ml_pred = prediction['ml_predictions']
            direction = 'Bullish' if ml_pred.get('ensemble_prediction', 0) > 0 else 'Bearish'
            conf = ml_pred.get('ensemble_confidence', 0.0) * 100
            print(f"\n   ML Prediction: {direction} ({conf:.1f}% confidence)")
        
        # Position sizing
        pos_size = prediction.get('position_size', {}).get('recommended_size', 0.0) * 100
        print(f"\n   Position Size: {pos_size:.1f}% of max")
        
        print("\n⚠️  DISCLAIMER:")
        print("   • Educational purposes only")
        print("   • Options involve substantial risk")
        print("   • Always use proper risk management")
        print("   • Consult a financial advisor")
        print("\n" + "=" * 60)