#!/usr/bin/env python3
"""
Daily automation script for sentiment caching and model retraining
Run via cron job: 0 16 * * 1-5  (4 PM on weekdays)
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from sentiment_analyzer import NewsSentimentAnalyzer
from model_monitor import ModelMonitor
import yaml
from datetime import datetime, timedelta
import json
import pandas as pd
import numpy as np

def load_config():
    """Load system configuration"""
    try:
        with open('config.yaml', 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"   ⚠️ Config loading failed: {e}")
        return {}

def daily_sentiment_update():
    """Update sentiment cache"""
    print("\n" + "="*60)
    print("📅 DAILY SENTIMENT UPDATE")
    print("="*60)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    analyzer = NewsSentimentAnalyzer()
    cache = analyzer.cache_daily_sentiment()
    
    print(f"✅ Sentiment cached for {len(cache)} days")
    return cache

def check_model_health():
    """Check model performance and drift"""
    print("\n" + "="*60)
    print("🔍 MODEL HEALTH CHECK")
    print("="*60)
    
    monitor = ModelMonitor()
    monitor.load_cached_metrics()
    
    drift_info = monitor.check_performance_drift()
    if drift_info['drift_detected']:
        monitor.generate_alert(drift_info)
        print("⚠️  Performance drift detected!")
    else:
        print("✅ No significant performance drift")
    
    # Plot performance trends
    monitor.plot_performance_trend()
    print("✅ Performance trends plotted")

def weekly_model_retrain():
    """Retrain models weekly (Friday)"""
    if datetime.now().weekday() != 4:  # 4 = Friday
        print("ℹ️  Skipping retrain (not Friday)")
        return
    
    print("\n" + "="*60)
    print("🔄 WEEKLY MODEL RETRAINING")
    print("="*60)
    
    config = load_config()
    
    try:
        # Get next Friday's expiry
        next_friday = datetime.now() + timedelta(days=(4-datetime.now().weekday())%7)
        expiry_date = next_friday.strftime('%Y-%m-%d')
        
        # Import here to avoid circular imports
        from niftypred import EnhancedNiftyPredictionSystem
        
        system = EnhancedNiftyPredictionSystem(expiry_date=expiry_date)
        system.load_data(days=365, force_refresh=True)
        system.calculate_indicators()
        system.train_ml_models()
        
        # Save models
        if hasattr(system, 'ml_ensemble') and system.ml_ensemble:
            model_path = Path('models')
            model_path.mkdir(exist_ok=True)
            
            # Save with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M')
            system.ml_ensemble.save(model_path / f'ensemble_{timestamp}.pkl')
            
            # Also save as latest
            system.ml_ensemble.save(model_path / 'latest_ensemble.pkl')
        
        print("✅ Models retrained and saved")
        
    except Exception as e:
        print(f"❌ Retraining failed: {e}")
        import traceback
        traceback.print_exc()

def cleanup_old_files():
    """Clean up old model files and logs"""
    print("\n" + "="*60)
    print("🧹 CLEANUP")
    print("="*60)
    
    try:
        # Keep only last 5 model versions
        model_path = Path('models')
        if model_path.exists():
            model_files = sorted(list(model_path.glob('ensemble_*.pkl')))
            if len(model_files) > 5:
                for f in model_files[:-5]:
                    f.unlink()
                print(f"✅ Cleaned up {len(model_files)-5} old model files")
        
        # Compress logs older than 7 days
        log_path = Path('logs')
        if log_path.exists():
            import tarfile
            from datetime import datetime, timedelta
            
            cutoff = datetime.now() - timedelta(days=7)
            
            for log_file in log_path.glob('*.log'):
                if log_file.stat().st_mtime < cutoff.timestamp():
                    with tarfile.open(f"{log_file}.tar.gz", "w:gz") as tar:
                        tar.add(log_file)
                    log_file.unlink()
            
            print("✅ Compressed old log files")
        
    except Exception as e:
        print(f"⚠️ Cleanup failed: {e}")

def main():
    """Main daily update routine"""
    try:
        start_time = datetime.now()
        print(f"🔄 Starting daily update at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Always update sentiment
        daily_sentiment_update()
        
        # Check model health
        check_model_health()
        
        # Retrain models on Friday
        weekly_model_retrain()
        
        # Cleanup old files
        cleanup_old_files()
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        print("\n✅ Daily update complete")
        print(f"⏱️  Duration: {duration:.1f} seconds")
        
    except Exception as e:
        print(f"\n❌ Daily update failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()