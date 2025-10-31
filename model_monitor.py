"""
Model Performance Monitoring and Drift Detection
"""

import numpy as np
import pandas as pd
from datetime import datetime
import json
from pathlib import Path
from sklearn.metrics import brier_score_loss, roc_auc_score, log_loss
import matplotlib.pyplot as plt

class ModelMonitor:
    """Monitor model performance and detect concept drift"""
    
    def __init__(self, alert_threshold=0.1):
        self.performance_history = []
        self.calibration_history = []
        self.alert_threshold = alert_threshold
        self.metrics_cache_file = Path('monitor_cache/metrics.json')
        self.metrics_cache_file.parent.mkdir(exist_ok=True)
    
    def record_performance(self, metrics):
        """Record performance metrics"""
        metrics['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.performance_history.append(metrics)
        
        # Keep last 90 days
        if len(self.performance_history) > 90:
            self.performance_history.pop(0)
        
        # Cache metrics
        self._cache_metrics()
    
    def _cache_metrics(self):
        """Cache metrics to disk"""
        try:
            with open(self.metrics_cache_file, 'w') as f:
                json.dump(self.performance_history, f, default=str, indent=2)
        except Exception as e:
            print(f"   ⚠️ Metrics caching failed: {e}")
    
    def load_cached_metrics(self):
        """Load cached metrics"""
        try:
            if self.metrics_cache_file.exists():
                with open(self.metrics_cache_file, 'r') as f:
                    self.performance_history = json.load(f)
                print(f"   ✅ Loaded {len(self.performance_history)} cached metrics")
        except Exception as e:
            print(f"   ⚠️ Metrics loading failed: {e}")
    
    def check_performance_drift(self):
        """Detect performance degradation"""
        if len(self.performance_history) < 20:
            return {'drift_detected': False, 'reason': 'Insufficient history'}
        
        recent = self.performance_history[-10:]
        older = self.performance_history[-30:-10]
        
        recent_metrics = {
            'accuracy': np.mean([m.get('accuracy', 0) for m in recent]),
            'auc': np.mean([m.get('auc', 0.5) for m in recent]),
            'logloss': np.mean([m.get('logloss', 0.69) for m in recent])
        }
        
        older_metrics = {
            'accuracy': np.mean([m.get('accuracy', 0) for m in older]),
            'auc': np.mean([m.get('auc', 0.5) for m in older]),
            'logloss': np.mean([m.get('logloss', 0.69) for m in older])
        }
        
        # Calculate drift for each metric
        drifts = {
            'accuracy': older_metrics['accuracy'] - recent_metrics['accuracy'],
            'auc': older_metrics['auc'] - recent_metrics['auc'],
            'logloss': recent_metrics['logloss'] - older_metrics['logloss']
        }
        
        max_drift_metric = max(drifts.items(), key=lambda x: abs(x[1]))
        
        if abs(max_drift_metric[1]) > self.alert_threshold:
            return {
                'drift_detected': True,
                'primary_metric': max_drift_metric[0],
                'drift_magnitude': max_drift_metric[1],
                'recent_metrics': recent_metrics,
                'older_metrics': older_metrics,
                'recommendation': 'RETRAIN MODEL'
            }
        
        return {'drift_detected': False, 'drift_magnitude': max(drifts.values())}
    
    def check_calibration_drift(self, y_true, y_proba):
        """Check probability calibration drift"""
        try:
            brier = brier_score_loss(y_true, y_proba)
            
            self.calibration_history.append({
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'brier_score': brier
            })
            
            if len(self.calibration_history) < 10:
                return {'calibration_drift': False}
            
            recent_brier = np.mean([c['brier_score'] for c in self.calibration_history[-5:]])
            older_brier = np.mean([c['brier_score'] for c in self.calibration_history[-10:-5]])
            
            if recent_brier > older_brier * 1.2:  # 20% worse
                return {
                    'calibration_drift': True,
                    'recent_brier': recent_brier,
                    'older_brier': older_brier,
                    'recommendation': 'RECALIBRATE MODEL'
                }
            
            return {'calibration_drift': False}
            
        except Exception as e:
            print(f"   ⚠️ Calibration check failed: {e}")
            return {'calibration_drift': False, 'error': str(e)}
    
    def plot_performance_trend(self, save_path='monitor_performance.png'):
        """Plot performance metrics over time"""
        try:
            if not self.performance_history:
                return
            
            dates = [pd.to_datetime(m['timestamp']) for m in self.performance_history]
            metrics = ['accuracy', 'auc', 'logloss']
            
            fig, axes = plt.subplots(len(metrics), 1, figsize=(12, 8))
            
            for i, metric in enumerate(metrics):
                values = [m.get(metric, 0) for m in self.performance_history]
                axes[i].plot(dates, values, marker='o')
                axes[i].set_ylabel(metric.upper())
                axes[i].grid(True, alpha=0.3)
                
                # Add trend line
                z = np.polyfit(range(len(dates)), values, 1)
                p = np.poly1d(z)
                axes[i].plot(dates, p(range(len(dates))), "r--", alpha=0.8)
            
            axes[-1].set_xlabel('Date')
            plt.tight_layout()
            plt.savefig(save_path)
            plt.close()
            
        except Exception as e:
            print(f"   ⚠️ Performance plotting failed: {e}")
    
    def generate_alert(self, drift_info):
        """Generate drift alert"""
        if drift_info.get('drift_detected') or drift_info.get('calibration_drift'):
            alert = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'type': 'MODEL_DRIFT',
                'severity': 'HIGH',
                'details': drift_info,
                'action_required': True
            }
            
            alert_file = Path('logs/model_alerts.json')
            alert_file.parent.mkdir(exist_ok=True)
            
            # Append alert to log
            try:
                if alert_file.exists():
                    with open(alert_file, 'r') as f:
                        alerts = json.load(f)
                else:
                    alerts = []
                
                alerts.append(alert)
                
                with open(alert_file, 'w') as f:
                    json.dump(alerts, f, indent=2, default=str)
                
            except Exception as e:
                print(f"   ⚠️ Alert logging failed: {e}")
            
            print("\n" + "="*60)
            print("⚠️  MODEL DRIFT ALERT")
            print("="*60)
            print(json.dumps(drift_info, indent=2))
            print("="*60 + "\n")
            
            return alert
        
        return None