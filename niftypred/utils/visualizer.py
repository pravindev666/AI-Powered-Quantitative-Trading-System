"""
Enhanced visualization utilities for TFT and sentiment analysis
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import seaborn as sns

class Visualizer:
    """Enhanced visualization tools"""
    
    @staticmethod
    def plot_tft_quantile_forecast(prediction_dict, save_path='plots/tft_forecast.png'):
        """Plot TFT quantile predictions as fan chart"""
        try:
            if 'ml_predictions' not in prediction_dict:
                return
            
            ml_pred = prediction_dict['ml_predictions']
            if 'TFT_quantiles' not in ml_pred.get('individual', {}):
                return
            
            quantiles = ml_pred['individual']['TFT_quantiles']
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            current = prediction_dict['current_price']
            
            x = [0, 1]  # Today, Tomorrow
            
            # Fill between quantiles
            colors = sns.color_palette("husl", 3)
            alphas = [0.1, 0.2, 0.3]
            
            for i, (q_low, q_high) in enumerate([
                ('q_02', 'q_98'),
                ('q_10', 'q_90'),
                ('q_25', 'q_75')
            ]):
                y_low = [current, quantiles[q_low] * current]
                y_high = [current, quantiles[q_high] * current]
                
                ax.fill_between(x, y_low, y_high, 
                              color=colors[i], alpha=alphas[i],
                              label=f'{q_high[2:]}% CI')
            
            # Median prediction
            median = [current, quantiles['q_50'] * current]
            ax.plot(x, median, 'r-', linewidth=2, label='Median Forecast')
            
            ax.set_xlabel('Time Horizon')
            ax.set_ylabel('Price Level')
            ax.set_title('TFT Quantile Forecast (Probabilistic Fan Chart)')
            ax.set_xticks([0, 1])
            ax.set_xticklabels(['Today', 'Tomorrow'])
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)
            
            # Add price labels
            for i, price in enumerate([current, median[1]]):
                ax.annotate(f'₹{price:,.2f}', 
                          (x[i], price),
                          xytext=(5, 5),
                          textcoords='offset points')
            
            # Save plot
            save_path = Path(save_path)
            save_path.parent.mkdir(exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"   ✅ TFT quantile plot saved to: {save_path}")
            plt.close()
            
        except ImportError:
            print("   ⚠️ Required visualization packages not available")
        except Exception as e:
            print(f"   ⚠️ TFT plotting failed: {e}")
    
    @staticmethod
    def plot_sentiment_timeline(sentiment_df, save_path='plots/sentiment_timeline.png'):
        """Plot historical sentiment scores"""
        try:
            fig, ax = plt.subplots(figsize=(14, 6))
            
            ax.plot(sentiment_df.index, sentiment_df['Sentiment'], 
                   linewidth=1.5, color='blue', alpha=0.7)
            ax.axhline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.5)
            
            # Color positive/negative regions
            ax.fill_between(sentiment_df.index, 0, sentiment_df['Sentiment'], 
                          where=(sentiment_df['Sentiment'] > 0), 
                          color='green', alpha=0.3, label='Positive')
            ax.fill_between(sentiment_df.index, 0, sentiment_df['Sentiment'], 
                          where=(sentiment_df['Sentiment'] <= 0), 
                          color='red', alpha=0.3, label='Negative')
            
            ax.set_xlabel('Date')
            ax.set_ylabel('Sentiment Score')
            ax.set_title('News Sentiment Timeline (FinBERT)')
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)
            
            # Add trend line
            z = np.polyfit(range(len(sentiment_df)), sentiment_df['Sentiment'], 1)
            p = np.poly1d(z)
            ax.plot(sentiment_df.index, p(range(len(sentiment_df))), 
                   "k--", alpha=0.5, label='Trend')
            
            # Save plot
            save_path = Path(save_path)
            save_path.parent.mkdir(exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"   ✅ Sentiment timeline saved to: {save_path}")
            plt.close()
            
        except ImportError:
            print("   ⚠️ Required visualization packages not available")
        except Exception as e:
            print(f"   ⚠️ Sentiment plotting failed: {e}")
    
    @staticmethod
    def plot_tft_attention(attention_weights, feature_names, save_path='plots/tft_attention.png'):
        """Plot TFT attention patterns"""
        try:
            fig, ax = plt.subplots(figsize=(12, 6))
            
            sns.heatmap(attention_weights.T, 
                       xticklabels=range(-len(feature_names)+1, 1),
                       yticklabels=feature_names,
                       cmap='YlOrRd',
                       ax=ax)
            
            ax.set_xlabel('Time Lag (t-n to t)')
            ax.set_ylabel('Features')
            ax.set_title('TFT Variable Attention Weights')
            
            # Save plot
            save_path = Path(save_path)
            save_path.parent.mkdir(exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"   ✅ Attention visualization saved to: {save_path}")
            plt.close()
            
        except ImportError:
            print("   ⚠️ Required visualization packages not available")
        except Exception as e:
            print(f"   ⚠️ Attention visualization failed: {e}")
    
    @staticmethod
    def plot_performance_metrics(metrics_history, save_path='plots/performance_metrics.png'):
        """Plot model performance metrics over time"""
        try:
            metrics = ['accuracy', 'auc', 'logloss']
            fig, axes = plt.subplots(len(metrics), 1, figsize=(12, 8))
            
            dates = [pd.to_datetime(m['timestamp']) for m in metrics_history]
            
            for i, metric in enumerate(metrics):
                values = [m.get(metric, 0) for m in metrics_history]
                
                axes[i].plot(dates, values, marker='o', markersize=4)
                axes[i].set_ylabel(metric.upper())
                axes[i].grid(True, alpha=0.3)
                
                # Add trend line
                z = np.polyfit(range(len(dates)), values, 1)
                p = np.poly1d(z)
                trend = p(range(len(dates)))
                axes[i].plot(dates, trend, "r--", alpha=0.8, label='Trend')
                
                # Add mean line
                mean_val = np.mean(values)
                axes[i].axhline(mean_val, color='g', linestyle=':', alpha=0.8,
                              label=f'Mean ({mean_val:.3f})')
                
                axes[i].legend()
            
            axes[-1].set_xlabel('Date')
            plt.suptitle('Model Performance Metrics Over Time')
            plt.tight_layout()
            
            # Save plot
            save_path = Path(save_path)
            save_path.parent.mkdir(exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"   ✅ Performance metrics plot saved to: {save_path}")
            plt.close()
            
        except ImportError:
            print("   ⚠️ Required visualization packages not available")
        except Exception as e:
            print(f"   ⚠️ Performance plotting failed: {e}")
    
    @staticmethod
    def plot_calibration_curve(y_true, y_prob, save_path='plots/calibration_curve.png'):
        """Plot probability calibration curve"""
        try:
            from sklearn.calibration import calibration_curve
            
            fig, ax = plt.subplots(figsize=(8, 8))
            
            # Plot perfectly calibrated
            ax.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
            
            # Plot calibration curve
            prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)
            ax.plot(prob_pred, prob_true, "s-", label="Model")
            
            ax.set_xlabel("Mean predicted probability")
            ax.set_ylabel("Fraction of positives")
            ax.set_title("Calibration Curve")
            ax.legend(loc="best")
            ax.grid(True, alpha=0.3)
            
            # Save plot
            save_path = Path(save_path)
            save_path.parent.mkdir(exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"   ✅ Calibration curve saved to: {save_path}")
            plt.close()
            
        except ImportError:
            print("   ⚠️ Required visualization packages not available")
        except Exception as e:
            print(f"   ⚠️ Calibration plotting failed: {e}")