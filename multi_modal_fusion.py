"""
Multi-Modal Feature Fusion for Enhanced Prediction
Combines Technical, ML, TFT, and Sentiment signals
"""

import numpy as np
import pandas as pd
from scipy.special import expit  # sigmoid function

class MultiModalFusion:
    """
    Combines multiple prediction modalities into a unified forecast
    Uses dynamic weighting based on confidence and historical performance
    """
    
    def __init__(self, config=None):
        self.config = config or {}
        self.weights = self.config.get('fusion', {}).get('weights', {
            'technical': 0.25,
            'ml_ensemble': 0.35,
            'tft': 0.25,
            'sentiment': 0.15
        })
        
        # Initialize performance tracking
        self.component_performance = {
            'technical': {'correct': 0, 'total': 0},
            'ml_ensemble': {'correct': 0, 'total': 0},
            'tft': {'correct': 0, 'total': 0},
            'sentiment': {'correct': 0, 'total': 0}
        }
    
    def _normalize_weights(self, weights):
        """Ensure weights sum to 1.0"""
        total = sum(weights.values())
        if abs(total - 1.0) > 0.001:
            return {k: v/total for k, v in weights.items()}
        return weights
    
    def _calibrate_score_to_prob(self, score, volatility=15):
        """Convert raw score to probability with volatility adjustment"""
        z = score * 5.0 / (1 + volatility/20)
        prob = 1 / (1 + np.exp(-z))
        return np.clip(prob, 0.05, 0.95)
    
    def _dynamic_weight_adjustment(self, base_weights, confidences):
        """Adjust weights based on component confidence"""
        adjusted_weights = {}
        
        # Scale weights by confidence
        for component, weight in base_weights.items():
            conf = confidences.get(component, 0.5)
            adjusted_weights[component] = weight * conf
        
        # Normalize
        return self._normalize_weights(adjusted_weights)
    
    def _update_component_performance(self, components, actual_direction):
        """Track predictive performance of each component"""
        for component, pred in components.items():
            if 'direction' in pred:
                self.component_performance[component]['total'] += 1
                if np.sign(pred['direction']) == np.sign(actual_direction):
                    self.component_performance[component]['correct'] += 1
    
    def _get_component_weights(self):
        """Get weights adjusted for historical performance"""
        performance_weights = {}
        
        for component, stats in self.component_performance.items():
            if stats['total'] > 0:
                accuracy = stats['correct'] / stats['total']
                performance_weights[component] = accuracy
            else:
                performance_weights[component] = self.weights[component]
        
        return self._normalize_weights(performance_weights)
    
    def fuse_predictions(self, technical_pred, ml_pred=None, tft_pred=None, sentiment_score=0.0, 
                        volatility=15.0):
        """
        Combine predictions from multiple sources
        
        Args:
            technical_pred (dict): Technical analysis prediction
            ml_pred (dict): ML ensemble prediction
            tft_pred (dict): TFT quantile predictions
            sentiment_score (float): News sentiment score (-1 to +1)
            volatility (float): Current volatility estimate
            
        Returns:
            dict: Combined prediction with fusion details
        """
        components = {}
        confidences = {}
        
        # Process technical prediction
        if technical_pred:
            tech_score = technical_pred.get('technical_score', 0.0)
            tech_prob = self._calibrate_score_to_prob(tech_score, volatility)
            components['technical'] = {
                'score': tech_score,
                'probability': tech_prob,
                'direction': 1 if tech_prob > 0.5 else -1,
                'raw_confidence': abs(tech_prob - 0.5) * 2
            }
            confidences['technical'] = components['technical']['raw_confidence']
        
        # Process ML ensemble prediction
        if ml_pred:
            ml_direction = ml_pred.get('ensemble_prediction', 0)
            ml_conf = ml_pred.get('ensemble_confidence', 0.5)
            components['ml_ensemble'] = {
                'direction': ml_direction,
                'probability': ml_conf if ml_direction > 0 else (1 - ml_conf),
                'raw_confidence': ml_conf
            }
            confidences['ml_ensemble'] = ml_conf
        
        # Process TFT prediction
        if tft_pred:
            median = tft_pred['q_50']
            spread = tft_pred['q_75'] - tft_pred['q_25']
            tft_direction = 1 if median > 1.0 else -1
            tft_conf = 1.0 - min(spread, 0.5) * 2  # Lower spread = higher confidence
            
            components['tft'] = {
                'direction': tft_direction,
                'probability': 0.5 + (median - 1.0),
                'raw_confidence': tft_conf
            }
            confidences['tft'] = tft_conf
        
        # Process sentiment
        if abs(sentiment_score) > 0.001:
            sent_prob = 0.5 + (sentiment_score / 2)  # Convert -1/+1 to 0-1
            components['sentiment'] = {
                'direction': 1 if sentiment_score > 0 else -1,
                'probability': sent_prob,
                'raw_confidence': abs(sentiment_score)
            }
            confidences['sentiment'] = abs(sentiment_score)
        
        # Get base weights
        base_weights = {k: v for k, v in self.weights.items() 
                       if k in components}
        base_weights = self._normalize_weights(base_weights)
        
        # Get performance-adjusted weights
        perf_weights = self._get_component_weights()
        
        # Get confidence-adjusted weights
        conf_weights = self._dynamic_weight_adjustment(base_weights, confidences)
        
        # Calculate final fusion score
        fusion_score = 0.0
        final_weights = {}
        
        for component, pred in components.items():
            # Blend different weight types
            w_base = base_weights.get(component, 0)
            w_perf = perf_weights.get(component, 0)
            w_conf = conf_weights.get(component, 0)
            
            weight = (w_base + w_perf + w_conf) / 3
            final_weights[component] = weight
            
            # Add weighted contribution
            direction = pred.get('direction', 0)
            prob = pred.get('probability', 0.5)
            fusion_score += direction * (prob - 0.5) * 2 * weight
        
        # Normalize weights
        final_weights = self._normalize_weights(final_weights)
        
        # Convert fusion score to probability
        fusion_prob = self._calibrate_score_to_prob(fusion_score, volatility)
        
        # Determine final trend
        if fusion_prob > 0.5 + (0.15 / (1 + volatility/30)):
            final_trend = "UPTREND"
            final_direction = 1
            confidence = (fusion_prob - 0.5) * 200
        elif fusion_prob < 0.5 - (0.15 / (1 + volatility/30)):
            final_trend = "DOWNTREND"
            final_direction = -1
            confidence = (0.5 - fusion_prob) * 200
        else:
            final_trend = "SIDEWAYS"
            final_direction = 0
            confidence = 0
        
        return {
            'trend': final_trend,
            'direction': final_direction,
            'confidence': min(confidence, 100.0),
            'probability': fusion_prob,
            'fusion_score': fusion_score,
            'components': components,
            'weights': final_weights,
            'volatility_adjusted': True
        }