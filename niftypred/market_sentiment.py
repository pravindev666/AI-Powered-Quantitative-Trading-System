"""Market sentiment analysis module."""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
import pandas as pd
import numpy as np
from typing import Dict, Optional, List
import logging
from datetime import datetime, timedelta
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob
from gnews import GNews
import re

logger = logging.getLogger(__name__)

class FinBERTSentiment(nn.Module):
    """FinBERT-based sentiment analysis model."""
    
    def __init__(self, model_name: str = 'ProsusAI/finbert'):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(768, 3)  # 3 classes: positive, negative, neutral
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]
        x = self.dropout(pooled_output)
        return self.fc(x)

class MarketSentimentAnalyzer:
    """Analyzes market sentiment from multiple sources."""
    
    def __init__(self, api_keys: Optional[Dict] = None):
        self.vader = SentimentIntensityAnalyzer()
        self.gnews = GNews(language='en', country='IN', period='1d', max_results=10)
        
        # Keywords for market sentiment
        self.pos_keywords = ['bullish', 'rally', 'growth', 'positive', 'uptrend']
        self.neg_keywords = ['bearish', 'decline', 'negative', 'downtrend', 'crash']
        
    def analyze_news_sentiment(self, symbol: str, lookback_days: int = 7) -> Dict:
        """Analyze sentiment from recent news articles."""
        try:
            news_items = self._fetch_news(symbol, lookback_days)
            if not news_items:
                return {'sentiment': 0, 'confidence': 0}
            
            sentiments = []
            for item in news_items:
                title = item.get('title', '')
                text = item.get('description', '')
                combined_text = f"{title}. {text}"
                sentiment = self._analyze_text(combined_text)
                sentiments.append(sentiment)
            
            df = pd.DataFrame(sentiments)
            return self._calculate_sentiment_trend(df)
            
        except Exception as e:
            logger.error(f"Error analyzing news sentiment: {e}")
            return {'sentiment': 0, 'confidence': 0}
    
    def analyze_social_sentiment(self, symbol: str) -> Dict:
        """Analyze sentiment from social media."""
        try:
            social_data = self._fetch_social_data(symbol)
            if not social_data:
                return {'sentiment': 0, 'confidence': 0}
            
            sentiments = []
            for post in social_data.get('posts', []):
                sentiment = self._analyze_text(post['text'])
                sentiments.append(sentiment)
            
            df = pd.DataFrame(sentiments)
            return self._calculate_sentiment_trend(df)
            
        except Exception as e:
            logger.error(f"Error analyzing social sentiment: {e}")
            return {'sentiment': 0, 'confidence': 0}
    
    def analyze_earnings_sentiment(self, symbol: str) -> Dict:
        """Analyze sentiment from earnings call transcripts."""
        try:
            earnings_data = self._fetch_earnings_data(symbol)
            if not earnings_data:
                return {'sentiment': 0, 'confidence': 0}
            
            # Split into sections and analyze separately
            sections = self._split_earnings_call(earnings_data['transcript'])
            
            section_sentiments = {}
            for section, text in sections.items():
                sentiment = self._analyze_text(text)
                section_sentiments[section] = sentiment
            
            # Weight different sections
            weights = {
                'prepared_remarks': 0.4,
                'qa_session': 0.3,
                'guidance': 0.3
            }
            
            weighted_sentiment = sum(
                sentiment['compound'] * weights[section]
                for section, sentiment in section_sentiments.items()
            )
            
            return {
                'sentiment': weighted_sentiment,
                'confidence': 0.7,  # Earnings calls typically have high confidence
                'section_sentiments': section_sentiments
            }
            
        except Exception as e:
            logger.error(f"Error analyzing earnings sentiment: {e}")
            return {'sentiment': 0, 'confidence': 0}
    
    def _analyze_text(self, text: str) -> Dict:
        """Analyze sentiment of a piece of text using multiple methods."""
        try:
            # Clean and preprocess text
            text = self._preprocess_text(text)
            
            # Get VADER sentiment
            vader_scores = self.vader.polarity_scores(text)
            
            # Get TextBlob sentiment
            blob = TextBlob(text)
            textblob_sentiment = blob.sentiment.polarity
            
            # Get FinBERT sentiment (when available)
            finbert_sentiment = self._get_finbert_sentiment(text)
            
            # Get keyword-based sentiment
            keyword_sentiment = self._analyze_keywords(text)
            
            # Combine scores with weights
            weights = {
                'vader': 0.4,
                'textblob': 0.2,
                'finbert': 0.3,
                'keyword': 0.1
            }
            
            compound_score = (
                vader_scores['compound'] * weights['vader'] +
                textblob_sentiment * weights['textblob'] +
                finbert_sentiment['sentiment'] * weights['finbert'] +
                keyword_sentiment * weights['keyword']
            )
            
            return {
                'compound': compound_score,
                'vader': vader_scores,
                'textblob': textblob_sentiment,
                'finbert': finbert_sentiment['sentiment'],
                'keyword': keyword_sentiment,
                'confidence': finbert_sentiment['confidence']
            }
            
        except Exception as e:
            logger.error(f"Error in text analysis: {e}")
            return {'compound': 0, 'confidence': 0}
    
    def _preprocess_text(self, text: str) -> str:
        """Clean and preprocess text for sentiment analysis."""
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove special characters and digits
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def _get_finbert_sentiment(self, text: str) -> Dict:
        """Get sentiment using FinBERT model."""
        try:
            # Note: In production, load this once and cache
            tokenizer = AutoTokenizer.from_pretrained('ProsusAI/finbert')
            model = FinBERTSentiment()
            
            inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
            outputs = model(**inputs)
            probs = torch.softmax(outputs, dim=1)
            
            sentiment_score = (probs[0][0] - probs[0][2]).item()  # positive - negative
            confidence = torch.max(probs).item()
            
            return {'sentiment': sentiment_score, 'confidence': confidence}
            
        except Exception as e:
            logger.error(f"Error in FinBERT sentiment: {e}")
            return {'sentiment': 0, 'confidence': 0}
    
    def _analyze_keywords(self, text: str) -> float:
        """Analyze sentiment based on keyword presence."""
        words = text.lower().split()
        pos_count = sum(1 for word in words if word in self.pos_keywords)
        neg_count = sum(1 for word in words if word in self.neg_keywords)
        
        total = pos_count + neg_count
        if total == 0:
            return 0
            
        return (pos_count - neg_count) / total
    
    def _fetch_news(self, symbol: str, lookback_days: int) -> List[Dict]:
        """Fetch recent news articles."""
        try:
            news = self.gnews.get_news(f"{symbol} stock market")
            return news[:10]  # Return top 10 news items
        except Exception as e:
            logger.error(f"Error fetching news: {e}")
            return []
    
    def _fetch_social_data(self, symbol: str) -> Dict:
        """Fetch social media data (placeholder)."""
        # In production, implement API calls to social platforms
        return {
            'posts': [],
            'sentiment_score': 0,
            'volume': 0
        }
    
    def _fetch_earnings_data(self, symbol: str) -> Dict:
        """Fetch earnings call data (placeholder)."""
        return {}
    
    def _split_earnings_call(self, text: str) -> Dict:
        """Split earnings call transcript into sections."""
        # Implement splitting logic here
        return {
            'prepared_remarks': '',
            'qa_session': '',
            'guidance': ''
        }
    
    def _calculate_sentiment_trend(self, df: pd.DataFrame) -> Dict:
        """Calculate sentiment trend from a series of sentiment scores."""
        if df.empty:
            return {'sentiment': 0, 'confidence': 0}
        
        recent_weight = 2.0  # Give more weight to recent sentiment
        df['weight'] = np.linspace(1, recent_weight, len(df))
        
        weighted_sentiment = np.average(df['compound'], weights=df['weight'])
        confidence = np.mean(df['confidence'])
        
        return {
            'sentiment': weighted_sentiment,
            'confidence': confidence
        }
    
    def _calculate_weighted_sentiment(self, platform_sentiments: Dict) -> float:
        """Calculate weighted sentiment across different platforms."""
        weights = {
            'news': 0.4,
            'social': 0.3,
            'earnings': 0.3
        }
        
        weighted_sum = 0
        total_weight = 0
        
        for platform, sentiment in platform_sentiments.items():
            if platform in weights and sentiment['confidence'] > 0:
                weight = weights[platform] * sentiment['confidence']
                weighted_sum += sentiment['sentiment'] * weight
                total_weight += weight
        
        if total_weight == 0:
            return 0
            
        return weighted_sum / total_weight
    
    def get_sentiment_summary(self, symbol: str) -> Dict:
        """Get comprehensive sentiment analysis."""
        sentiments = {
            'news': self.analyze_news_sentiment(symbol),
            'social': self.analyze_social_sentiment(symbol),
            'earnings': self.analyze_earnings_sentiment(symbol)
        }
        
        overall_sentiment = self._calculate_weighted_sentiment(sentiments)
        
        return {
            'overall_sentiment': overall_sentiment,
            'detailed_sentiments': sentiments,
            'timestamp': datetime.now().isoformat(),
            'symbol': symbol
        }

class NewsTradingSignalGenerator:
    """Generates trading signals based on news sentiment."""
    
    def __init__(self):
        self.sentiment_analyzer = MarketSentimentAnalyzer()
        self.signal_history = []
        
    def generate_signal(self, symbol: str, current_price: float) -> Dict:
        """Generate trading signal based on sentiment analysis."""
        sentiment_data = self.sentiment_analyzer.get_sentiment_summary(symbol)
        
        # Extract sentiment scores
        overall_sentiment = sentiment_data['overall_sentiment']
        news_sentiment = sentiment_data['detailed_sentiments']['news']['sentiment']
        social_sentiment = sentiment_data['detailed_sentiments']['social']['sentiment']
        
        # Calculate signal strength (-1 to 1)
        signal_strength = np.tanh(overall_sentiment)
        
        # Define confidence thresholds
        high_confidence = 0.7
        medium_confidence = 0.5
        
        # Generate signal
        if signal_strength > high_confidence:
            signal = "STRONG_BUY"
            confidence = 0.9
        elif signal_strength > medium_confidence:
            signal = "BUY"
            confidence = 0.7
        elif signal_strength < -high_confidence:
            signal = "STRONG_SELL"
            confidence = 0.9
        elif signal_strength < -medium_confidence:
            signal = "SELL"
            confidence = 0.7
        else:
            signal = "HOLD"
            confidence = 0.5
        
        # Store signal
        signal_record = {
            'timestamp': datetime.now(),
            'symbol': symbol,
            'signal': signal,
            'strength': signal_strength,
            'confidence': confidence,
            'price': current_price,
            'sentiment_data': sentiment_data
        }
        
        self.signal_history.append(signal_record)
        
        return signal_record
    
    def get_signal_metrics(self) -> Dict:
        """Get metrics about signal generation performance."""
        if not self.signal_history:
            return {
                'total_signals': 0,
                'accuracy': 0,
                'avg_confidence': 0
            }
        
        total_signals = len(self.signal_history)
        avg_confidence = np.mean([s['confidence'] for s in self.signal_history])
        
        return {
            'total_signals': total_signals,
            'avg_confidence': avg_confidence,
            'recent_signals': self.signal_history[-5:]  # Last 5 signals
        }