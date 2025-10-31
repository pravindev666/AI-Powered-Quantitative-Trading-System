import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Union
from datetime import datetime, timedelta
import torch
import torch.nn as nn
from torch import nn  # Add explicit nn import
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import yfinance as yf
import requests
from bs4 import BeautifulSoup
import logging
import json
import re

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('averaged_perceptron_tagger')

logger = logging.getLogger(__name__)

class FinBERTSentiment(nn.Module):
    """FinBERT-based sentiment analysis model."""
    
    def __init__(self, model_name: str = 'ProsusAI/finbert'):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(768, 3)  # 3 classes: positive, negative, neutral
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        return self.classifier(pooled_output)

class MarketSentimentAnalyzer:
    """Advanced market sentiment analysis using multiple sources and methods."""
    
    def __init__(self, api_keys: Optional[Dict] = None):
        self.api_keys = api_keys or {}
        self.vader = SentimentIntensityAnalyzer()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # Initialize FinBERT model
        self.finbert = FinBERTSentiment()
        self.sentiment_history = []
        
        # Keyword dictionaries
        self.bullish_keywords = set(['bullish', 'buy', 'long', 'growth', 'uptrend', 'outperform'])
        self.bearish_keywords = set(['bearish', 'sell', 'short', 'decline', 'downtrend', 'underperform'])
        
    def analyze_news_sentiment(self, symbol: str, lookback_days: int = 7) -> Dict:
        """Analyze sentiment from recent news articles."""
        news_data = self._fetch_news(symbol, lookback_days)
        
        if not news_data:
            return None
            
        # Analyze each article
        sentiments = []
        for article in news_data:
            sentiment = self._analyze_text(article['title'] + ' ' + article.get('description', ''))
            sentiment['timestamp'] = article['datetime']
            sentiment['source'] = article.get('source', 'unknown')
            sentiments.append(sentiment)
            
        # Calculate aggregate metrics
        df = pd.DataFrame(sentiments)
        
        metrics = {
            'overall_sentiment': df['compound'].mean(),
            'sentiment_std': df['compound'].std(),
            'positive_ratio': len(df[df['compound'] > 0.2]) / len(df),
            'negative_ratio': len(df[df['compound'] < -0.2]) / len(df),
            'source_breakdown': df.groupby('source')['compound'].mean().to_dict(),
            'trend': self._calculate_sentiment_trend(df)
        }
        
        return {
            'metrics': metrics,
            'detailed_sentiments': sentiments
        }
        
    def analyze_social_sentiment(self, symbol: str) -> Dict:
        """Analyze sentiment from social media sources."""
        social_data = self._fetch_social_data(symbol)
        
        if not social_data:
            return None
            
        # Analyze sentiment by platform
        platform_sentiments = {}
        for platform, posts in social_data.items():
            sentiments = [self._analyze_text(post['text']) for post in posts]
            df = pd.DataFrame(sentiments)
            
            platform_sentiments[platform] = {
                'average_sentiment': df['compound'].mean(),
                'sentiment_std': df['compound'].std(),
                'volume': len(posts),
                'bullish_ratio': len(df[df['compound'] > 0.2]) / len(df)
            }
            
        return {
            'platform_sentiments': platform_sentiments,
            'overall_sentiment': np.mean([p['average_sentiment'] 
                                       for p in platform_sentiments.values()]),
            'weighted_sentiment': self._calculate_weighted_sentiment(platform_sentiments)
        }
        
    def analyze_earnings_sentiment(self, symbol: str) -> Dict:
        """Analyze sentiment from earnings calls and reports."""
        earnings_data = self._fetch_earnings_data(symbol)
        
        if not earnings_data:
            return None
            
        # Analyze earnings call transcripts
        call_sentiments = []
        for call in earnings_data.get('transcripts', []):
            sections = self._split_earnings_call(call['text'])
            section_sentiments = {}
            
            for section, text in sections.items():
                sentiment = self._analyze_text(text)
                sentiment['section'] = section
                section_sentiments[section] = sentiment
                
            call_sentiments.append({
                'date': call['date'],
                'sections': section_sentiments
            })
            
        return {
            'call_sentiments': call_sentiments,
            'guidance_sentiment': self._analyze_guidance(earnings_data.get('guidance', '')),
            'qa_sentiment': self._analyze_qa_session(earnings_data.get('qa_session', ''))
        }
        
    def _analyze_text(self, text: str) -> Dict:
        """Analyze text using multiple sentiment analysis methods."""
        # Clean text
        clean_text = self._preprocess_text(text)
        
        # VADER sentiment
        vader_scores = self.vader.polarity_scores(clean_text)
        
        # TextBlob sentiment
        blob = TextBlob(clean_text)
        textblob_sentiment = blob.sentiment
        
        # FinBERT sentiment
        finbert_sentiment = self._get_finbert_sentiment(clean_text)
        
        # Keyword analysis
        keyword_sentiment = self._analyze_keywords(clean_text)
        
        # Combine sentiments
        combined_sentiment = (vader_scores['compound'] + 
                            textblob_sentiment.polarity +
                            finbert_sentiment['score']) / 3
                            
        return {
            'compound': combined_sentiment,
            'vader': vader_scores,
            'textblob': {
                'polarity': textblob_sentiment.polarity,
                'subjectivity': textblob_sentiment.subjectivity
            },
            'finbert': finbert_sentiment,
            'keyword_sentiment': keyword_sentiment
        }
        
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for sentiment analysis."""
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenize
        words = word_tokenize(text)
        
        # Remove stopwords and lemmatize
        words = [self.lemmatizer.lemmatize(word) for word in words 
                if word not in self.stop_words]
        
        return ' '.join(words)
        
    def _get_finbert_sentiment(self, text: str) -> Dict:
        """Get sentiment using FinBERT."""
        inputs = self.finbert.tokenizer(text, return_tensors='pt', 
                                      truncation=True, max_length=512)
        
        with torch.no_grad():
            outputs = self.finbert(**inputs)
            probs = torch.softmax(outputs, dim=1)
            
        sentiment_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
        pred_sentiment = sentiment_map[torch.argmax(probs).item()]
        
        return {
            'label': pred_sentiment,
            'score': probs[0][2].item() - probs[0][0].item()  # positive - negative
        }
        
    def _analyze_keywords(self, text: str) -> float:
        """Analyze sentiment based on keyword presence."""
        words = set(word_tokenize(text))
        
        bullish_count = len(words.intersection(self.bullish_keywords))
        bearish_count = len(words.intersection(self.bearish_keywords))
        
        if bullish_count + bearish_count == 0:
            return 0
            
        return (bullish_count - bearish_count) / (bullish_count + bearish_count)
        
    def _fetch_news(self, symbol: str, lookback_days: int) -> List[Dict]:
        """Fetch news articles from multiple sources."""
        news = []
        
        # Fetch from various sources
        try:
            # Yahoo Finance
            ticker = yf.Ticker(symbol)
            news.extend(ticker.news)
            
            # Additional news sources could be added here
            
        except Exception as e:
            logger.error(f"Error fetching news: {str(e)}")
            
        return news
        
    def _fetch_social_data(self, symbol: str) -> Dict:
        """Fetch social media data."""
        social_data = {
            'twitter': [],
            'reddit': []
        }
        
        # Implementation would depend on available APIs
        # This is a placeholder that would need to be implemented
        # based on available data sources and API keys
        
        return social_data
        
    def _fetch_earnings_data(self, symbol: str) -> Dict:
        """Fetch earnings call transcripts and related data."""
        # Implementation would depend on available data sources
        # This is a placeholder that would need to be implemented
        return {}
        
    def _split_earnings_call(self, text: str) -> Dict:
        """Split earnings call transcript into sections."""
        sections = {
            'introduction': '',
            'prepared_remarks': '',
            'qa_session': ''
        }
        
        # Implementation would split the text into relevant sections
        return sections
        
    def _calculate_sentiment_trend(self, df: pd.DataFrame) -> Dict:
        """Calculate sentiment trend over time."""
        if len(df) < 2:
            return {'direction': 'neutral', 'strength': 0}
            
        # Calculate trend
        sentiment_series = df.sort_values('timestamp')['compound']
        trend = np.polyfit(range(len(sentiment_series)), sentiment_series, 1)[0]
        
        # Determine direction and strength
        direction = 'positive' if trend > 0 else 'negative' if trend < 0 else 'neutral'
        strength = abs(trend)
        
        return {
            'direction': direction,
            'strength': strength
        }
        
    def _calculate_weighted_sentiment(self, platform_sentiments: Dict) -> float:
        """Calculate weighted sentiment across platforms."""
        weights = {
            'twitter': 0.4,
            'reddit': 0.3,
            'stocktwits': 0.3
        }
        
        weighted_sum = 0
        total_weight = 0
        
        for platform, sentiment in platform_sentiments.items():
            if platform in weights:
                weight = weights[platform]
                weighted_sum += sentiment['average_sentiment'] * weight
                total_weight += weight
                
        return weighted_sum / total_weight if total_weight > 0 else 0
        
    def get_sentiment_summary(self, symbol: str) -> Dict:
        """Get comprehensive sentiment summary."""
        news_sentiment = self.analyze_news_sentiment(symbol)
        social_sentiment = self.analyze_social_sentiment(symbol)
        earnings_sentiment = self.analyze_earnings_sentiment(symbol)
        
        # Combine sentiments with weights
        weights = {
            'news': 0.4,
            'social': 0.3,
            'earnings': 0.3
        }
        
        sentiments = {
            'news': news_sentiment['metrics']['overall_sentiment'] if news_sentiment else 0,
            'social': social_sentiment['overall_sentiment'] if social_sentiment else 0,
            'earnings': np.mean([s['compound'] for s in earnings_sentiment['call_sentiments']])
            if earnings_sentiment and earnings_sentiment['call_sentiments'] else 0
        }
        
        weighted_sentiment = sum(sentiments[k] * weights[k] for k in weights)
        
        return {
            'overall_sentiment': weighted_sentiment,
            'detailed_sentiments': {
                'news': news_sentiment,
                'social': social_sentiment,
                'earnings': earnings_sentiment
            },
            'trend': self._calculate_sentiment_trend(
                pd.DataFrame(self.sentiment_history)
            ) if self.sentiment_history else None
        }
        
class NewsTradingSignalGenerator:
    """Generate trading signals based on news and sentiment analysis."""
    
    def __init__(self):
        self.sentiment_analyzer = MarketSentimentAnalyzer()
        self.signal_threshold = 0.3
        self.signal_history = []
        
    def generate_signal(self, symbol: str, current_price: float) -> Dict:
        """Generate trading signal based on news and sentiment."""
        # Get sentiment summary
        sentiment = self.sentiment_analyzer.get_sentiment_summary(symbol)
        
        # Calculate base signal
        base_signal = sentiment['overall_sentiment']
        
        # Adjust signal based on trend
        if sentiment['trend']:
            trend_adjustment = sentiment['trend']['strength'] * (
                1 if sentiment['trend']['direction'] == 'positive' else -1
            )
            base_signal += trend_adjustment * 0.2
            
        # Generate final signal
        signal = np.clip(base_signal, -1, 1)
        
        # Calculate confidence
        confidence = min(1.0, abs(signal) * 2)
        
        # Store signal
        self.signal_history.append({
            'timestamp': datetime.now(),
            'symbol': symbol,
            'signal': signal,
            'confidence': confidence,
            'sentiment': sentiment
        })
        
        return {
            'signal': signal if abs(signal) > self.signal_threshold else 0,
            'confidence': confidence,
            'sentiment': sentiment,
            'metadata': {
                'news_impact': sentiment['detailed_sentiments']['news']['metrics']['overall_sentiment']
                if sentiment['detailed_sentiments']['news'] else 0,
                'social_impact': sentiment['detailed_sentiments']['social']['overall_sentiment']
                if sentiment['detailed_sentiments']['social'] else 0
            }
        }
        
    def get_signal_metrics(self) -> Dict:
        """Get metrics about signal generation performance."""
        if not self.signal_history:
            return None
            
        signals = pd.DataFrame(self.signal_history)
        
        return {
            'avg_signal_strength': signals['signal'].abs().mean(),
            'avg_confidence': signals['confidence'].mean(),
            'signal_distribution': {
                'positive': (signals['signal'] > 0).mean(),
                'negative': (signals['signal'] < 0).mean(),
                'neutral': (signals['signal'] == 0).mean()
            },
            'confidence_by_direction': {
                'positive': signals[signals['signal'] > 0]['confidence'].mean(),
                'negative': signals[signals['signal'] < 0]['confidence'].mean()
            }
        }