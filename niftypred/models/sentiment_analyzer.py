"""
Financial News Sentiment Analysis using FinBERT
"""

from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
import numpy as np
import pandas as pd
from datetime import datetime
from gnews import GNews
import json

class NewsSentimentAnalyzer:
    """Fetch and analyze financial news sentiment using FinBERT"""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.sentiment_pipeline = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Load FinBERT model"""
        try:
            model_name = "ProsusAI/finbert"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model=self.model,
                tokenizer=self.tokenizer,
                device=-1  # CPU
            )
            
            print("   ✅ FinBERT model loaded")
            
        except Exception as e:
            print(f"   ⚠️ FinBERT model loading failed: {e}")
            self.sentiment_pipeline = None
    
    def fetch_news_gnews(self, query="NIFTY India", max_results=20):
        """Fetch news using GNews"""
        try:
            google_news = GNews(
                language='en',
                country='IN',
                period='1d',
                max_results=max_results
            )
            
            news_items = google_news.get_news(query)
            
            articles = []
            for item in news_items:
                articles.append({
                    'title': item.get('title', ''),
                    'description': item.get('description', ''),
                    'published': item.get('published date', ''),
                    'url': item.get('url', '')
                })
            
            return articles
            
        except Exception as e:
            print(f"   ⚠️ GNews fetch failed: {e}")
            return []
    
    def analyze_sentiment(self, text):
        """Analyze text sentiment using FinBERT"""
        if not self.sentiment_pipeline:
            return {'label': 'neutral', 'score': 0.5}
        
        try:
            text = text[:512]  # Truncate to max length
            result = self.sentiment_pipeline(text)[0]
            
            label_map = {
                'positive': 1.0,
                'negative': -1.0,
                'neutral': 0.0
            }
            
            label = result['label'].lower()
            confidence = result['score']
            sentiment_score = label_map.get(label, 0.0) * confidence
            
            return {
                'label': label,
                'score': confidence,
                'sentiment_score': sentiment_score
            }
            
        except Exception as e:
            print(f"   ⚠️ Sentiment analysis failed: {e}")
            return {'label': 'neutral', 'score': 0.5, 'sentiment_score': 0.0}
    
    def get_market_sentiment_score(self):
        """Get aggregate market sentiment"""
        print("\n📰 Fetching and analyzing news sentiment...")
        
        articles = self.fetch_news_gnews()
        
        if not articles:
            print("   ⚠️ No news articles found")
            return {
                'sentiment_score': 0.0,
                'positive_count': 0,
                'negative_count': 0,
                'neutral_count': 0,
                'articles_analyzed': 0
            }
        
        print(f"   Analyzing {len(articles)} articles...")
        
        sentiments = []
        positive_count = 0
        negative_count = 0
        neutral_count = 0
        
        for article in articles:
            text = f"{article['title']}. {article['description']}"
            sentiment = self.analyze_sentiment(text)
            sentiments.append(sentiment['sentiment_score'])
            
            if sentiment['label'] == 'positive':
                positive_count += 1
            elif sentiment['label'] == 'negative':
                negative_count += 1
            else:
                neutral_count += 1
        
        avg_sentiment = np.mean(sentiments) if sentiments else 0.0
        
        # Calculate momentum
        if len(sentiments) > 5:
            recent_sentiment = np.mean(sentiments[:5])
            sentiment_momentum = recent_sentiment - avg_sentiment
        else:
            sentiment_momentum = 0.0
        
        print(f"   ✅ Sentiment Score: {avg_sentiment:+.3f}")
        print(f"      Positive: {positive_count}, Negative: {negative_count}, Neutral: {neutral_count}")
        
        return {
            'sentiment_score': float(avg_sentiment),
            'sentiment_momentum': float(sentiment_momentum),
            'positive_count': positive_count,
            'negative_count': negative_count,
            'neutral_count': neutral_count,
            'articles_analyzed': len(articles),
            'timestamp': datetime.now()
        }
    
    def cache_daily_sentiment(self, cache_file='sentiment_cache.json'):
        """Cache daily sentiment scores"""
        try:
            # Load existing cache
            try:
                with open(cache_file, 'r') as f:
                    cache = json.load(f)
            except FileNotFoundError:
                cache = {}
            
            # Get today's sentiment
            today = datetime.now().strftime('%Y-%m-%d')
            
            if today not in cache:
                sentiment_data = self.get_market_sentiment_score()
                cache[today] = sentiment_data
                
                # Save cache
                with open(cache_file, 'w') as f:
                    json.dump(cache, f, indent=2, default=str)
                
                print(f"   ✅ Cached sentiment for {today}")
            else:
                print(f"   ℹ️ Sentiment already cached for {today}")
            
            return cache
            
        except Exception as e:
            print(f"   ⚠️ Sentiment caching failed: {e}")
            return {}