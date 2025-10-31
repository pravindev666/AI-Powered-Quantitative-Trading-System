"""
NSE Option Chain Fetcher
Fetches real option chain data from NSE for Nifty
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime
import json
import time

class NSEOptionChainFetcher:
    """Fetch real option chain from NSE"""
    
    def __init__(self):
        self.base_url = "https://www.nseindia.com/api/option-chain-indices"
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "application/json",
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://www.nseindia.com/option-chain"
        }
        self.session = None
    
    def _init_session(self):
        """Initialize session with cookies"""
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        # Get cookies
        self.session.get("https://www.nseindia.com", timeout=10)
    
    def fetch_nifty_option_chain(self, symbol="NIFTY"):
        """Fetch live NIFTY option chain"""
        if not self.session:
            self._init_session()
        
        try:
            url = f"{self.base_url}?symbol={symbol}"
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            records = data['records']['data']
            
            # Extract call/put data
            strikes = []
            call_prices = []
            put_prices = []
            call_ivs = []
            put_ivs = []
            
            for record in records:
                strike = record['strikePrice']
                
                # Call data
                if 'CE' in record:
                    ce = record['CE']
                    call_bid = ce.get('bidprice', 0)
                    call_ask = ce.get('askprice', 0)
                    call_mid = (call_bid + call_ask) / 2 if (call_bid and call_ask) else ce.get('lastPrice', 0)
                    call_iv = ce.get('impliedVolatility', 0)
                    
                    if call_mid > 0:
                        strikes.append(strike)
                        call_prices.append(call_mid)
                        call_ivs.append(call_iv)
                
                # Put data
                if 'PE' in record:
                    pe = record['PE']
                    put_bid = pe.get('bidprice', 0)
                    put_ask = pe.get('askprice', 0)
                    put_mid = (put_bid + put_ask) / 2 if (put_bid and put_ask) else pe.get('lastPrice', 0)
                    put_iv = pe.get('impliedVolatility', 0)
                    
                    if put_mid > 0:
                        put_prices.append(put_mid)
                        put_ivs.append(put_iv)
            
            spot_price = data['records']['underlyingValue']
            
            return {
                'spot': spot_price,
                'strikes': np.array(strikes),
                'call_prices': np.array(call_prices),
                'put_prices': np.array(put_prices),
                'call_ivs': np.array(call_ivs),
                'put_ivs': np.array(put_ivs),
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            print(f"   ⚠️ NSE option chain fetch failed: {e}")
            return None