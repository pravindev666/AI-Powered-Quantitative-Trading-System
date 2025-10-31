"""
Update script for Market Microstructure Analysis enhancement
"""

from pathlib import Path
import shutil
import os

def backup_existing_file():
    """Create backup of existing market_microstructure.py"""
    src = Path('market_microstructure.py')
    if not src.exists():
        return False
        
    backup = Path('market_microstructure.py.bak')
    shutil.copy2(src, backup)
    return True

def update_microstructure_module():
    """Apply Market Microstructure Analysis enhancement"""
    enhanced_content = '''"""
Market Microstructure Analysis Module for Enhanced Price Discovery
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from collections import deque
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

@dataclass
class OrderBookLevel:
    """Represents a single price level in the order book"""
    price: float
    size: float
    order_count: int

@dataclass
class OrderBookUpdate:
    """Order book update event"""
    timestamp: float
    book_type: str  # 'bid' or 'ask'
    action: str    # 'add', 'modify', 'delete'
    price: float
    size: float
    order_id: Optional[str] = None

class OrderBook:
    """Level 2 order book implementation with advanced analytics"""
    
    def __init__(self, max_levels: int = 10):
        self.bids = {}  # price -> OrderBookLevel
        self.asks = {}  # price -> OrderBookLevel
        self.max_levels = max_levels
        self.last_update = None
        self.updates_queue = deque(maxlen=1000)
    
    def update(self, book_update: OrderBookUpdate):
        """Process an order book update"""
        self.last_update = book_update.timestamp
        self.updates_queue.append(book_update)
        
        book = self.bids if book_update.book_type == 'bid' else self.asks
        
        if book_update.action == 'add':
            if book_update.price not in book:
                book[book_update.price] = OrderBookLevel(
                    price=book_update.price,
                    size=book_update.size,
                    order_count=1
                )
            else:
                book[book_update.price].size += book_update.size
                book[book_update.price].order_count += 1
                
        elif book_update.action == 'modify':
            if book_update.price in book:
                book[book_update.price].size = book_update.size
                
        elif book_update.action == 'delete':
            if book_update.price in book:
                book[book_update.price].size -= book_update.size
                book[book_update.price].order_count -= 1
                if book[book_update.price].size <= 0 or book[book_update.price].order_count <= 0:
                    del book[book_update.price]
        
        self._maintain_levels()
    
    def _maintain_levels(self):
        """Maintain maximum number of price levels"""
        if len(self.bids) > self.max_levels:
            sorted_bids = sorted(self.bids.items(), reverse=True)
            self.bids = dict(sorted_bids[:self.max_levels])
        
        if len(self.asks) > self.max_levels:
            sorted_asks = sorted(self.asks.items())
            self.asks = dict(sorted_asks[:self.max_levels])
    
    def get_spread(self) -> float:
        """Calculate current bid-ask spread"""
        if not self.bids or not self.asks:
            return float('inf')
        
        best_bid = max(self.bids.keys())
        best_ask = min(self.asks.keys())
        return best_ask - best_bid
    
    def get_mid_price(self) -> float:
        """Calculate mid price"""
        if not self.bids or not self.asks:
            return None
        
        best_bid = max(self.bids.keys())
        best_ask = min(self.asks.keys())
        return (best_ask + best_bid) / 2
    
    def get_market_depth(self, levels: int = 5) -> Dict:
        """Get market depth statistics"""
        bid_volume = 0
        ask_volume = 0
        
        sorted_bids = sorted(self.bids.items(), reverse=True)[:levels]
        sorted_asks = sorted(self.asks.items())[:levels]
        
        for price, level in sorted_bids:
            bid_volume += level.size
            
        for price, level in sorted_asks:
            ask_volume += level.size
            
        return {
            'bid_volume': bid_volume,
            'ask_volume': ask_volume,
            'total_volume': bid_volume + ask_volume,
            'bid_ask_ratio': bid_volume / ask_volume if ask_volume > 0 else float('inf'),
            'bid_levels': len(sorted_bids),
            'ask_levels': len(sorted_asks)
        }
        
    def calculate_vwap(self, quantity: float, side: str) -> Optional[float]:
        """Calculate VWAP for a given quantity"""
        if not self.bids or not self.asks:
            return None
            
        book = self.bids if side.lower() == 'sell' else self.asks
        sorted_levels = sorted(book.items(), reverse=(side.lower() == 'sell'))
        
        total_quantity = 0
        total_price = 0
        
        for price, level in sorted_levels:
            available = min(quantity - total_quantity, level.size)
            if available <= 0:
                break
                
            total_quantity += available
            total_price += available * price
            
        if total_quantity > 0:
            return total_price / total_quantity
        return None

class MarketMicrostructureAnalyzer:
    """Advanced market microstructure analysis"""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.order_book = OrderBook()
        self.metrics_history = deque(maxlen=window_size)
        self.trade_history = deque(maxlen=window_size)
    
    def process_order_book_update(self, update: OrderBookUpdate) -> Dict:
        """Process order book update and calculate metrics"""
        self.order_book.update(update)
        
        metrics = self._calculate_microstructure_metrics()
        self.metrics_history.append(metrics)
        
        return metrics
    
    def process_trade(self, trade: Dict):
        """Process a trade and update metrics"""
        self.trade_history.append(trade)
    
    def _calculate_microstructure_metrics(self) -> Dict:
        """Calculate comprehensive microstructure metrics"""
        spread = self.order_book.get_spread()
        mid_price = self.order_book.get_mid_price()
        depth = self.order_book.get_market_depth()
        
        metrics = {
            'spread': spread,
            'relative_spread': spread / mid_price if mid_price else None,
            'mid_price': mid_price,
            'depth': depth,
            'order_flow_imbalance': self._calculate_order_flow_imbalance(),
            'tick_size': self._estimate_tick_size(),
            'market_resiliency': self._calculate_market_resiliency(),
            'price_impact': self._estimate_price_impact(),
            'liquidity_score': self._calculate_liquidity_score()
        }
        
        return metrics
    
    def _calculate_order_flow_imbalance(self) -> float:
        """Calculate order flow imbalance"""
        if not self.order_book.updates_queue:
            return 0.0
        
        recent_updates = list(self.order_book.updates_queue)[-100:]
        
        buy_volume = sum(update.size for update in recent_updates 
                        if update.book_type == 'bid' and update.action == 'add')
        sell_volume = sum(update.size for update in recent_updates 
                         if update.book_type == 'ask' and update.action == 'add')
        
        total_volume = buy_volume + sell_volume
        return (buy_volume - sell_volume) / total_volume if total_volume > 0 else 0
    
    def _estimate_tick_size(self) -> float:
        """Estimate effective tick size"""
        all_prices = (list(self.order_book.bids.keys()) + 
                     list(self.order_book.asks.keys()))
        
        if len(all_prices) < 2:
            return None
        
        price_diffs = np.diff(sorted(all_prices))
        return np.min(price_diffs[price_diffs > 0]) if len(price_diffs) > 0 else None
    
    def _calculate_market_resiliency(self) -> float:
        """Calculate market resiliency metric"""
        if len(self.metrics_history) < 2:
            return None
        
        spreads = [m['spread'] for m in self.metrics_history]
        return np.mean([s2/s1 for s1, s2 in zip(spreads[:-1], spreads[1:])])
    
    def _estimate_price_impact(self) -> Dict:
        """Estimate price impact coefficients"""
        if not self.trade_history:
            return {'temporary': None, 'permanent': None}
        
        # Calculate temporary impact
        trade_prices = [t.get('price', 0) for t in self.trade_history]
        mid_prices = [t.get('mid_price', 0) for t in self.trade_history]
        volumes = [t.get('volume', 0) for t in self.trade_history]
        
        if not all(trade_prices) or not all(mid_prices) or not all(volumes):
            return {'temporary': None, 'permanent': None}
        
        temp_impacts = [abs(p - m)/m for p, m in zip(trade_prices, mid_prices)]
        temp_coef = np.polyfit(volumes, temp_impacts, 1)[0] if len(volumes) > 1 else None
        
        # Calculate permanent impact
        perm_impacts = []
        for i in range(len(trade_prices)-1):
            if mid_prices[i] and mid_prices[i+1]:
                perm_impacts.append(abs(mid_prices[i+1] - mid_prices[i])/mid_prices[i])
        
        perm_coef = np.mean(perm_impacts) if perm_impacts else None
        
        return {
            'temporary': temp_coef,
            'permanent': perm_coef
        }
    
    def _calculate_liquidity_score(self) -> float:
        """Calculate composite liquidity score"""
        metrics = self._calculate_microstructure_metrics()
        
        if not metrics['depth'] or not metrics['spread']:
            return None
        
        depth_score = min(1.0, metrics['depth']['total_volume'] / 100000)  # Normalize
        spread_score = min(1.0, 1 / (1 + metrics['relative_spread']))
        flow_score = (metrics['order_flow_imbalance'] + 1) / 2  # Convert to [0,1]
        
        # Weighted average
        return 0.4 * depth_score + 0.4 * spread_score + 0.2 * flow_score

class VolumeProfileAnalyzer:
    """Volume profile and order dynamics analysis"""
    
    def __init__(self, num_bins: int = 50):
        self.num_bins = num_bins
        self.volume_profile = {}
        self.trade_sizes = deque(maxlen=1000)
        self.price_levels = deque(maxlen=1000)
    
    def update_volume_profile(self, price: float, volume: float, timestamp: float):
        """Update volume profile with new trade"""
        price_bin = round(price / self.num_bins) * self.num_bins
        
        if price_bin not in self.volume_profile:
            self.volume_profile[price_bin] = {
                'volume': 0,
                'trade_count': 0,
                'last_update': None
            }
        
        self.volume_profile[price_bin]['volume'] += volume
        self.volume_profile[price_bin]['trade_count'] += 1
        self.volume_profile[price_bin]['last_update'] = timestamp
        
        self.trade_sizes.append(volume)
        self.price_levels.append(price_bin)
    
    def get_volume_profile_metrics(self) -> Dict:
        """Get volume profile analytics"""
        if not self.volume_profile:
            return None
        
        total_volume = sum(level['volume'] for level in self.volume_profile.values())
        
        # Find POC (Point of Control)
        poc_level = max(self.volume_profile.items(), 
                       key=lambda x: x[1]['volume'])
        
        # Calculate Value Area
        value_area_volume = 0
        value_area_levels = []
        target_volume = total_volume * 0.68  # 68% of volume
        
        sorted_levels = sorted(self.volume_profile.items(), 
                             key=lambda x: x[1]['volume'],
                             reverse=True)
        
        for price, data in sorted_levels:
            if value_area_volume < target_volume:
                value_area_levels.append(price)
                value_area_volume += data['volume']
        
        return {
            'total_volume': total_volume,
            'poc_price': poc_level[0],
            'poc_volume': poc_level[1]['volume'],
            'value_area': {
                'low': min(value_area_levels),
                'high': max(value_area_levels)
            },
            'volume_distribution': {
                'mean': np.mean(list(self.trade_sizes)),
                'std': np.std(list(self.trade_sizes)),
                'skew': pd.Series(list(self.trade_sizes)).skew()
            }
        }
    
    def analyze_trade_size_distribution(self) -> Dict:
        """Analyze trade size patterns"""
        if not self.trade_sizes:
            return None
        
        sizes = np.array(list(self.trade_sizes))
        
        return {
            'mean_size': float(np.mean(sizes)),
            'median_size': float(np.median(sizes)),
            'size_std': float(np.std(sizes)),
            'large_trade_ratio': float(np.sum(sizes > np.percentile(sizes, 90)) / len(sizes))
        }
    
    def calculate_volume_momentum(self) -> float:
        """Calculate volume momentum indicator"""
        if len(self.trade_sizes) < 20:
            return 0.0
        
        recent_vol = np.mean(list(self.trade_sizes)[-10:])
        older_vol = np.mean(list(self.trade_sizes)[-20:-10])
        
        return (recent_vol - older_vol) / older_vol if older_vol > 0 else 0.0
'''
    
    # Write enhanced content
    with open('market_microstructure.py', 'w') as f:
        f.write(enhanced_content)

def main():
    """Apply Market Microstructure Analysis enhancement"""
    print("Applying Market Microstructure Analysis enhancement (Patch 19)...")
    
    # Backup existing file
    if backup_existing_file():
        print("✅ Created backup of existing market_microstructure.py")
    
    # Update module
    update_microstructure_module()
    print("✅ Updated market_microstructure.py with enhanced functionality")
    
    print("\nEnhancements added:")
    print("1. Advanced order book analytics")
    print("2. Market microstructure metrics")
    print("3. Volume profile analysis")
    print("4. Trade size distribution analytics")
    print("5. Price impact estimation")
    print("6. Market resiliency calculation")
    print("7. Composite liquidity scoring")

if __name__ == "__main__":
    main()