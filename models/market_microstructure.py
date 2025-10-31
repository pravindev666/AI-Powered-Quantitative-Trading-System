"""Market microstructure analysis module"""
from dataclasses import dataclass
from collections import deque
from typing import Dict, Optional, List
import numpy as np

@dataclass
class OrderBookLevel:
    """A single price level in the order book"""
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
    """Level 2 order book implementation"""
    def __init__(self, max_levels: int = 10):
        self.bids = {}  # price -> OrderBookLevel
        self.asks = {}  # price -> OrderBookLevel
        self.max_levels = max_levels
        self.last_update = None
        self.updates_queue = deque(maxlen=1000)

    def update(self, book_update: OrderBookUpdate) -> None:
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
                if book[book_update.price].size <= 0:
                    del book[book_update.price]
        
        self._maintain_levels()

    def _maintain_levels(self) -> None:
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

    def get_mid_price(self) -> Optional[float]:
        """Calculate mid price"""
        if not self.bids or not self.asks:
            return None
        best_bid = max(self.bids.keys())
        best_ask = min(self.asks.keys())
        return (best_bid + best_ask) / 2
        
    def get_market_depth(self) -> Dict:
        """Get market depth statistics"""
        if not self.bids or not self.asks:
            return {'bid_depth': 0, 'ask_depth': 0, 'total_depth': 0}
            
        bid_depth = sum(level.size for level in self.bids.values())
        ask_depth = sum(level.size for level in self.asks.values())
        
        return {
            'bid_depth': bid_depth,
            'ask_depth': ask_depth,
            'total_depth': bid_depth + ask_depth,
            'depth_imbalance': (bid_depth - ask_depth) / (bid_depth + ask_depth)
        }