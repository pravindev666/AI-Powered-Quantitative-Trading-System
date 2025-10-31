import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Union, Tuple
from dataclasses import dataclass
from enum import Enum
import time
from concurrent.futures import ThreadPoolExecutor
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OrderType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"
    TWAP = "TWAP"
    VWAP = "VWAP"
    SMART = "SMART"

class OrderSide(Enum):
    BUY = "BUY"
    SELL = "SELL"

class VenueType(Enum):
    PRIMARY = "PRIMARY"
    DARK_POOL = "DARK_POOL"
    ATS = "ATS"
    SMART = "SMART"

@dataclass
class OrderRequest:
    symbol: str
    side: OrderSide
    quantity: float
    order_type: OrderType
    price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: str = "DAY"
    max_slippage_bps: float = 10.0
    min_execution_size: Optional[float] = None
    smart_routing: bool = True
    venue_preferences: Optional[List[VenueType]] = None

@dataclass
class MarketData:
    symbol: str
    bid: float
    ask: float
    last: float
    volume: float
    timestamp: float
    bid_size: float
    ask_size: float
    venues: Dict[str, Dict[str, float]]

class SmartOrderRouter:
    """Smart Order Router (SOR) for optimal venue selection and execution."""
    
    def __init__(self, execution_config: Optional[Dict] = None):
        self.config = execution_config or {
            'max_venue_slippage': 20,  # Max acceptable slippage in bps
            'min_fill_rate': 0.8,      # Minimum acceptable fill rate
            'dark_pool_threshold': 0.1, # Volume threshold for dark pool routing
            'venue_rankings': {         # Historical venue performance rankings
                VenueType.PRIMARY: 1.0,
                VenueType.DARK_POOL: 0.8,
                VenueType.ATS: 0.9,
                VenueType.SMART: 0.95
            }
        }
        self.venue_stats = {}  # Historical venue statistics
        
    def select_optimal_venues(self, order: OrderRequest, market_data: MarketData) -> List[Tuple[VenueType, float]]:
        """Select optimal venues for order execution based on market data and historical performance."""
        venues = []
        total_quantity = order.quantity
        remaining_qty = total_quantity
        
        # Calculate venue scores based on multiple factors
        venue_scores = {}
        for venue_type in VenueType:
            if venue_type in market_data.venues:
                venue_data = market_data.venues[venue_type.value]
                
                # Calculate venue score based on:
                # 1. Price improvement
                # 2. Available liquidity
                # 3. Historical fill rate
                # 4. Historical slippage
                # 5. Venue ranking
                
                price_improvement = self._calculate_price_improvement(order, venue_data)
                liquidity_score = min(1.0, venue_data.get('size', 0) / total_quantity)
                fill_rate = self.venue_stats.get((venue_type, order.symbol), {}).get('fill_rate', 0.9)
                hist_slippage = self.venue_stats.get((venue_type, order.symbol), {}).get('avg_slippage', 5)
                
                venue_scores[venue_type] = (
                    price_improvement * 0.3 +
                    liquidity_score * 0.2 +
                    fill_rate * 0.2 +
                    (1 - hist_slippage/self.config['max_venue_slippage']) * 0.15 +
                    self.config['venue_rankings'][venue_type] * 0.15
                )
        
        # Sort venues by score and allocate quantity
        sorted_venues = sorted(venue_scores.items(), key=lambda x: x[1], reverse=True)
        
        for venue_type, score in sorted_venues:
            venue_data = market_data.venues[venue_type.value]
            available_qty = venue_data.get('size', 0)
            
            if available_qty > order.min_execution_size or not order.min_execution_size:
                alloc_qty = min(remaining_qty, available_qty)
                if alloc_qty > 0:
                    venues.append((venue_type, alloc_qty))
                    remaining_qty -= alloc_qty
                    
            if remaining_qty <= 0:
                break
                
        return venues
    
    def _calculate_price_improvement(self, order: OrderRequest, venue_data: Dict) -> float:
        """Calculate potential price improvement at a venue."""
        if order.side == OrderSide.BUY:
            best_price = venue_data.get('ask', float('inf'))
            benchmark = venue_data.get('last', best_price)
        else:
            best_price = venue_data.get('bid', 0)
            benchmark = venue_data.get('last', best_price)
            
        price_improvement = abs(best_price - benchmark) / benchmark
        return min(1.0, price_improvement * 100)  # Convert to percentage and cap at 100%

class ExecutionOptimizer:
    """Optimizes order execution using various algorithms and strategies."""
    
    def __init__(self):
        self.router = SmartOrderRouter()
        self.market_impact_model = self._initialize_market_impact_model()
        self.execution_stats = {}
        
    def _initialize_market_impact_model(self):
        """Initialize the market impact model with historical data."""
        # Simplified market impact model
        # In production, this would be a more sophisticated ML model
        return lambda size, adf: 0.1 * np.sqrt(size / adf)
    
    def optimize_execution(self, order: OrderRequest, market_data: MarketData) -> Dict:
        """Optimize order execution strategy."""
        if order.order_type == OrderType.SMART:
            return self._optimize_smart_order(order, market_data)
        elif order.order_type in [OrderType.TWAP, OrderType.VWAP]:
            return self._optimize_algo_order(order, market_data)
        else:
            return self._optimize_direct_order(order, market_data)
    
    def _optimize_smart_order(self, order: OrderRequest, market_data: MarketData) -> Dict:
        """Optimize smart order execution using adaptive strategies."""
        # Calculate market impact
        avg_daily_volume = market_data.volume
        market_impact = self.market_impact_model(order.quantity, avg_daily_volume)
        
        # Select optimal venues
        venues = self.router.select_optimal_venues(order, market_data)
        
        # Calculate optimal execution schedule
        schedule = self._calculate_execution_schedule(order, market_data, market_impact)
        
        return {
            'venues': venues,
            'schedule': schedule,
            'estimated_impact': market_impact,
            'strategy': 'ADAPTIVE'
        }
    
    def _optimize_algo_order(self, order: OrderRequest, market_data: MarketData) -> Dict:
        """Optimize algorithmic order execution (TWAP/VWAP)."""
        if order.order_type == OrderType.TWAP:
            schedule = self._calculate_twap_schedule(order, market_data)
        else:  # VWAP
            schedule = self._calculate_vwap_schedule(order, market_data)
            
        return {
            'schedule': schedule,
            'type': order.order_type.value,
            'estimated_impact': self._estimate_algo_impact(order, market_data)
        }
    
    def _optimize_direct_order(self, order: OrderRequest, market_data: MarketData) -> Dict:
        """Optimize direct order execution."""
        # Calculate optimal price and size
        optimal_price = self._calculate_optimal_price(order, market_data)
        
        # Check if order should be split
        should_split = order.quantity > market_data.volume * 0.1
        
        if should_split:
            child_orders = self._split_order(order, market_data)
            return {
                'type': 'SPLIT',
                'child_orders': child_orders,
                'optimal_price': optimal_price
            }
        else:
            return {
                'type': 'DIRECT',
                'price': optimal_price,
                'venue': self._select_best_venue(order, market_data)
            }
    
    def _calculate_execution_schedule(self, order: OrderRequest, 
                                   market_data: MarketData,
                                   market_impact: float) -> List[Dict]:
        """Calculate optimal execution schedule based on market conditions."""
        total_qty = order.quantity
        remaining_qty = total_qty
        schedule = []
        
        # Calculate participation rate based on market impact
        base_participation_rate = min(0.3, market_impact * 2)
        
        # Adjust for time of day and volatility
        current_hour = time.localtime().tm_hour
        if 9 <= current_hour < 10 or 15 <= current_hour < 16:
            # Higher participation during market open/close
            participation_rate = base_participation_rate * 1.5
        else:
            participation_rate = base_participation_rate
            
        # Generate schedule
        interval_minutes = 5
        num_intervals = max(1, int(remaining_qty / (market_data.volume * participation_rate * interval_minutes/390)))
        
        for i in range(num_intervals):
            interval_qty = remaining_qty / (num_intervals - i)
            schedule.append({
                'time': i * interval_minutes,
                'quantity': interval_qty,
                'participation_rate': participation_rate
            })
            remaining_qty -= interval_qty
            
        return schedule
    
    def _calculate_twap_schedule(self, order: OrderRequest, market_data: MarketData) -> List[Dict]:
        """Calculate Time-Weighted Average Price execution schedule."""
        total_qty = order.quantity
        num_intervals = 6  # 5-minute intervals
        qty_per_interval = total_qty / num_intervals
        
        return [{
            'time': i * 5,
            'quantity': qty_per_interval
        } for i in range(num_intervals)]
    
    def _calculate_vwap_schedule(self, order: OrderRequest, market_data: MarketData) -> List[Dict]:
        """Calculate Volume-Weighted Average Price execution schedule."""
        # Simplified volume profile
        volume_profile = {
            0: 0.12,  # 9:30-10:30
            1: 0.08,  # 10:30-11:30
            2: 0.06,  # 11:30-12:30
            3: 0.06,  # 12:30-1:30
            4: 0.08,  # 1:30-2:30
            5: 0.12   # 2:30-3:30
        }
        
        schedule = []
        total_qty = order.quantity
        
        for hour, vol_fraction in volume_profile.items():
            qty = total_qty * vol_fraction
            schedule.append({
                'time': hour * 60,
                'quantity': qty,
                'expected_volume_fraction': vol_fraction
            })
            
        return schedule
    
    def _calculate_optimal_price(self, order: OrderRequest, market_data: MarketData) -> float:
        """Calculate optimal limit price based on order characteristics and market data."""
        spread = market_data.ask - market_data.bid
        mid_price = (market_data.ask + market_data.bid) / 2
        
        if order.side == OrderSide.BUY:
            optimal_price = mid_price + spread * 0.3  # Aggressive for buys
        else:
            optimal_price = mid_price - spread * 0.3  # Aggressive for sells
            
        return optimal_price
    
    def _split_order(self, order: OrderRequest, market_data: MarketData) -> List[Dict]:
        """Split large orders into child orders."""
        total_qty = order.quantity
        avg_trade_size = market_data.volume / 100  # Estimate
        
        num_children = min(10, int(total_qty / avg_trade_size))
        child_orders = []
        
        for i in range(num_children):
            child_qty = total_qty / num_children
            child_orders.append({
                'quantity': child_qty,
                'delay': i * 2  # 2-minute spacing
            })
            
        return child_orders
    
    def _select_best_venue(self, order: OrderRequest, market_data: MarketData) -> VenueType:
        """Select the best venue for direct order execution."""
        venues = self.router.select_optimal_venues(order, market_data)
        return venues[0][0] if venues else VenueType.PRIMARY
    
    def _estimate_algo_impact(self, order: OrderRequest, market_data: MarketData) -> float:
        """Estimate market impact for algorithmic orders."""
        # Simplified impact estimate
        participation_rate = order.quantity / market_data.volume
        return 0.1 * np.sqrt(participation_rate)  # Simplified square-root model

class ExecutionAnalytics:
    """Analyzes execution performance and provides feedback for optimization."""
    
    def __init__(self):
        self.execution_history = []
        
    def add_execution(self, order: OrderRequest, execution_details: Dict):
        """Add execution details to history."""
        self.execution_history.append({
            'timestamp': time.time(),
            'order': order,
            'execution': execution_details
        })
        
    def calculate_metrics(self, window_minutes: int = 60) -> Dict:
        """Calculate execution quality metrics."""
        recent_executions = [
            ex for ex in self.execution_history
            if time.time() - ex['timestamp'] < window_minutes * 60
        ]
        
        if not recent_executions:
            return {}
            
        metrics = {
            'fill_rate': self._calculate_fill_rate(recent_executions),
            'slippage': self._calculate_slippage(recent_executions),
            'venue_analysis': self._analyze_venues(recent_executions),
            'timing_analysis': self._analyze_timing(recent_executions)
        }
        
        return metrics
    
    def _calculate_fill_rate(self, executions: List[Dict]) -> float:
        """Calculate average fill rate."""
        fill_rates = []
        for ex in executions:
            ordered = ex['order'].quantity
            filled = sum(child.get('filled_quantity', 0) 
                        for child in ex['execution'].get('child_orders', []))
            fill_rates.append(filled / ordered if ordered > 0 else 0)
        
        return np.mean(fill_rates) if fill_rates else 0
    
    def _calculate_slippage(self, executions: List[Dict]) -> float:
        """Calculate average price slippage."""
        slippages = []
        for ex in executions:
            if 'expected_price' in ex['execution'] and 'achieved_price' in ex['execution']:
                expected = ex['execution']['expected_price']
                achieved = ex['execution']['achieved_price']
                slippages.append((achieved - expected) / expected * 10000)  # in bps
                
        return np.mean(slippages) if slippages else 0
    
    def _analyze_venues(self, executions: List[Dict]) -> Dict:
        """Analyze venue performance."""
        venue_stats = {}
        for ex in executions:
            venue = ex['execution'].get('venue')
            if venue:
                if venue not in venue_stats:
                    venue_stats[venue] = {'count': 0, 'slippage': []}
                venue_stats[venue]['count'] += 1
                if 'slippage' in ex['execution']:
                    venue_stats[venue]['slippage'].append(ex['execution']['slippage'])
                    
        # Calculate average slippage per venue
        for venue in venue_stats:
            slippages = venue_stats[venue]['slippage']
            venue_stats[venue]['avg_slippage'] = np.mean(slippages) if slippages else 0
            
        return venue_stats
    
    def _analyze_timing(self, executions: List[Dict]) -> Dict:
        """Analyze execution timing patterns."""
        timing_stats = {
            'morning': {'count': 0, 'slippage': []},
            'midday': {'count': 0, 'slippage': []},
            'afternoon': {'count': 0, 'slippage': []}
        }
        
        for ex in executions:
            hour = time.localtime(ex['timestamp']).tm_hour
            period = (
                'morning' if hour < 11 else
                'afternoon' if hour >= 14 else
                'midday'
            )
            
            timing_stats[period]['count'] += 1
            if 'slippage' in ex['execution']:
                timing_stats[period]['slippage'].append(ex['execution']['slippage'])
                
        # Calculate average slippage per period
        for period in timing_stats:
            slippages = timing_stats[period]['slippage']
            timing_stats[period]['avg_slippage'] = np.mean(slippages) if slippages else 0
            
        return timing_stats