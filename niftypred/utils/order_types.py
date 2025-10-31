"""
Order Types and Trading Infrastructure
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict
from datetime import datetime

class OrderSide(Enum):
    """Order side enumeration"""
    BUY = "BUY"
    SELL = "SELL"

class OrderType(Enum):
    """Order type enumeration"""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"

@dataclass
class OrderRequest:
    """Order request data class"""
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: str = "DAY"
    order_id: Optional[str] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        """Validate order parameters after initialization"""
        if self.timestamp is None:
            self.timestamp = datetime.now()
            
        if self.order_type in [OrderType.LIMIT, OrderType.STOP_LIMIT] and self.price is None:
            raise ValueError(f"Price required for {self.order_type} orders")
            
        if self.order_type in [OrderType.STOP, OrderType.STOP_LIMIT] and self.stop_price is None:
            raise ValueError(f"Stop price required for {self.order_type} orders")
    
    def to_dict(self) -> Dict:
        """Convert order to dictionary"""
        return {
            'symbol': self.symbol,
            'side': self.side.value,
            'type': self.order_type.value,
            'quantity': self.quantity,
            'price': self.price,
            'stopPrice': self.stop_price,
            'timeInForce': self.time_in_force,
            'orderId': self.order_id,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'OrderRequest':
        """Create order from dictionary"""
        return cls(
            symbol=data['symbol'],
            side=OrderSide(data['side']),
            order_type=OrderType(data['type']),
            quantity=data['quantity'],
            price=data.get('price'),
            stop_price=data.get('stopPrice'),
            time_in_force=data.get('timeInForce', 'DAY'),
            order_id=data.get('orderId'),
            timestamp=datetime.fromisoformat(data['timestamp']) if data.get('timestamp') else None
        )
        
@dataclass
class OrderResponse:
    """Order response data class"""
    order_id: str
    symbol: str
    status: str
    filled_quantity: float = 0
    remaining_quantity: float = 0
    average_price: Optional[float] = None
    last_filled_price: Optional[float] = None
    last_filled_quantity: Optional[float] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        """Set timestamp if not provided"""
        if self.timestamp is None:
            self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict:
        """Convert response to dictionary"""
        return {
            'orderId': self.order_id,
            'symbol': self.symbol,
            'status': self.status,
            'filledQuantity': self.filled_quantity,
            'remainingQuantity': self.remaining_quantity,
            'averagePrice': self.average_price,
            'lastFilledPrice': self.last_filled_price,
            'lastFilledQuantity': self.last_filled_quantity,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'OrderResponse':
        """Create response from dictionary"""
        return cls(
            order_id=data['orderId'],
            symbol=data['symbol'],
            status=data['status'],
            filled_quantity=data.get('filledQuantity', 0),
            remaining_quantity=data.get('remainingQuantity', 0),
            average_price=data.get('averagePrice'),
            last_filled_price=data.get('lastFilledPrice'),
            last_filled_quantity=data.get('lastFilledQuantity'),
            timestamp=datetime.fromisoformat(data['timestamp']) if data.get('timestamp') else None
        )