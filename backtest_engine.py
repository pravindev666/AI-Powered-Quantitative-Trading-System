"""Enhanced backtesting engine with advanced performance analytics and risk management"""
import numpy as np
import pandas as pd
from datetime import datetime
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

@dataclass
class TradeStats:
    """Container for trade statistics"""
    entry_price: float
    entry_time: pd.Timestamp
    position_size: float
    stop_price: Optional[float] = None
    target_price: Optional[float] = None
    exit_price: Optional[float] = None
    exit_time: Optional[pd.Timestamp] = None
    pnl: Optional[float] = None
    pnl_pct: Optional[float] = None
    trade_duration: Optional[pd.Timedelta] = None
    exit_reason: Optional[str] = None
    mae: Optional[float] = None  # Maximum Adverse Excursion
    mfe: Optional[float] = None  # Maximum Favorable Excursion

class PortfolioMetrics:
    """Calculate and track portfolio performance metrics"""
    
    def __init__(self):
        """Initialize metrics tracker"""
        self.trades: List[TradeStats] = []
        self.daily_returns: List[float] = []
        self.equity_curve: List[float] = []
        
    def add_trade(self, trade: TradeStats):
        """Add completed trade to history"""
        self.trades.append(trade)
        
    def add_daily_return(self, ret: float):
        """Add daily return to series"""
        self.daily_returns.append(ret)
        
    def update_equity(self, equity: float):
        """Update equity curve"""
        self.equity_curve.append(equity)
        
    def calculate_metrics(self) -> Dict:
        """Calculate comprehensive performance metrics"""
        returns = pd.Series(self.daily_returns)
        equity = pd.Series(self.equity_curve)
        
        # Basic metrics
        total_trades = len(self.trades)
        winning_trades = len([t for t in self.trades if t.pnl > 0])
        losing_trades = len([t for t in self.trades if t.pnl < 0])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # Returns and volatility
        total_return = (equity.iloc[-1] / equity.iloc[0] - 1) if len(equity) > 0 else 0
        annual_return = (1 + total_return) ** (252 / len(returns)) - 1 if len(returns) > 0 else 0
        volatility = returns.std() * np.sqrt(252) if len(returns) > 0 else 0
        
        # Drawdown analysis
        rolling_max = equity.expanding().max()
        drawdown = (equity - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        # Risk metrics
        sharpe = annual_return / volatility if volatility > 0 else 0
        sortino = (annual_return / (returns[returns < 0].std() * np.sqrt(252))) if len(returns[returns < 0]) > 0 else 0
        calmar = abs(annual_return / max_drawdown) if max_drawdown < 0 else 0
        
        # Trade analysis
        avg_win = np.mean([t.pnl for t in self.trades if t.pnl > 0]) if winning_trades > 0 else 0
        avg_loss = np.mean([t.pnl for t in self.trades if t.pnl < 0]) if losing_trades > 0 else 0
        profit_factor = abs(avg_win * winning_trades / (avg_loss * losing_trades)) if losing_trades > 0 and avg_loss != 0 else 0
        
        # Average trade metrics
        avg_mae = np.mean([t.mae for t in self.trades if t.mae is not None])
        avg_mfe = np.mean([t.mfe for t in self.trades if t.mfe is not None])
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'calmar_ratio': calmar,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'avg_mae': avg_mae,
            'avg_mfe': avg_mfe
        }

class EnhancedBacktest:
    """Advanced backtesting engine with risk management and analytics"""
    
    def __init__(self, df: pd.DataFrame, risk_free_rate: float = 0.05):
        """Initialize backtester with data and parameters"""
        self.df = df.copy()
        self.risk_free_rate = risk_free_rate
        self.metrics = PortfolioMetrics()
        self.current_trade: Optional[TradeStats] = None
        self.equity = 100_000  # Starting equity
        self._setup_logging()
        
    def _setup_logging(self):
        """Configure logging"""
        log_dir = Path('backtest_logs')
        log_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = log_dir / f'backtest_{timestamp}.log'
        
        handler = logging.FileHandler(log_file)
        handler.setFormatter(
            logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        )
        logger.addHandler(handler)
        
    def run(self, 
            initial_capital: float = 100_000,
            position_size: float = 0.02,
            trailing_stop: float = 0.02,
            time_stop_days: int = 5,
            take_profit: float = 0.05) -> Dict:
        """
        Run backtest with advanced position management
        
        Args:
            initial_capital: Starting capital
            position_size: Position size as fraction of capital
            trailing_stop: Trailing stop distance as fraction
            time_stop_days: Maximum days to hold position
            take_profit: Take profit target as fraction
        """
        logger.info(f"Starting backtest with {initial_capital:,.0f} capital")
        
        self.equity = initial_capital
        self.metrics = PortfolioMetrics()
        self.df['Returns'] = self.df['Close'].pct_change()
        
        # Track highest price since entry for trailing stops
        highest_since_entry = 0
        days_in_trade = 0
        
        for i in range(1, len(self.df)):
            current_row = self.df.iloc[i]
            prev_row = self.df.iloc[i-1]
            
            # Update daily metrics
            daily_return = current_row['Returns']
            self.metrics.add_daily_return(daily_return)
            self.metrics.update_equity(self.equity)
            
            # Check for exit if in position
            if self.current_trade is not None:
                days_in_trade += 1
                current_price = current_row['Close']
                
                # Update MAE/MFE tracking
                pnl_pct = (current_price / self.current_trade.entry_price - 1)
                self.current_trade.mae = min(self.current_trade.mae or 0, pnl_pct)
                self.current_trade.mfe = max(self.current_trade.mfe or 0, pnl_pct)
                
                # Update trailing stop
                highest_since_entry = max(highest_since_entry, current_price)
                trailing_stop_price = highest_since_entry * (1 - trailing_stop)
                
                # Check exit conditions
                exit_reason = None
                if current_price <= trailing_stop_price:
                    exit_reason = "Trailing Stop"
                elif days_in_trade >= time_stop_days:
                    exit_reason = "Time Stop"
                elif (current_price / self.current_trade.entry_price - 1) >= take_profit:
                    exit_reason = "Take Profit"
                    
                if exit_reason:
                    self._exit_trade(current_price, current_row.name, exit_reason)
                    days_in_trade = 0
                    highest_since_entry = 0
            
            # Check for entry if not in position
            if self.current_trade is None:
                signal = current_row.get('Signal', 0)
                
                if signal > 0:  # Entry signal
                    size = position_size * self.equity
                    self._enter_trade(current_row['Close'], current_row.name, size)
                    highest_since_entry = current_row['Close']
                    days_in_trade = 0
        
        # Close any open position at end of test
        if self.current_trade is not None:
            self._exit_trade(self.df['Close'].iloc[-1], self.df.index[-1], "Test End")
        
        # Calculate final metrics
        results = self.metrics.calculate_metrics()
        self._log_results(results)
        
        return {
            'metrics': results,
            'equity_curve': pd.Series(self.metrics.equity_curve, index=self.df.index),
            'trades': self.metrics.trades
        }
    
    def _enter_trade(self, price: float, timestamp: pd.Timestamp, size: float):
        """Record trade entry"""
        self.current_trade = TradeStats(
            entry_price=price,
            entry_time=timestamp,
            position_size=size,
            mae=0,
            mfe=0
        )
        logger.info(f"Entered trade at {price:.2f}")
    
    def _exit_trade(self, price: float, timestamp: pd.Timestamp, reason: str):
        """Record trade exit and update metrics"""
        if self.current_trade is None:
            return
            
        # Calculate P&L
        pnl = (price - self.current_trade.entry_price) * self.current_trade.position_size
        pnl_pct = price / self.current_trade.entry_price - 1
        
        # Update trade record
        self.current_trade.exit_price = price
        self.current_trade.exit_time = timestamp
        self.current_trade.pnl = pnl
        self.current_trade.pnl_pct = pnl_pct
        self.current_trade.exit_reason = reason
        self.current_trade.trade_duration = timestamp - self.current_trade.entry_time
        
        # Update portfolio
        self.equity += pnl
        
        # Log trade
        logger.info(
            f"Exited trade at {price:.2f} ({reason}), "
            f"P&L: {pnl:,.2f} ({pnl_pct:.1%})"
        )
        
        # Store completed trade
        self.metrics.add_trade(self.current_trade)
        self.current_trade = None
    
    def _log_results(self, results: Dict):
        """Log backtest results"""
        logger.info("\n=== Backtest Results ===")
        logger.info(f"Total Return: {results['total_return']:.1%}")
        logger.info(f"Annual Return: {results['annual_return']:.1%}")
        logger.info(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
        logger.info(f"Max Drawdown: {results['max_drawdown']:.1%}")
        logger.info(f"Win Rate: {results['win_rate']:.1%}")
        logger.info(f"Profit Factor: {results['profit_factor']:.2f}")
        
    def plot_results(self, results: Dict):
        """Generate performance visualization"""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # Create figure with subplots
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12))
            
            # Plot equity curve
            results['equity_curve'].plot(ax=ax1)
            ax1.set_title('Equity Curve')
            ax1.grid(True)
            
            # Plot drawdown
            drawdown = (results['equity_curve'] / results['equity_curve'].expanding().max() - 1)
            drawdown.plot(ax=ax2, color='red', alpha=0.7)
            ax2.set_title('Drawdown')
            ax2.grid(True)
            
            # Plot trade P&Ls
            trade_pnls = [t.pnl for t in results['trades']]
            sns.histplot(trade_pnls, ax=ax3, bins=50)
            ax3.set_title('Trade P&L Distribution')
            ax3.grid(True)
            
            plt.tight_layout()
            
            # Save plot
            plots_dir = Path('backtest_plots')
            plots_dir.mkdir(exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            plot_file = plots_dir / f'backtest_results_{timestamp}.png'
            plt.savefig(plot_file)
            logger.info(f"Results plot saved to {plot_file}")
            
        except ImportError:
            logger.warning("Plotting requires matplotlib and seaborn")
            
        except Exception as e:
            logger.error(f"Error plotting results: {str(e)}")