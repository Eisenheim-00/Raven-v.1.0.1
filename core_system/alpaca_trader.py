# alpaca_trader.py - Modern Alpaca Trading Interface (APCA Authentication)
# Updated for correct APCA-API-KEY-ID and APCA-API-SECRET-KEY format

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, GetOrdersRequest
from alpaca.trading.enums import OrderSide, TimeInForce, QueryOrderStatus
from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame
import pandas as pd
from datetime import datetime, timedelta
import time
from config import Config

class AlpacaTrader:
    """
    Modern Alpaca trading interface for Raven using correct APCA authentication
    Updated for 2025 authentication format per https://docs.alpaca.markets/reference/authentication-2
    """
    
    def __init__(self, api_key_id, api_secret_key):
        self.api_key_id = api_key_id      # APCA-API-KEY-ID
        self.api_secret_key = api_secret_key  # APCA-API-SECRET-KEY
        
        # Initialize modern Alpaca Trading Client with correct APCA format
        try:
            self.trading_client = TradingClient(
                api_key=api_key_id,       # Uses APCA-API-KEY-ID
                secret_key=api_secret_key, # Uses APCA-API-SECRET-KEY
                paper=True  # Paper trading mode (safe!)
            )
            
            # Initialize data client for Bitcoin prices
            self.data_client = CryptoHistoricalDataClient(
                api_key=api_key_id,
                secret_key=api_secret_key
            )
            
            # Test connection
            account = self.trading_client.get_account()
            print(f"✅ Connected to Alpaca (APCA Authentication)")
            print(f"   Account Status: {account.status}")
            print(f"   Buying Power: ${float(account.buying_power):,.2f}")
            print(f"   Portfolio Value: ${float(account.portfolio_value):,.2f}")
            print(f"   Auth Format: APCA-API-KEY-ID / APCA-API-SECRET-KEY")
            
            # Trading parameters
            self.symbol = Config.SYMBOL
            self.position_size = Config.POSITION_SIZE_USD
            self.max_positions = Config.MAX_POSITIONS
            
            # Track trades
            self.trades_today = 0
            self.last_trade_time = None
            
        except Exception as e:
            print(f"❌ Alpaca connection failed: {e}")
            print(f"🔍 Check your APCA API keys in config.py")
            print(f"📖 Authentication Docs: https://docs.alpaca.markets/reference/authentication-2")
            print(f"🔑 Required format:")
            print(f"   APCA_API_KEY_ID = 'PK...' (starts with PK)")
            print(f"   APCA_API_SECRET_KEY = '...' (your secret key)")
            self.trading_client = None
            self.data_client = None
    
    def get_account_info(self):
        """
        Get current account information using APCA authentication
        """
        try:
            account = self.trading_client.get_account()
            return {
                'status': account.status,
                'buying_power': float(account.buying_power),
                'portfolio_value': float(account.portfolio_value),
                'equity': float(account.equity),
                'cash': float(account.cash),
                'day_trade_buying_power': float(account.day_trade_buying_power),
                'pattern_day_trader': account.pattern_day_trader
            }
        except Exception as e:
            print(f"❌ Failed to get account info: {e}")
            return None
    
    def get_current_position(self):
        """
        Check if Raven currently holds Bitcoin position
        """
        try:
            positions = self.trading_client.get_all_positions()
            
            for position in positions:
                if position.symbol == self.symbol:
                    return {
                        'symbol': position.symbol,
                        'qty': float(position.qty),
                        'side': position.side,
                        'market_value': float(position.market_value),
                        'cost_basis': float(position.cost_basis),
                        'unrealized_pl': float(position.unrealized_pl),
                        'unrealized_plpc': float(position.unrealized_plpc)
                    }
            
            return None  # No position
            
        except Exception as e:
            print(f"❌ Failed to get position: {e}")
            return None
    
    def get_current_bitcoin_price(self):
        """
        Get current Bitcoin price using APCA authenticated data API
        """
        try:
            # Get latest bars for Bitcoin
            request_params = CryptoBarsRequest(
                symbol_or_symbols=[self.symbol],
                timeframe=TimeFrame.Minute,
                limit=1
            )
            
            bars = self.data_client.get_crypto_bars(request_params)
            
            if bars and len(bars) > 0:
                latest_bar = bars[self.symbol][-1]
                return float(latest_bar.close)
            else:
                print("⚠️ No price data available from Alpaca")
                return None
                
        except Exception as e:
            print(f"⚠️ Failed to get Bitcoin price from Alpaca: {e}")
            return None
    
    def can_trade(self, signal):
        """
        Check if Raven can execute trades right now
        Enhanced safety checks for evolution system
        """
        reasons = []
        
        # Check 1: API connection
        if not self.trading_client:
            reasons.append("❌ No Alpaca connection")
            return False, reasons
        
        # Check 2: Daily trade limit
        if self.trades_today >= Config.MAX_DAILY_TRADES:
            reasons.append(f"❌ Daily trade limit reached ({self.trades_today})")
            return False, reasons
        
        # Check 3: Time between trades
        if self.last_trade_time:
            time_since_last = datetime.now() - self.last_trade_time
            min_interval = timedelta(minutes=Config.MIN_TIME_BETWEEN_TRADES)
            
            if time_since_last < min_interval:
                remaining = min_interval - time_since_last
                reasons.append(f"❌ Too soon since last trade ({remaining} remaining)")
                return False, reasons
        
        # Check 4: Account status
        account = self.get_account_info()
        if not account or account['status'] != 'ACTIVE':
            reasons.append("❌ Account not active")
            return False, reasons
        
        # Check 5: Sufficient buying power
        if account['buying_power'] < self.position_size:
            reasons.append(f"❌ Insufficient buying power: ${account['buying_power']:,.2f}")
            return False, reasons
        
        # Check 6: Position limits
        current_position = self.get_current_position()
        if current_position and abs(float(current_position['qty'])) >= self.max_positions:
            reasons.append(f"❌ Position limit reached")
            return False, reasons
        
        # All checks passed!
        reasons.append("✅ All trading checks passed")
        return True, reasons
    
    def calculate_position_size(self, signal):
        """
        Calculate how much Bitcoin Raven should buy/sell
        Enhanced for evolution system
        """
        try:
            # Use current price from signal or get fresh price
            current_price = signal.get('current_price')
            if not current_price:
                current_price = self.get_current_bitcoin_price()
            
            if not current_price:
                print("❌ Cannot get Bitcoin price for position sizing")
                return None
            
            # Calculate quantity based on position size
            quantity = self.position_size / current_price
            
            # Round to appropriate decimal places (crypto allows many decimals)
            quantity = round(quantity, 6)
            
            print(f"   Position size: ${self.position_size:,.2f} / ${current_price:,.2f} = {quantity} BTC")
            
            return quantity
            
        except Exception as e:
            print(f"❌ Failed to calculate position size: {e}")
            return None
    
    def execute_trade(self, signal):
        """
        Execute Raven's trade using APCA authenticated API
        Enhanced with evolution system metadata
        """
        if not self.trading_client:
            print("❌ No Alpaca connection")
            return None
        
        print(f"🎯 Raven executing {signal['trade_signal']} trade...")
        
        # Show evolution context
        if 'current_phase' in signal:
            print(f"   🎮 Evolution Phase: {signal['current_phase']}")
        if 'regime' in signal:
            print(f"   🎭 Market Regime: {signal['regime']}")
        
        # Check if we can trade
        can_trade, reasons = self.can_trade(signal)
        
        for reason in reasons:
            print(f"   {reason}")
        
        if not can_trade:
            return None
        
        # Calculate position size
        quantity = self.calculate_position_size(signal)
        if not quantity:
            print("❌ Could not calculate position size")
            return None
        
        try:
            # Determine order side and create order request
            if signal['trade_signal'] == 'BUY':
                side = OrderSide.BUY
            elif signal['trade_signal'] == 'SELL':
                side = OrderSide.SELL
            else:
                print("❌ Invalid trade signal")
                return None
            
            # Create market order request using APCA authenticated API
            order_request = MarketOrderRequest(
                symbol=self.symbol,
                qty=quantity,
                side=side,
                time_in_force=TimeInForce.DAY
            )
            
            print(f"   Placing {side.value} order for {quantity} {self.symbol}")
            print(f"   Estimated value: ${quantity * signal['current_price']:,.2f}")
            print(f"   Confidence: {signal['enhanced_confidence']:.3f}")
            print(f"   Auth: APCA-API-KEY-ID authentication")
            
            # Submit the order using APCA authenticated API
            order = self.trading_client.submit_order(order_data=order_request)
            
            # Wait a moment for order to process
            time.sleep(2)
            
            # Get updated order status
            order_status = self.trading_client.get_order_by_id(order.id)
            
            # Update tracking
            self.trades_today += 1
            self.last_trade_time = datetime.now()
            
            # Create trade result with evolution metadata
            trade_result = {
                'order_id': order.id,
                'symbol': order.symbol,
                'qty': float(order.qty),
                'side': order.side.value,
                'type': order.order_type.value,
                'status': order_status.status.value,
                'filled_qty': float(order_status.filled_qty) if order_status.filled_qty else 0,
                'filled_avg_price': float(order_status.filled_avg_price) if order_status.filled_avg_price else None,
                'submitted_at': order.submitted_at,
                'filled_at': order_status.filled_at,
                'signal_confidence': signal['enhanced_confidence'],
                'signal_reason': signal['confidence_reason'],
                # Evolution metadata
                'evolution_phase': signal.get('current_phase', 'unknown'),
                'market_regime': signal.get('regime', 'unknown'),
                'evolution_features': signal.get('evolution_features', []),
                'auth_format': 'APCA'
            }
            
            print(f"✅ Raven's evolution trade placed successfully!")
            print(f"   Order ID: {order.id}")
            print(f"   Status: {order_status.status.value}")
            print(f"   Authentication: APCA format")
            
            if order_status.status.value == 'filled':
                print(f"   Filled: {trade_result['filled_qty']} @ ${trade_result['filled_avg_price']:,.2f}")
            
            return trade_result
            
        except Exception as e:
            print(f"❌ Raven's trade execution failed: {e}")
            print(f"🔍 Check APCA authentication and API limits")
            return None
    
    def close_position(self, reason="Manual close"):
        """
        Close Raven's current Bitcoin position using APCA authentication
        """
        try:
            current_position = self.get_current_position()
            
            if not current_position:
                print("ℹ️ Raven has no position to close")
                return True
            
            print(f"🔄 Raven closing position: {current_position['qty']} {self.symbol}")
            print(f"   Current value: ${current_position['market_value']:,.2f}")
            print(f"   Unrealized P&L: ${current_position['unrealized_pl']:,.2f}")
            print(f"   Reason: {reason}")
            print(f"   Auth: APCA-API-KEY-ID")
            
            # Close position using APCA authenticated API
            close_response = self.trading_client.close_position(self.symbol)
            
            print(f"✅ Raven's position close order submitted: {close_response.id}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to close Raven's position: {e}")
            return False
    
    def get_portfolio_status(self):
        """
        Get Raven's overall portfolio performance
        Enhanced for evolution system
        """
        try:
            account = self.get_account_info()
            position = self.get_current_position()
            
            # Calculate today's P&L (simple approximation)
            todays_pnl = 0
            if position:
                todays_pnl = position['unrealized_pl']
            
            status = {
                'account': account,
                'current_position': position,
                'trades_today': self.trades_today,
                'todays_pnl': todays_pnl,
                'last_trade_time': self.last_trade_time,
                'portfolio_value': account['portfolio_value'] if account else 0,
                'auth_format': 'APCA'
            }
            
            return status
            
        except Exception as e:
            print(f"❌ Failed to get Raven's portfolio status: {e}")
            return None
    
    def get_recent_trades(self, days=1):
        """
        Get Raven's recent trade history using APCA authenticated API
        """
        try:
            # Get orders from last N days using APCA authenticated API
            since = datetime.now() - timedelta(days=days)
            
            # Create request for recent orders
            request_params = GetOrdersRequest(
                status=QueryOrderStatus.ALL,
                symbols=[self.symbol],
                after=since,
                limit=50
            )
            
            orders = self.trading_client.get_orders(filter=request_params)
            
            trades = []
            for order in orders:
                if order.status.value == 'filled':
                    trades.append({
                        'order_id': order.id,
                        'symbol': order.symbol,
                        'qty': float(order.qty),
                        'side': order.side.value,
                        'filled_price': float(order.filled_avg_price) if order.filled_avg_price else 0,
                        'filled_at': order.filled_at,
                        'value': float(order.qty) * float(order.filled_avg_price or 0)
                    })
            
            return trades
            
        except Exception as e:
            print(f"❌ Failed to get Raven's recent trades: {e}")
            return []
    
    def cancel_all_orders(self):
        """
        Cancel all of Raven's open orders (emergency stop)
        """
        try:
            print("🛑 Raven canceling all open orders...")
            cancel_responses = self.trading_client.cancel_orders()
            
            canceled_count = 0
            for response in cancel_responses:
                if response.success:
                    canceled_count += 1
            
            print(f"✅ Raven canceled {canceled_count} orders")
            return True
            
        except Exception as e:
            print(f"❌ Failed to cancel Raven's orders: {e}")
            return False
    
    def print_portfolio_summary(self):
        """
        Print evolution-enhanced portfolio summary with APCA authentication
        """
        print(f"\n🐦‍⬛ RAVEN'S EVOLUTION PORTFOLIO SUMMARY")
        print(f"="*50)
        
        status = self.get_portfolio_status()
        if not status:
            print("❌ Could not get portfolio status")
            return
        
        account = status['account']
        position = status['current_position']
        
        print(f"💰 Account Information:")
        print(f"   Portfolio Value: ${account['portfolio_value']:,.2f}")
        print(f"   Cash: ${account['cash']:,.2f}")
        print(f"   Buying Power: ${account['buying_power']:,.2f}")
        print(f"   Today's P&L: ${status['todays_pnl']:,.2f}")
        print(f"   Auth Format: APCA-API-KEY-ID")
        
        if position:
            print(f"\n📊 Current Position:")
            print(f"   Symbol: {position['symbol']}")
            print(f"   Quantity: {position['qty']}")
            print(f"   Market Value: ${position['market_value']:,.2f}")
            print(f"   Unrealized P&L: ${position['unrealized_pl']:,.2f} ({position['unrealized_plpc']:.2%})")
        else:
            print(f"\n📊 Current Position: None (Raven is hunting)")
        
        print(f"\n📈 Trading Activity:")
        print(f"   Trades Today: {status['trades_today']}")
        if status['last_trade_time']:
            print(f"   Last Trade: {status['last_trade_time'].strftime('%H:%M:%S')}")
        
        # Recent trades
        recent_trades = self.get_recent_trades(days=1)
        if recent_trades:
            print(f"\n📋 Recent Trades (24h):")
            for trade in recent_trades[-3:]:  # Show last 3 trades
                print(f"   {trade['side'].upper()} {trade['qty']} @ ${trade['filled_price']:,.2f}")
        
        print(f"="*50)
    
    def test_connection(self):
        """
        Test Raven's connection to Alpaca with APCA authentication
        """
        try:
            print("🔍 Testing Raven's APCA authentication connection...")
            
            # Test account access
            account = self.trading_client.get_account()
            print(f"   ✅ Account access: {account.status}")
            
            # Test Bitcoin price data
            price = self.get_current_bitcoin_price()
            if price:
                print(f"   ✅ Bitcoin price data: ${price:,.2f}")
            else:
                print(f"   ⚠️ Bitcoin price data: Not available")
            
            # Test position access
            positions = self.trading_client.get_all_positions()
            print(f"   ✅ Position access: {len(positions)} positions")
            
            print(f"✅ Raven's APCA connection test completed successfully!")
            print(f"📖 Using APCA-API-KEY-ID / APCA-API-SECRET-KEY format")
            print(f"🎯 Paper trading domain: https://paper-api.alpaca.markets")
            return True
            
        except Exception as e:
            print(f"❌ Raven's APCA connection test failed: {e}")
            print(f"🔧 Troubleshooting:")
            print(f"   - Check APCA_API_KEY_ID in config.py (starts with 'PK')")
            print(f"   - Check APCA_API_SECRET_KEY in config.py")
            print(f"   - Verify paper trading account is active")
            print(f"   - Check Alpaca status: https://status.alpaca.markets/")
            print(f"📖 Auth docs: https://docs.alpaca.markets/reference/authentication-2")
            return False