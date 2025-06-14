# logger.py - Bot Activity Logger
# Updated to use centralized database paths in data_&_models/databases/

import logging
import pandas as pd
import sqlite3
import json
from datetime import datetime
import os
from config import Config

class RavenLogger:
    """
    Comprehensive logging system for Raven - AI Trading Bot
    Think of this as Raven's memory keeper - it remembers everything!
    Updated to use centralized database paths.
    """
    
    def __init__(self):
        # Use centralized paths from Config
        self.log_file = Config.get_log_file_path()
        self.db_path = Config.get_logs_db_path()  # Centralized in data_&_models/databases/
        
        # Setup file logging
        self.setup_file_logging()
        
        # Setup database logging
        self.setup_database_logging()
        
        print(f"📝 Raven's memory system initialized")
        print(f"   Log file: {self.log_file}")
        print(f"   Database: {self.db_path}")
        print(f"   📁 All data centralized in: {os.path.dirname(os.path.dirname(self.db_path))}")
    
    def setup_file_logging(self):
        """
        Setup file-based logging with centralized log directory
        """
        # Ensure log directory exists
        log_dir = os.path.dirname(self.log_file)
        os.makedirs(log_dir, exist_ok=True)
        
        # Create logger
        self.logger = logging.getLogger('Raven')
        self.logger.setLevel(logging.INFO)
        
        # Remove existing handlers to avoid duplicates
        if self.logger.handlers:
            self.logger.handlers.clear()
        
        # File handler
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers
        self.logger.addHandler(file_handler)
        # Note: We don't add console handler to avoid duplicate console output
    
    def setup_database_logging(self):
        """
        Setup database for structured logging in centralized location
        """
        try:
            # Ensure database directory exists
            db_dir = os.path.dirname(self.db_path)
            os.makedirs(db_dir, exist_ok=True)
            
            conn = sqlite3.connect(self.db_path)
            
            # Table for general bot activities
            conn.execute('''
                CREATE TABLE IF NOT EXISTS bot_activities (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    activity_type TEXT,
                    message TEXT,
                    details TEXT,
                    created_at TEXT
                )
            ''')
            
            # Table for trading signals
            conn.execute('''
                CREATE TABLE IF NOT EXISTS trading_signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    prediction TEXT,
                    probability REAL,
                    enhanced_confidence REAL,
                    trade_signal TEXT,
                    time_reason TEXT,
                    confidence_reason TEXT,
                    hour INTEGER,
                    day_of_week INTEGER,
                    current_price REAL,
                    volatility REAL,
                    rsi REAL,
                    created_at TEXT
                )
            ''')
            
            # Table for executed trades
            conn.execute('''
                CREATE TABLE IF NOT EXISTS executed_trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    order_id TEXT,
                    symbol TEXT,
                    side TEXT,
                    quantity REAL,
                    filled_price REAL,
                    filled_qty REAL,
                    status TEXT,
                    signal_confidence REAL,
                    signal_reason TEXT,
                    created_at TEXT
                )
            ''')
            
            # Table for portfolio snapshots
            conn.execute('''
                CREATE TABLE IF NOT EXISTS portfolio_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    portfolio_value REAL,
                    cash REAL,
                    position_value REAL,
                    unrealized_pnl REAL,
                    trades_today INTEGER,
                    created_at TEXT
                )
            ''')
            
            conn.commit()
            conn.close()
            
            print(f"✅ Centralized logging database initialized: {self.db_path}")
            
        except Exception as e:
            print(f"❌ Database logging setup failed: {e}")
    
    def log_info(self, message, details=None):
        """
        Log general information (like writing a normal diary entry)
        """
        self.logger.info(message)
        self._log_to_database('INFO', message, details)
    
    def log_error(self, message, details=None):
        """
        Log errors (like writing about problems)
        """
        self.logger.error(message)
        self._log_to_database('ERROR', message, details)
    
    def log_warning(self, message, details=None):
        """
        Log warnings (like noting concerns)
        """
        self.logger.warning(message)
        self._log_to_database('WARNING', message, details)
    
    def log_trade(self, signal, trade_result):
        """
        Log executed trades (like recording your purchases)
        """
        message = f"TRADE EXECUTED: {trade_result['side'].upper()} {trade_result['qty']} {trade_result['symbol']}"
        self.log_info(message)
        
        # Log to centralized database
        try:
            conn = sqlite3.connect(self.db_path)
            
            conn.execute('''
                INSERT INTO executed_trades 
                (timestamp, order_id, symbol, side, quantity, filled_price, filled_qty, 
                 status, signal_confidence, signal_reason, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                signal['timestamp'].strftime('%Y-%m-%d %H:%M:%S'),
                trade_result['order_id'],
                trade_result['symbol'],
                trade_result['side'],
                trade_result['qty'],
                trade_result.get('filled_avg_price', 0),
                trade_result.get('filled_qty', 0),
                trade_result['status'],
                signal['enhanced_confidence'],
                signal['confidence_reason'],
                datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.log_error(f"Failed to log trade to centralized database: {e}")
    
    def log_signal(self, signal):
        """
        Log trading signals (like recording your thoughts)
        """
        message = f"SIGNAL: {signal['prediction']} (conf: {signal['enhanced_confidence']:.3f}) -> {signal['trade_signal']}"
        self.log_info(message)
        
        # Log to centralized database
        try:
            conn = sqlite3.connect(self.db_path)
            
            conn.execute('''
                INSERT INTO trading_signals 
                (timestamp, prediction, probability, enhanced_confidence, trade_signal,
                 time_reason, confidence_reason, hour, day_of_week, current_price,
                 volatility, rsi, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                signal['timestamp'].strftime('%Y-%m-%d %H:%M:%S'),
                signal['prediction'],
                signal['probability'],
                signal['enhanced_confidence'],
                signal['trade_signal'],
                signal['time_reason'],
                signal['confidence_reason'],
                signal['hour'],
                signal['day_of_week'],
                signal['current_price'],
                signal.get('volatility', 0),
                signal.get('rsi', 0),
                datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.log_error(f"Failed to log signal to centralized database: {e}")
    
    def log_cycle(self, signal):
        """
        Log completion of trading cycle (like ending a diary entry)
        """
        if signal:
            self.log_signal(signal)
        
        message = "Trading cycle completed"
        self.log_info(message)
    
    def log_portfolio_snapshot(self, portfolio_status):
        """
        Log portfolio status (like taking a photo of your collection)
        """
        if not portfolio_status:
            return
        
        try:
            account = portfolio_status['account']
            position = portfolio_status.get('current_position')
            
            # Log to centralized database
            conn = sqlite3.connect(self.db_path)
            
            conn.execute('''
                INSERT INTO portfolio_snapshots 
                (timestamp, portfolio_value, cash, position_value, unrealized_pnl, 
                 trades_today, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                account['portfolio_value'],
                account['cash'],
                position['market_value'] if position else 0,
                position['unrealized_pl'] if position else 0,
                portfolio_status['trades_today'],
                datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            ))
            
            conn.commit()
            conn.close()
            
            message = f"Portfolio snapshot: ${account['portfolio_value']:,.2f} (P&L: ${portfolio_status['todays_pnl']:,.2f})"
            self.log_info(message)
            
        except Exception as e:
            self.log_error(f"Failed to log portfolio snapshot to centralized database: {e}")
    
    def _log_to_database(self, activity_type, message, details):
        """
        Internal method to log to centralized database
        """
        try:
            conn = sqlite3.connect(self.db_path)
            
            details_json = json.dumps(details) if details else None
            
            conn.execute('''
                INSERT INTO bot_activities 
                (timestamp, activity_type, message, details, created_at)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                activity_type,
                message,
                details_json,
                datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            # Don't log this error to avoid infinite loop
            print(f"Centralized database logging failed: {e}")
    
    def get_trading_performance(self, days=7):
        """
        Get trading performance summary from centralized database
        """
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Get signals from last N days
            since = (datetime.now() - pd.Timedelta(days=days)).strftime('%Y-%m-%d')
            
            signals_df = pd.read_sql_query('''
                SELECT * FROM trading_signals 
                WHERE timestamp >= ?
                ORDER BY timestamp
            ''', conn, params=(since,))
            
            trades_df = pd.read_sql_query('''
                SELECT * FROM executed_trades 
                WHERE timestamp >= ?
                ORDER BY timestamp
            ''', conn, params=(since,))
            
            conn.close()
            
            # Calculate performance metrics
            performance = {
                'total_signals': len(signals_df),
                'trade_signals': len(signals_df[signals_df['trade_signal'].isin(['BUY', 'SELL'])]),
                'executed_trades': len(trades_df),
                'avg_confidence': signals_df['enhanced_confidence'].mean() if len(signals_df) > 0 else 0,
                'signals_by_hour': signals_df.groupby('hour').size().to_dict() if len(signals_df) > 0 else {},
                'signals_by_day': signals_df.groupby('day_of_week').size().to_dict() if len(signals_df) > 0 else {}
            }
            
            return performance
            
        except Exception as e:
            self.log_error(f"Failed to get trading performance from centralized database: {e}")
            return None
    
    def get_recent_activities(self, hours=24):
        """
        Get recent bot activities from centralized database
        """
        try:
            conn = sqlite3.connect(self.db_path)
            
            since = (datetime.now() - pd.Timedelta(hours=hours)).strftime('%Y-%m-%d %H:%M:%S')
            
            activities_df = pd.read_sql_query('''
                SELECT * FROM bot_activities 
                WHERE timestamp >= ?
                ORDER BY timestamp DESC
                LIMIT 50
            ''', conn, params=(since,))
            
            conn.close()
            
            return activities_df
            
        except Exception as e:
            self.log_error(f"Failed to get recent activities from centralized database: {e}")
            return None
    
    def print_daily_summary(self):
        """
        Print a summary of today's activities from centralized database
        """
        print(f"\n📊 DAILY ACTIVITY SUMMARY - {datetime.now().strftime('%Y-%m-%d')}")
        print(f"📁 Data source: {self.db_path}")
        print(f"="*60)
        
        performance = self.get_trading_performance(days=1)
        
        if performance:
            print(f"🤖 Bot Activity:")
            print(f"   Total signals generated: {performance['total_signals']}")
            print(f"   Trade signals (BUY/SELL): {performance['trade_signals']}")
            print(f"   Trades executed: {performance['executed_trades']}")
            print(f"   Average confidence: {performance['avg_confidence']:.3f}")
            
            if performance['signals_by_hour']:
                print(f"\n⏰ Signals by hour (UTC):")
                for hour, count in sorted(performance['signals_by_hour'].items()):
                    print(f"   {hour:02d}:00 - {count} signals")
        
        # Get recent activities
        activities = self.get_recent_activities(hours=24)
        if activities is not None and len(activities) > 0:
            print(f"\n📋 Recent Activities:")
            for _, activity in activities.head(10).iterrows():
                timestamp = pd.to_datetime(activity['timestamp']).strftime('%H:%M')
                print(f"   {timestamp} | {activity['activity_type']} | {activity['message']}")
        
        print(f"="*60)
    
    def export_logs(self, output_dir=None):
        """
        Export all logs to CSV files in centralized location
        """
        if not output_dir:
            # Export to data_&_models/exports/ by default
            data_dir = Config.get_data_models_dir()
            output_dir = os.path.join(data_dir, "exports")
            os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Export each table
            tables = ['bot_activities', 'trading_signals', 'executed_trades', 'portfolio_snapshots']
            
            for table in tables:
                df = pd.read_sql_query(f"SELECT * FROM {table}", conn)
                csv_file = os.path.join(output_dir, f"raven_logs_{table}_{timestamp}.csv")
                df.to_csv(csv_file, index=False)
                print(f"✅ Exported {len(df)} records to {csv_file}")
            
            conn.close()
            
            self.log_info(f"Logs exported to {output_dir}")
            print(f"📁 All exports saved to: {output_dir}")
            
        except Exception as e:
            self.log_error(f"Failed to export logs: {e}")
    
    def cleanup_old_logs(self, days_to_keep=30):
        """
        Clean up old log files and database records
        """
        try:
            cutoff_date = (datetime.now() - pd.Timedelta(days=days_to_keep)).strftime('%Y-%m-%d')
            
            conn = sqlite3.connect(self.db_path)
            
            # Delete old records
            tables = ['bot_activities', 'trading_signals', 'executed_trades', 'portfolio_snapshots']
            
            for table in tables:
                result = conn.execute(f"DELETE FROM {table} WHERE timestamp < ?", (cutoff_date,))
                deleted_count = result.rowcount
                if deleted_count > 0:
                    print(f"🧹 Cleaned up {deleted_count} old records from {table}")
            
            conn.commit()
            conn.close()
            
            self.log_info(f"Cleaned up logs older than {days_to_keep} days from centralized database")
            
        except Exception as e:
            self.log_error(f"Failed to cleanup old logs: {e}")
    
    def get_database_info(self):
        """Get information about the centralized logging database"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Get table sizes
            tables = ['bot_activities', 'trading_signals', 'executed_trades', 'portfolio_snapshots']
            table_info = {}
            
            for table in tables:
                count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                table_info[table] = count
            
            conn.close()
            
            return {
                'database_path': self.db_path,
                'database_size_mb': os.path.getsize(self.db_path) / (1024 * 1024),
                'table_record_counts': table_info,
                'log_file_path': self.log_file,
                'centralized_location': os.path.dirname(os.path.dirname(self.db_path))
            }
            
        except Exception as e:
            print(f"Failed to get database info: {e}")
            return None