# data_manager.py - Bitcoin Data Collection
# Updated to use centralized database paths in data_&_models/databases/

import pandas as pd
import numpy as np
import sqlite3
import yfinance as yf
import requests
from datetime import datetime, timedelta
import time
import os
from config import Config

class DataManager:
    """
    Manages Bitcoin data collection and storage
    Think of this as your data collector robot
    Updated to use centralized database paths.
    """
    
    def __init__(self):
        # Use centralized database path from Config
        self.db_path = Config.get_bitcoin_data_db_path()
        self.symbol = Config.SYMBOL
        self.setup_database()
        
        print("📊 Data Manager initialized")
        print(f"   📁 Bitcoin data database: {self.db_path}")
        print(f"   📁 Centralized in: {os.path.dirname(os.path.dirname(self.db_path))}")
    
    def setup_database(self):
        """
        Create centralized database to store Bitcoin data
        """
        try:
            # Ensure database directory exists
            db_dir = os.path.dirname(self.db_path)
            os.makedirs(db_dir, exist_ok=True)
            
            conn = sqlite3.connect(self.db_path)
            
            # Create table for hourly Bitcoin data
            conn.execute('''
                CREATE TABLE IF NOT EXISTS btc_hourly (
                    timestamp TEXT PRIMARY KEY,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume REAL,
                    created_at TEXT
                )
            ''')
            
            conn.commit()
            conn.close()
            print(f"✅ Centralized Bitcoin database setup complete: {self.db_path}")
            
        except Exception as e:
            print(f"❌ Database setup failed: {e}")
    
    def get_latest_bitcoin_price(self):
        """
        Get current Bitcoin price from multiple sources
        (Like checking the price at different stores)
        """
        try:
            # Method 1: Yahoo Finance (most reliable)
            print("   Trying Yahoo Finance...")
            ticker = yf.Ticker("BTC-USD")
            data = ticker.history(period="1d", interval="1h")
            
            if not data.empty:
                latest = data.iloc[-1]
                price_data = {
                    'timestamp': datetime.now().replace(minute=0, second=0, microsecond=0),
                    'open': float(latest['Open']),
                    'high': float(latest['High']),
                    'low': float(latest['Low']),
                    'close': float(latest['Close']),
                    'volume': float(latest['Volume'])
                }
                print(f"   ✅ Got Bitcoin price: ${price_data['close']:,.2f}")
                return price_data
            
        except Exception as e:
            print(f"   ❌ Yahoo Finance failed: {e}")
        
        try:
            # Method 2: CoinGecko API (backup)
            print("   Trying CoinGecko API...")
            url = "https://api.coingecko.com/api/v3/simple/price"
            params = {
                'ids': 'bitcoin',
                'vs_currencies': 'usd',
                'include_24hr_vol': 'true',
                'include_24hr_change': 'true'
            }
            
            response = requests.get(url, params=params, timeout=10)
            data = response.json()
            
            if 'bitcoin' in data:
                price = data['bitcoin']['usd']
                volume = data['bitcoin'].get('usd_24h_vol', 0)
                
                price_data = {
                    'timestamp': datetime.now().replace(minute=0, second=0, microsecond=0),
                    'open': price,
                    'high': price,
                    'low': price,
                    'close': price,
                    'volume': volume
                }
                print(f"   ✅ Got Bitcoin price: ${price_data['close']:,.2f}")
                return price_data
                
        except Exception as e:
            print(f"   ❌ CoinGecko API failed: {e}")
        
        print("   ❌ All data sources failed")
        return None
    
    def store_price_data(self, price_data):
        """
        Save price data to centralized database
        """
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Convert timestamp to string
            timestamp_str = price_data['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
            
            # Insert or replace data
            conn.execute('''
                INSERT OR REPLACE INTO btc_hourly 
                (timestamp, open, high, low, close, volume, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                timestamp_str,
                price_data['open'],
                price_data['high'],
                price_data['low'],
                price_data['close'],
                price_data['volume'],
                datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            ))
            
            conn.commit()
            conn.close()
            
            print(f"   ✅ Price data saved to centralized database")
            return True
            
        except Exception as e:
            print(f"   ❌ Failed to save data to centralized database: {e}")
            return False
    
    def get_historical_data(self, hours=200):
        """
        Get historical data for feature calculation from centralized database
        """
        try:
            conn = sqlite3.connect(self.db_path)
            
            query = """
                SELECT * FROM btc_hourly 
                ORDER BY timestamp DESC 
                LIMIT ?
            """
            
            df = pd.read_sql_query(query, conn, params=(hours,))
            conn.close()
            
            if df.empty:
                print("   ⚠️ No historical data in centralized database, downloading...")
                return self.download_initial_data()
            
            # Convert timestamp and sort
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            df.sort_index(inplace=True)
            
            print(f"   ✅ Retrieved {len(df)} hours of historical data from centralized database")
            return df
            
        except Exception as e:
            print(f"   ❌ Failed to get historical data from centralized database: {e}")
            return None
    
    def download_initial_data(self):
        """
        Download initial historical data if centralized database is empty
        """
        try:
            print("   📥 Downloading initial Bitcoin data to centralized database...")
            
            # Get 3 months of hourly data
            ticker = yf.Ticker("BTC-USD")
            data = ticker.history(period="3mo", interval="1h")
            
            if data.empty:
                print("   ❌ Failed to download initial data")
                return None
            
            # Store all historical data in centralized database
            conn = sqlite3.connect(self.db_path)
            
            for timestamp, row in data.iterrows():
                timestamp_str = timestamp.strftime('%Y-%m-%d %H:%M:%S')
                
                conn.execute('''
                    INSERT OR REPLACE INTO btc_hourly 
                    (timestamp, open, high, low, close, volume, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    timestamp_str,
                    float(row['Open']),
                    float(row['High']),
                    float(row['Low']),
                    float(row['Close']),
                    float(row['Volume']),
                    datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                ))
            
            conn.commit()
            conn.close()
            
            print(f"   ✅ Downloaded and stored {len(data)} hours of data in centralized database")
            
            # Return the data
            df = data.copy()
            df.index.name = 'timestamp'
            return df
            
        except Exception as e:
            print(f"   ❌ Failed to download initial data to centralized database: {e}")
            return None
    
    def create_features(self, df):
        """
        Create all 31 features for your 60.8% accuracy model
        (Like doing homework with the data)
        """
        if df is None or df.empty:
            return None
        
        print("   🛠️ Creating features for prediction...")
        
        # Make a copy to avoid modifying original
        df = df.copy()
        
        # Calculate returns
        df['returns'] = df['close'].pct_change()
        df['future_returns'] = df['returns'].shift(-1)
        df['target'] = np.where(df['future_returns'] >= 0, 1, 0)
        
        # 1. Price vs Moving Average features
        for window in [6, 12, 24, 48]:
            df[f'ma_{window}'] = df['close'].rolling(window).mean()
            df[f'price_vs_ma_{window}'] = (df['close'].shift(1) / df[f'ma_{window}'].shift(1)) - 1
        
        # 2. Lagged returns
        for lag in [1, 2, 3, 6, 12, 24]:
            df[f'returns_lag_{lag}'] = df['returns'].shift(lag)
        
        # 3. Returns statistics
        for window in [6, 12, 24]:
            df[f'returns_mean_{window}'] = df['returns'].rolling(window).mean().shift(1)
            df[f'returns_std_{window}'] = df['returns'].rolling(window).std().shift(1)
        
        # 4. Momentum features
        for lag in [6, 12, 24]:
            df[f'momentum_{lag}'] = (df['close'].shift(1) / df['close'].shift(lag+1)) - 1
        
        # 5. Volume features
        df['volume_ma'] = df['volume'].rolling(24).mean()
        df['volume_ratio'] = df['volume'].shift(1) / df['volume_ma'].shift(1)
        df['volume_log'] = np.log(df['volume'].shift(1) + 1)
        
        # 6. Technical indicators
        df['rsi'] = self._calculate_rsi(df['close'])
        df['rsi_lag'] = df['rsi'].shift(1)
        
        bb_upper, bb_lower = self._calculate_bollinger_bands(df['close'])
        df['bb_position'] = ((df['close'].shift(1) - bb_lower.shift(1)) / 
                            (bb_upper.shift(1) - bb_lower.shift(1)))
        
        df['volatility_lag'] = df['returns'].rolling(24).std().shift(1)
        
        # 7. Market structure
        df['gap'] = (df['open'] / df['close'].shift(1)) - 1
        df['trend_strength'] = abs(df['price_vs_ma_24'])
        
        # 8. Direction lags
        for lag in [1, 2, 3, 6]:
            df[f'direction_lag_{lag}'] = np.where(df[f'returns_lag_{lag}'] >= 0, 1, 0)
        
        # 9. Time features
        df['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
        
        # 10. Market regime
        vol_q75 = df['volatility_lag'].rolling(168).quantile(0.75)
        df['high_vol_regime'] = (df['volatility_lag'] > vol_q75).astype(int)
        
        # Clean data
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.ffill(inplace=True, limit=10)  # Updated method
        df.fillna(0, inplace=True)
        
        print(f"   ✅ Features created successfully")
        return df
    
    def _calculate_rsi(self, prices, period=14):
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)
    
    def _calculate_bollinger_bands(self, prices, period=20, std_dev=2):
        """Calculate Bollinger Bands"""
        ma = prices.rolling(period).mean()
        std = prices.rolling(period).std()
        upper = ma + (std * std_dev)
        lower = ma - (std * std_dev)
        return upper.fillna(prices), lower.fillna(prices)
    
    def get_latest_data(self):
        """
        Main method: Get latest data with all features
        (Like getting everything ready for your super brain)
        """
        print("📊 Getting latest Bitcoin data for analysis...")
        
        # Step 1: Get current price
        current_price = self.get_latest_bitcoin_price()
        if current_price is None:
            return None
        
        # Step 2: Store it in centralized database
        self.store_price_data(current_price)
        
        # Step 3: Get historical data for features from centralized database
        historical_data = self.get_historical_data(hours=200)
        if historical_data is None:
            return None
        
        # Step 4: Create features
        featured_data = self.create_features(historical_data)
        if featured_data is None:
            return None
        
        print("✅ Latest data prepared for signal generation")
        return featured_data
    
    def get_data_summary(self):
        """
        Get summary of available data in centralized database
        """
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Count total records
            count_query = "SELECT COUNT(*) FROM btc_hourly"
            total_records = conn.execute(count_query).fetchone()[0]
            
            # Get date range
            range_query = """
                SELECT MIN(timestamp) as earliest, MAX(timestamp) as latest 
                FROM btc_hourly
            """
            result = conn.execute(range_query).fetchone()
            
            conn.close()
            
            return {
                'total_records': total_records,
                'earliest_date': result[0],
                'latest_date': result[1],
                'database_path': self.db_path,
                'database_size_mb': os.path.getsize(self.db_path) / (1024 * 1024),
                'centralized_location': os.path.dirname(os.path.dirname(self.db_path))
            }
            
        except Exception as e:
            print(f"❌ Failed to get data summary from centralized database: {e}")
            return None
    
    def export_bitcoin_data(self, output_dir=None, days=None):
        """Export Bitcoin data for analysis"""
        if not output_dir:
            # Export to data_&_models/exports/ by default
            data_dir = Config.get_data_models_dir()
            output_dir = os.path.join(data_dir, "exports")
            os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        try:
            conn = sqlite3.connect(self.db_path)
            
            if days:
                # Export specific number of days
                since = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
                df = pd.read_sql_query('''
                    SELECT * FROM btc_hourly 
                    WHERE timestamp >= ? 
                    ORDER BY timestamp
                ''', conn, params=(since,))
                filename = f"bitcoin_data_{days}days_{timestamp}.csv"
            else:
                # Export all data
                df = pd.read_sql_query("SELECT * FROM btc_hourly ORDER BY timestamp", conn)
                filename = f"bitcoin_data_all_{timestamp}.csv"
            
            conn.close()
            
            csv_file = os.path.join(output_dir, filename)
            df.to_csv(csv_file, index=False)
            print(f"✅ Exported {len(df)} Bitcoin records to {csv_file}")
            print(f"📁 Export saved to: {output_dir}")
            
        except Exception as e:
            print(f"❌ Failed to export Bitcoin data: {e}")
    
    def cleanup_old_data(self, days_to_keep=90):
        """Clean up old Bitcoin data from centralized database"""
        try:
            cutoff_date = (datetime.now() - timedelta(days=days_to_keep)).strftime('%Y-%m-%d')
            
            conn = sqlite3.connect(self.db_path)
            
            # Delete old records
            result = conn.execute("DELETE FROM btc_hourly WHERE timestamp < ?", (cutoff_date,))
            deleted_count = result.rowcount
            
            conn.commit()
            conn.close()
            
            if deleted_count > 0:
                print(f"🧹 Cleaned up {deleted_count} old Bitcoin records from centralized database")
                print(f"   Database: {self.db_path}")
            else:
                print(f"✅ No old Bitcoin data to clean up")
            
        except Exception as e:
            print(f"❌ Failed to cleanup old Bitcoin data: {e}")
    
    def print_data_summary(self):
        """Print comprehensive data summary from centralized database"""
        print(f"\n📊 BITCOIN DATA SUMMARY")
        print(f"="*50)
        
        summary = self.get_data_summary()
        if summary:
            print(f"📁 Database Location: {summary['database_path']}")
            print(f"📁 Centralized Directory: {summary['centralized_location']}")
            print(f"💾 Database Size: {summary['database_size_mb']:.2f} MB")
            print(f"📈 Total Records: {summary['total_records']:,}")
            print(f"📅 Date Range: {summary['earliest_date']} to {summary['latest_date']}")
            
            if summary['total_records'] > 0:
                # Calculate coverage
                start_date = pd.to_datetime(summary['earliest_date'])
                end_date = pd.to_datetime(summary['latest_date'])
                coverage_days = (end_date - start_date).days
                print(f"⏰ Coverage: {coverage_days} days")
                
                expected_records = coverage_days * 24  # 24 hours per day
                completeness = (summary['total_records'] / expected_records) * 100 if expected_records > 0 else 0
                print(f"📊 Data Completeness: {completeness:.1f}%")
        else:
            print("❌ Could not get data summary")
        
        print(f"="*50)
    
    def get_database_info(self):
        """Get detailed information about the centralized Bitcoin database"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Get table info
            table_info = conn.execute("PRAGMA table_info(btc_hourly)").fetchall()
            
            # Get record count
            record_count = conn.execute("SELECT COUNT(*) FROM btc_hourly").fetchone()[0]
            
            # Get latest record
            latest_record = conn.execute('''
                SELECT timestamp, close FROM btc_hourly 
                ORDER BY timestamp DESC 
                LIMIT 1
            ''').fetchone()
            
            conn.close()
            
            return {
                'database_path': self.db_path,
                'database_size_mb': os.path.getsize(self.db_path) / (1024 * 1024),
                'table_structure': table_info,
                'record_count': record_count,
                'latest_record': latest_record,
                'centralized_location': os.path.dirname(os.path.dirname(self.db_path))
            }
            
        except Exception as e:
            print(f"Failed to get Bitcoin database info: {e}")
            return None