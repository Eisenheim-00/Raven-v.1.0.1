# evolution_tracker.py - Raven's Learning Memory System
# Updated to use centralized database paths in data_&_models/databases/

import sys
import os

# ============================================================================
# PATH SETUP - Fix Import Issues
# ============================================================================

# Get current directory (evolution_system folder)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)  # RAVEN V.0.1.3 folder

# Add core_system path for Config import
core_system_path = os.path.join(project_root, 'core_system')
sys.path.append(core_system_path)

# ============================================================================
# IMPORTS - Now with fixed paths
# ============================================================================

import pandas as pd
import numpy as np
import sqlite3
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional

# Import Config for centralized database paths
from config import Config

class RavenEvolutionTracker:
    """
    Raven's memory and learning system
    Think of this as Raven's brain that remembers everything and learns!
    Updated to use centralized database paths.
    """
    
    def __init__(self, db_path=None):
        # Use centralized database path from Config
        if db_path is None:
            self.db_path = Config.get_evolution_db_path()
        else:
            self.db_path = db_path
            
        self.setup_evolution_database()
        
        # Performance tracking
        self.recent_trades_window = 20  # Look at last 20 trades for adaptation
        self.confidence_adjustment_factor = 0.02  # How much to adjust confidence
        
        print("🧠 Raven's evolution system activated!")
        print(f"   📁 Evolution database: {self.db_path}")
        print(f"   📁 Centralized in: {os.path.dirname(os.path.dirname(self.db_path))}")
    
    def setup_evolution_database(self):
        """Setup centralized database to track Raven's learning"""
        try:
            # Ensure database directory exists
            db_dir = os.path.dirname(self.db_path)
            os.makedirs(db_dir, exist_ok=True)
            
            conn = sqlite3.connect(self.db_path)
            
            # Table for trade outcomes and learning
            conn.execute('''
                CREATE TABLE IF NOT EXISTS trade_outcomes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    prediction INTEGER,
                    actual_outcome INTEGER,
                    confidence REAL,
                    market_regime TEXT,
                    btc_price REAL,
                    volatility REAL,
                    hour INTEGER,
                    day_of_week INTEGER,
                    trade_return REAL,
                    correct INTEGER,
                    created_at TEXT
                )
            ''')
            
            # Table for regime detection
            conn.execute('''
                CREATE TABLE IF NOT EXISTS market_regimes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    regime_type TEXT,
                    volatility_level TEXT,
                    trend_strength REAL,
                    volume_ratio REAL,
                    price_momentum REAL,
                    created_at TEXT
                )
            ''')
            
            # Table for confidence adjustments
            conn.execute('''
                CREATE TABLE IF NOT EXISTS confidence_adjustments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    old_threshold REAL,
                    new_threshold REAL,
                    reason TEXT,
                    recent_accuracy REAL,
                    created_at TEXT
                )
            ''')
            
            conn.commit()
            conn.close()
            print("✅ Centralized evolution database initialized")
            
        except Exception as e:
            print(f"❌ Evolution database setup failed: {e}")
    
    def detect_market_regime(self, current_data):
        """
        Detect current market regime (SIMPLE version for 2-week implementation)
        """
        try:
            # Calculate market indicators
            recent_returns = current_data['returns'].tail(24)  # Last 24 hours
            volatility = recent_returns.std()
            price_change_24h = (current_data['close'].iloc[-1] / current_data['close'].iloc[-24] - 1) if len(current_data) >= 24 else 0
            volume_ratio = current_data['volume_ratio'].iloc[-1] if 'volume_ratio' in current_data else 1.0
            
            # Simple regime classification
            high_vol_threshold = 0.03  # 3% hourly volatility
            trend_threshold = 0.05     # 5% daily price change
            
            if volatility > high_vol_threshold:
                if abs(price_change_24h) > trend_threshold:
                    regime = "WILD_TRENDING"  # High volatility + strong trend
                else:
                    regime = "WILD_SIDEWAYS"  # High volatility + no clear trend
            else:
                if abs(price_change_24h) > trend_threshold:
                    regime = "CHILL_TRENDING"  # Low volatility + trend
                else:
                    regime = "CHILL_SIDEWAYS"  # Low volatility + sideways
            
            # Store regime data in centralized database
            self.store_regime_data(regime, volatility, price_change_24h, volume_ratio)
            
            return regime
            
        except Exception as e:
            print(f"⚠️ Regime detection failed: {e}")
            return "UNKNOWN"
    
    def store_regime_data(self, regime, volatility, trend_strength, volume_ratio):
        """Store detected market regime in centralized database"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            conn.execute('''
                INSERT INTO market_regimes 
                (timestamp, regime_type, volatility_level, trend_strength, volume_ratio, price_momentum, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                regime,
                "HIGH" if volatility > 0.03 else "LOW",
                trend_strength,
                volume_ratio,
                trend_strength,  # Simple momentum = trend strength
                datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            print(f"⚠️ Failed to store regime data in centralized database: {e}")
    
    def record_trade_outcome(self, signal, actual_return, current_data):
        """
        Record the outcome of a trade for learning in centralized database
        """
        try:
            # Determine if prediction was correct
            predicted_direction = 1 if signal['prediction'] == 'Up' else 0
            actual_direction = 1 if actual_return >= 0 else 0
            is_correct = predicted_direction == actual_direction
            
            # Detect market regime at time of trade
            regime = self.detect_market_regime(current_data)
            
            # Store trade outcome in centralized database
            conn = sqlite3.connect(self.db_path)
            
            conn.execute('''
                INSERT INTO trade_outcomes 
                (timestamp, prediction, actual_outcome, confidence, market_regime, btc_price, 
                 volatility, hour, day_of_week, trade_return, correct, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                signal['timestamp'].strftime('%Y-%m-%d %H:%M:%S'),
                predicted_direction,
                actual_direction,
                signal['enhanced_confidence'],
                regime,
                signal['current_price'],
                current_data['volatility_lag'].iloc[-1] if 'volatility_lag' in current_data else 0,
                signal['hour'],
                signal['day_of_week'],
                actual_return,
                1 if is_correct else 0,
                datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            ))
            
            conn.commit()
            conn.close()
            
            print(f"📊 Recorded trade outcome: {'✅ Correct' if is_correct else '❌ Wrong'} ({regime}) -> Centralized DB")
            
            # Trigger learning from this outcome
            self.learn_from_recent_trades()
            
        except Exception as e:
            print(f"⚠️ Failed to record trade outcome in centralized database: {e}")
    
    def get_recent_performance(self, hours=48):
        """Get Raven's recent performance for adaptation from centralized database"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Get recent trades
            since = (datetime.now() - timedelta(hours=hours)).strftime('%Y-%m-%d %H:%M:%S')
            
            df = pd.read_sql_query('''
                SELECT * FROM trade_outcomes 
                WHERE timestamp >= ?
                ORDER BY timestamp DESC
                LIMIT ?
            ''', conn, params=(since, self.recent_trades_window))
            
            conn.close()
            
            if len(df) == 0:
                return None
            
            # Calculate performance metrics
            performance = {
                'total_trades': len(df),
                'correct_trades': df['correct'].sum(),
                'accuracy': df['correct'].mean(),
                'avg_confidence': df['confidence'].mean(),
                'regime_performance': df.groupby('market_regime')['correct'].mean().to_dict(),
                'recent_trend': df.tail(10)['correct'].mean() if len(df) >= 10 else df['correct'].mean()
            }
            
            return performance
            
        except Exception as e:
            print(f"⚠️ Failed to get recent performance from centralized database: {e}")
            return None
    
    def learn_from_recent_trades(self):
        """
        Raven learns from recent trades and adjusts his confidence
        """
        performance = self.get_recent_performance()
        
        if not performance or performance['total_trades'] < 5:
            return None  # Need at least 5 trades to learn
        
        current_accuracy = performance['accuracy']
        recent_trend = performance['recent_trend']
        
        print(f"🧠 Raven learning: {performance['total_trades']} trades, {current_accuracy:.1%} accuracy")
        
        # Decide if confidence threshold should be adjusted
        adjustment = None
        reason = ""
        
        if current_accuracy >= 0.70 and recent_trend >= 0.70:
            # Performing very well - be more aggressive
            adjustment = self.confidence_adjustment_factor
            reason = "Excellent performance - increasing aggression"
            
        elif current_accuracy <= 0.50 and recent_trend <= 0.50:
            # Performing poorly - be more conservative
            adjustment = -self.confidence_adjustment_factor
            reason = "Poor performance - increasing caution"
            
        elif recent_trend > current_accuracy + 0.15:
            # Recent improvement - cautiously increase aggression
            adjustment = self.confidence_adjustment_factor * 0.5
            reason = "Recent improvement - slight aggression increase"
            
        elif recent_trend < current_accuracy - 0.15:
            # Recent decline - increase caution
            adjustment = -self.confidence_adjustment_factor * 0.5
            reason = "Recent decline - slight caution increase"
        
        # Store adjustment in centralized database if made
        if adjustment is not None:
            try:
                conn = sqlite3.connect(self.db_path)
                conn.execute('''
                    INSERT INTO confidence_adjustments 
                    (timestamp, old_threshold, new_threshold, reason, recent_accuracy, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    Config.ENHANCED_CONFIDENCE_THRESHOLD,
                    Config.ENHANCED_CONFIDENCE_THRESHOLD + adjustment,
                    reason,
                    current_accuracy,
                    datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                ))
                conn.commit()
                conn.close()
            except Exception as e:
                print(f"⚠️ Failed to store confidence adjustment: {e}")
        
        return {
            'performance': performance,
            'suggested_adjustment': adjustment,
            'reason': reason
        }
    
    def get_regime_specific_performance(self, regime):
        """Get performance for specific market regime from centralized database"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            df = pd.read_sql_query('''
                SELECT * FROM trade_outcomes 
                WHERE market_regime = ?
                ORDER BY timestamp DESC
                LIMIT 50
            ''', conn, params=(regime,))
            
            conn.close()
            
            if len(df) == 0:
                return None
            
            return {
                'regime': regime,
                'total_trades': len(df),
                'accuracy': df['correct'].mean(),
                'avg_confidence': df['confidence'].mean(),
                'avg_return': df['trade_return'].mean()
            }
            
        except Exception as e:
            print(f"⚠️ Failed to get regime performance from centralized database: {e}")
            return None
    
    def get_evolution_summary(self):
        """Get overall summary of Raven's evolution from centralized database"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Overall stats
            total_trades = pd.read_sql_query(
                "SELECT COUNT(*) as count FROM trade_outcomes", conn
            ).iloc[0]['count']
            
            if total_trades == 0:
                return None
            
            # Performance by regime
            regime_performance = pd.read_sql_query('''
                SELECT market_regime, 
                       COUNT(*) as trades,
                       AVG(correct) as accuracy,
                       AVG(confidence) as avg_confidence
                FROM trade_outcomes 
                GROUP BY market_regime
            ''', conn)
            
            # Recent adjustments
            recent_adjustments = pd.read_sql_query('''
                SELECT * FROM confidence_adjustments 
                ORDER BY timestamp DESC 
                LIMIT 5
            ''', conn)
            
            conn.close()
            
            return {
                'total_trades': total_trades,
                'regime_performance': regime_performance.to_dict('records'),
                'recent_adjustments': recent_adjustments.to_dict('records') if len(recent_adjustments) > 0 else [],
                'database_location': self.db_path
            }
            
        except Exception as e:
            print(f"⚠️ Failed to get evolution summary from centralized database: {e}")
            return None
    
    def print_learning_status(self):
        """Print Raven's current learning status from centralized database"""
        print(f"\n🧠 RAVEN'S LEARNING STATUS")
        print(f"📁 Data source: {self.db_path}")
        print(f"="*40)
        
        performance = self.get_recent_performance()
        if performance:
            print(f"📊 Recent Performance ({performance['total_trades']} trades):")
            print(f"   Accuracy: {performance['accuracy']:.1%}")
            print(f"   Recent Trend: {performance['recent_trend']:.1%}")
            print(f"   Avg Confidence: {performance['avg_confidence']:.3f}")
            
            if performance['regime_performance']:
                print(f"\n🎭 Performance by Market Regime:")
                for regime, accuracy in performance['regime_performance'].items():
                    print(f"   {regime}: {accuracy:.1%}")
        
        summary = self.get_evolution_summary()
        if summary:
            print(f"\n📈 Total Evolution:")
            print(f"   Total Trades Learned From: {summary['total_trades']}")
            print(f"   Regimes Discovered: {len(summary['regime_performance'])}")
            print(f"   Database: {summary['database_location']}")
        
        print(f"="*40)
    
    def export_evolution_data(self, output_dir=None):
        """Export evolution data for analysis"""
        if not output_dir:
            # Export to data_&_models/exports/ by default
            data_dir = Config.get_data_models_dir()
            output_dir = os.path.join(data_dir, "exports")
            os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Export each table
            tables = ['trade_outcomes', 'market_regimes', 'confidence_adjustments']
            
            for table in tables:
                df = pd.read_sql_query(f"SELECT * FROM {table}", conn)
                csv_file = os.path.join(output_dir, f"raven_evolution_{table}_{timestamp}.csv")
                df.to_csv(csv_file, index=False)
                print(f"✅ Exported {len(df)} evolution records to {csv_file}")
            
            conn.close()
            print(f"📁 All evolution exports saved to: {output_dir}")
            
        except Exception as e:
            print(f"❌ Failed to export evolution data: {e}")
    
    def get_database_info(self):
        """Get information about the centralized evolution database"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Get table sizes
            tables = ['trade_outcomes', 'market_regimes', 'confidence_adjustments']
            table_info = {}
            
            for table in tables:
                count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                table_info[table] = count
            
            conn.close()
            
            return {
                'database_path': self.db_path,
                'database_size_mb': os.path.getsize(self.db_path) / (1024 * 1024),
                'table_record_counts': table_info,
                'centralized_location': os.path.dirname(os.path.dirname(self.db_path))
            }
            
        except Exception as e:
            print(f"Failed to get evolution database info: {e}")
            return None