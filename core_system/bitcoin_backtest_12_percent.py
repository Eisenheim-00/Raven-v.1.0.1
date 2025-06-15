# bitcoin_backtest_12_percent.py - Fixed for Yahoo Finance Limits
# NO DATA LEAKAGE - Walk-forward testing only

import sys
import os
import pandas as pd
import numpy as np
import yfinance as yf
import joblib
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Add paths to import your modules
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(os.path.join(project_root, 'core_system'))

# Import your actual components
from config import Config

class BitcoinBacktester:
    """
    Backtest the 12% confidence strategy on Bitcoin data
    STRICT NO FUTURE SNOOPING - only uses past data at each decision point
    """
    
    def __init__(self):
        # Strategy parameters (current Raven settings)
        self.confidence_threshold = 0.12  # 12% confidence threshold
        self.best_hours = [23, 19, 20, 4, 1, 7]  # UTC optimal hours
        self.worst_hours = [5, 18, 6, 21, 8, 15]  # Avoid these hours
        self.position_size_usd = 1000  # $1000 per trade
        
        # Load your actual trained model
        self.load_model()
        
        # Tracking
        self.trades = []
        self.portfolio_value = 100000  # Start with $100k
        self.cash = 100000
        self.btc_position = 0
        self.current_position_value = 0
        
        print("🧪 Bitcoin 12% Confidence Backtest Initialized")
        print(f"   📊 Confidence Threshold: {self.confidence_threshold:.1%}")
        print(f"   ⏰ Best Hours: {self.best_hours}")
        print(f"   💰 Position Size: ${self.position_size_usd:,}")
    
    def load_model(self):
        """Load your actual trained model"""
        try:
            # Try to find your model
            model_paths = [
                os.path.join(project_root, "data_&_models", "models", "optimized_bitcoin_model.pkl"),
                "../data_&_models/models/optimized_bitcoin_model.pkl"
            ]
            
            for model_path in model_paths:
                if os.path.exists(model_path):
                    self.model_data = joblib.load(model_path)
                    print(f"✅ Loaded model: {os.path.basename(model_path)}")
                    print(f"   Features: {len(self.model_data['features'])}")
                    print(f"   Model: {type(self.model_data['model']).__name__}")
                    return
            
            print("❌ Model not found - using dummy model for demo")
            self.model_data = None
            
        except Exception as e:
            print(f"⚠️ Model loading failed: {e}")
            self.model_data = None
    
    def download_bitcoin_data(self, days=600):
        """
        Download Bitcoin data (max 600 days due to Yahoo Finance limits)
        """
        print(f"\n📥 Downloading {days} days of Bitcoin data...")
        
        # Calculate date range (Yahoo Finance limits hourly data to ~730 days)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=min(days, 700))  # Stay within limits
        
        print(f"   Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        
        try:
            # Try different methods to get Bitcoin data
            methods = [
                ("BTC-USD", "Yahoo Finance"),
                ("BTCUSD=X", "Yahoo Finance Alt"),
            ]
            
            for symbol, source in methods:
                try:
                    print(f"   Trying {source} ({symbol})...")
                    ticker = yf.Ticker(symbol)
                    df = ticker.history(
                        start=start_date.strftime('%Y-%m-%d'),
                        end=end_date.strftime('%Y-%m-%d'),
                        interval="1h",
                        auto_adjust=True,
                        prepost=False
                    )
                    
                    if not df.empty and len(df) > 1000:  # Need reasonable amount of data
                        print(f"✅ Downloaded {len(df):,} hours of Bitcoin data from {source}")
                        print(f"   Price range: ${df['Close'].min():,.0f} - ${df['Close'].max():,.0f}")
                        return df
                    
                except Exception as e:
                    print(f"   ❌ {source} failed: {e}")
                    continue
            
            # If all methods fail, create synthetic data for demo
            print("⚠️ Using synthetic Bitcoin data for demonstration")
            return self.create_synthetic_data(days)
            
        except Exception as e:
            print(f"❌ All data sources failed: {e}")
            return self.create_synthetic_data(days)
    
    def create_synthetic_data(self, days):
        """Create synthetic Bitcoin data for testing"""
        print("🧪 Creating synthetic Bitcoin data...")
        
        # Create hourly timestamps
        end_date = datetime.now().replace(minute=0, second=0, microsecond=0)
        start_date = end_date - timedelta(days=days)
        timestamps = pd.date_range(start=start_date, end=end_date, freq='H')
        
        # Create realistic Bitcoin price movements
        np.random.seed(42)  # For reproducible results
        returns = np.random.normal(0.0001, 0.02, len(timestamps))  # Small hourly returns with volatility
        
        # Start price
        start_price = 30000
        prices = [start_price]
        
        # Generate prices with random walk + trend
        for i, ret in enumerate(returns[1:]):
            # Add some trend and volatility clustering
            trend = 0.00005 if i % 1000 < 500 else -0.00005  # Alternating trends
            volatility = 0.015 if abs(returns[i-1]) > 0.01 else 0.008  # Volatility clustering
            
            new_return = ret * volatility + trend
            new_price = prices[-1] * (1 + new_return)
            prices.append(max(new_price, 1000))  # Don't let price go too low
        
        # Create OHLCV data
        df = pd.DataFrame({
            'Open': prices,
            'High': [p * (1 + abs(np.random.normal(0, 0.005))) for p in prices],
            'Low': [p * (1 - abs(np.random.normal(0, 0.005))) for p in prices],
            'Close': prices,
            'Volume': np.random.exponential(1000, len(prices))
        }, index=timestamps)
        
        # Ensure High >= Open,Close and Low <= Open,Close
        df['High'] = df[['Open', 'Close', 'High']].max(axis=1)
        df['Low'] = df[['Open', 'Close', 'Low']].min(axis=1)
        
        print(f"✅ Created {len(df):,} hours of synthetic data")
        return df
    
    def create_features(self, df, current_idx):
        """
        Create features using ONLY past data (no future snooping)
        """
        # Only use data up to current_idx (strict walk-forward)
        past_data = df.iloc[:current_idx+1].copy()
        
        if len(past_data) < 200:  # Need enough history for features
            return None
        
        # Calculate returns
        past_data['returns'] = past_data['Close'].pct_change()
        
        # Create features using ONLY past data
        features = {}
        
        # 1. Price vs Moving Average features
        for window in [6, 12, 24, 48]:
            if len(past_data) > window:
                ma = past_data['Close'].rolling(window).mean().iloc[-1]
                features[f'price_vs_ma_{window}'] = (past_data['Close'].iloc[-2] / ma) - 1 if ma > 0 else 0
        
        # 2. Lagged returns (using past data only)
        for lag in [1, 2, 3, 6, 12, 24]:
            if len(past_data) > lag:
                features[f'returns_lag_{lag}'] = past_data['returns'].iloc[-(lag+1)]
        
        # 3. Returns statistics
        for window in [6, 12, 24]:
            if len(past_data) > window:
                recent_returns = past_data['returns'].tail(window)
                features[f'returns_mean_{window}'] = recent_returns.mean()
                features[f'returns_std_{window}'] = recent_returns.std()
        
        # 4. Momentum features
        for lag in [6, 12, 24]:
            if len(past_data) > lag + 1:
                features[f'momentum_{lag}'] = (past_data['Close'].iloc[-2] / past_data['Close'].iloc[-(lag+2)]) - 1
        
        # 5. Volume features
        if len(past_data) > 24:
            volume_ma = past_data['Volume'].rolling(24).mean().iloc[-1]
            features['volume_ratio'] = past_data['Volume'].iloc[-2] / volume_ma if volume_ma > 0 else 1
            features['volume_log'] = np.log(past_data['Volume'].iloc[-2] + 1)
        
        # 6. Technical indicators
        if len(past_data) > 14:
            # RSI
            delta = past_data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            features['rsi_lag'] = rsi.iloc[-2] if not pd.isna(rsi.iloc[-2]) else 50
        
        # 7. Volatility
        if len(past_data) > 24:
            features['volatility_lag'] = past_data['returns'].tail(24).std()
        
        # 8. Market structure
        if len(past_data) > 2:
            features['gap'] = (past_data['Open'].iloc[-1] / past_data['Close'].iloc[-2]) - 1
            features['trend_strength'] = abs(features.get('price_vs_ma_24', 0))
        
        # 9. Direction lags
        for lag in [1, 2, 3, 6]:
            if len(past_data) > lag:
                features[f'direction_lag_{lag}'] = 1 if features.get(f'returns_lag_{lag}', 0) >= 0 else 0
        
        # 10. Time features
        current_time = past_data.index[-1]
        features['hour_sin'] = np.sin(2 * np.pi * current_time.hour / 24)
        features['hour_cos'] = np.cos(2 * np.pi * current_time.hour / 24)
        
        # 11. Market regime
        if len(past_data) > 168:  # 1 week
            volatility = past_data['returns'].tail(168).std()
            vol_q75 = past_data['returns'].rolling(168).quantile(0.75).iloc[-1]
            features['high_vol_regime'] = 1 if volatility > vol_q75 else 0
        
        # Fill any missing values
        for key, value in features.items():
            if pd.isna(value) or np.isinf(value):
                features[key] = 0
        
        return features
    
    def generate_signal(self, df, current_idx):
        """
        Generate trading signal using past data only (no future snooping)
        """
        current_time = df.index[current_idx]
        current_price = df['Close'].iloc[current_idx]
        
        # Create features using only past data
        features = self.create_features(df, current_idx)
        if not features:
            return None
        
        # Check if we have the model
        if not self.model_data:
            # Use dummy prediction for demo
            confidence = np.random.uniform(0.05, 0.25)
            prediction = np.random.choice([0, 1])
        else:
            # Use actual model prediction
            try:
                model_features = self.model_data['features']
                feature_vector = [features.get(feat, 0) for feat in model_features]
                
                # Scale features
                scaler = self.model_data['scaler']
                feature_vector_scaled = scaler.transform([feature_vector])
                
                # Get prediction
                probabilities = self.model_data['model'].predict_proba(feature_vector_scaled)[0]
                prediction = self.model_data['model'].predict(feature_vector_scaled)[0]
                confidence = abs(probabilities[1] - 0.5)  # Distance from 50/50
                
            except Exception as e:
                confidence = 0.05
                prediction = 0
        
        # Check trading conditions (using Raven's logic)
        hour = current_time.hour
        
        # Time-based filtering
        if hour in self.worst_hours:
            trade_signal = "NO_TRADE"
            reason = f"worst_hour_{hour}"
        elif confidence >= self.confidence_threshold:
            if hour in self.best_hours:
                trade_signal = "BUY" if prediction == 1 else "SELL"
                reason = f"high_confidence_best_hour_{hour}"
            else:
                trade_signal = "BUY" if prediction == 1 else "SELL"  
                reason = f"high_confidence_neutral_hour_{hour}"
        else:
            trade_signal = "NO_TRADE"
            reason = f"low_confidence_{confidence:.3f}"
        
        return {
            'timestamp': current_time,
            'price': current_price,
            'prediction': 'Up' if prediction == 1 else 'Down',
            'confidence': confidence,
            'trade_signal': trade_signal,
            'reason': reason,
            'hour': hour
        }
    
    def execute_backtest_trade(self, signal, next_price):
        """
        Execute backtest trade and calculate returns
        """
        if signal['trade_signal'] == 'NO_TRADE':
            return None
        
        current_price = signal['price']
        
        # Calculate position size
        if signal['trade_signal'] == 'BUY':
            if self.cash >= self.position_size_usd:
                btc_amount = self.position_size_usd / current_price
                self.cash -= self.position_size_usd
                self.btc_position += btc_amount
                
                # Calculate return after 1 hour
                trade_return = (next_price - current_price) / current_price
                
                trade = {
                    'timestamp': signal['timestamp'],
                    'side': 'BUY',
                    'price': current_price,
                    'next_price': next_price,
                    'confidence': signal['confidence'],
                    'trade_return': trade_return,
                    'reason': signal['reason'],
                    'hour': signal['hour']
                }
                
                self.trades.append(trade)
                return trade
        
        elif signal['trade_signal'] == 'SELL':
            if self.btc_position > 0:
                # Sell some BTC
                btc_to_sell = min(self.btc_position, self.position_size_usd / current_price)
                self.btc_position -= btc_to_sell
                self.cash += btc_to_sell * current_price
                
                # For SELL, we benefit if price goes DOWN
                trade_return = -(next_price - current_price) / current_price
                
                trade = {
                    'timestamp': signal['timestamp'],
                    'side': 'SELL',
                    'price': current_price,
                    'next_price': next_price,
                    'confidence': signal['confidence'],
                    'trade_return': trade_return,
                    'reason': signal['reason'],
                    'hour': signal['hour']
                }
                
                self.trades.append(trade)
                return trade
        
        return None
    
    def run_backtest(self, days=600):
        """
        Run the full backtest with strict no-future-snooping
        """
        print(f"\n🚀 Starting {days}-day backtest...")
        print("🛡️ STRICT NO FUTURE SNOOPING - Walk-forward only")
        
        # Download data
        df = self.download_bitcoin_data(days)
        
        print(f"\n📊 Running backtest on {len(df):,} hours of data...")
        signals_generated = 0
        trades_executed = 0
        
        # Progress tracking
        total_steps = len(df) - 201
        progress_milestones = [0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
        next_milestone = 0
        
        # Walk forward through each hour (except last one)
        for i in range(200, len(df) - 1):  # Start after enough history, end before last
            # Show progress at milestones only
            progress = (i - 200) / total_steps
            if next_milestone < len(progress_milestones) and progress >= progress_milestones[next_milestone]:
                print(f"   Progress: {progress_milestones[next_milestone]:.0%} - {trades_executed:,} trades executed")
                next_milestone += 1
            
            # Generate signal using only past data
            signal = self.generate_signal(df, i)
            if signal:
                signals_generated += 1
                
                # Execute trade using next hour's price (simulating 1-hour holding)
                next_price = df['Close'].iloc[i + 1]
                trade = self.execute_backtest_trade(signal, next_price)
                
                if trade:
                    trades_executed += 1
        
        print(f"✅ Backtest completed!")
        print(f"   Signals generated: {signals_generated:,}")
        print(f"   Trades executed: {trades_executed:,}")
        
        return self.analyze_results(df)
    
    def analyze_results(self, df):
        """
        Analyze backtest results
        """
        if not self.trades:
            print("❌ No trades executed in backtest")
            return None
        
        trades_df = pd.DataFrame(self.trades)
        
        # Calculate metrics
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['trade_return'] > 0])
        losing_trades = len(trades_df[trades_df['trade_return'] <= 0])
        
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        avg_return = trades_df['trade_return'].mean()
        total_return = trades_df['trade_return'].sum()
        
        # Risk metrics
        returns = trades_df['trade_return'].values
        sharpe_ratio = (np.mean(returns) / np.std(returns)) * np.sqrt(8760) if np.std(returns) > 0 else 0  # Annualized
        max_drawdown = self.calculate_max_drawdown(returns)
        
        # Trading frequency
        start_date = trades_df['timestamp'].min()
        end_date = trades_df['timestamp'].max()
        days_traded = (end_date - start_date).days
        trades_per_month = (total_trades / days_traded) * 30 if days_traded > 0 else 0
        
        results = {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'avg_return_per_trade': avg_return,
            'total_return': total_return,
            'annualized_return': avg_return * trades_per_month * 12,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'trades_per_month': trades_per_month,
            'avg_confidence': trades_df['confidence'].mean(),
            'start_date': start_date,
            'end_date': end_date,
            'btc_start_price': df['Close'].iloc[200],
            'btc_end_price': df['Close'].iloc[-1]
        }
        
        return results, trades_df
    
    def calculate_max_drawdown(self, returns):
        """Calculate maximum drawdown"""
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        return abs(drawdown.min())
    
    def print_results(self, results, trades_df):
        """Print comprehensive backtest results"""
        print(f"\n" + "="*60)
        print(f"📊 BITCOIN 12% CONFIDENCE STRATEGY BACKTEST RESULTS")
        print(f"="*60)
        
        print(f"\n📈 TRADING PERFORMANCE:")
        print(f"   Total Trades: {results['total_trades']:,}")
        print(f"   Winning Trades: {results['winning_trades']:,} ({results['win_rate']:.1%})")
        print(f"   Losing Trades: {results['losing_trades']:,}")
        print(f"   Win Rate: {results['win_rate']:.1%}")
        
        print(f"\n💰 RETURNS:")
        print(f"   Average Return per Trade: {results['avg_return_per_trade']:.2%}")
        print(f"   Total Strategy Return: {results['total_return']:.1%}")
        print(f"   Annualized Return: {results['annualized_return']:.1%}")
        
        print(f"\n📊 RISK METRICS:")
        print(f"   Sharpe Ratio: {results['sharpe_ratio']:.2f}")
        print(f"   Max Drawdown: {results['max_drawdown']:.1%}")
        print(f"   Average Confidence: {results['avg_confidence']:.1%}")
        
        print(f"\n⏰ TRADING FREQUENCY:")
        print(f"   Trades per Month: {results['trades_per_month']:.1f}")
        print(f"   Period: {results['start_date'].strftime('%Y-%m-%d')} to {results['end_date'].strftime('%Y-%m-%d')}")
        
        print(f"\n🎯 BENCHMARK COMPARISON:")
        btc_return = (results['btc_end_price'] / results['btc_start_price']) - 1
        print(f"   Bitcoin Buy & Hold: {btc_return:.1%}")
        print(f"   Strategy vs B&H: {results['total_return'] - btc_return:.1%}")
        
        # Trading by hour analysis (top performers only)
        print(f"\n⏰ TOP PERFORMING HOURS:")
        hour_stats = trades_df.groupby('hour').agg({
            'trade_return': ['count', 'mean']
        })
        
        hour_performance = []
        for hour in sorted(trades_df['hour'].unique()):
            hour_data = trades_df[trades_df['hour'] == hour]
            if len(hour_data) >= 5:  # Only show hours with decent trade count
                hour_trades = len(hour_data)
                hour_win_rate = len(hour_data[hour_data['trade_return'] > 0]) / hour_trades
                hour_avg_return = hour_data['trade_return'].mean()
                
                hour_performance.append((hour, hour_trades, hour_win_rate, hour_avg_return))
        
        # Sort by average return and show top 8
        hour_performance.sort(key=lambda x: x[3], reverse=True)
        for hour, trades, win_rate, avg_return in hour_performance[:8]:
            is_best = "⭐" if hour in self.best_hours else ""
            is_worst = "❌" if hour in self.worst_hours else ""
            print(f"   Hour {hour:02d}: {trades:3d} trades, {win_rate:.1%} win rate, {avg_return:+.2%} avg {is_best}{is_worst}")
        
        print(f"\n🎯 STRATEGY ASSESSMENT:")
        win_rate = results['win_rate']
        trades_per_month = results['trades_per_month']
        
        if win_rate >= 0.60 and trades_per_month >= 15:
            print(f"✅ EXCELLENT: {win_rate:.1%} win rate, {trades_per_month:.1f} trades/month")
            print(f"💡 12% confidence strategy is performing very well!")
        elif win_rate >= 0.55 and trades_per_month >= 10:
            print(f"✅ GOOD: {win_rate:.1%} win rate, {trades_per_month:.1f} trades/month")
            print(f"💡 12% confidence strategy is solid")
        elif win_rate >= 0.50:
            print(f"🟡 MARGINAL: {win_rate:.1%} win rate, {trades_per_month:.1f} trades/month")
            print(f"💡 Consider testing 95% confidence strategy")
        else:
            print(f"❌ POOR: {win_rate:.1%} win rate, {trades_per_month:.1f} trades/month")
            print(f"💡 DEFINITELY switch back to 95% confidence strategy!")
        
        print(f"="*60)

def main():
    """Run the backtest"""
    print("🧪 BITCOIN 12% CONFIDENCE STRATEGY BACKTEST")
    print("🛡️ NO DATA LEAKAGE - Walk-forward testing only")
    print("="*60)
    
    # Initialize backtester
    backtester = BitcoinBacktester()
    
    # Run backtest (reduced to 600 days due to Yahoo Finance limits)
    results = backtester.run_backtest(days=600)
    
    if results:
        results_dict, trades_df = results
        backtester.print_results(results_dict, trades_df)
        
        # Save results
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        trades_df.to_csv(f'backtest_12_percent_{timestamp}.csv', index=False)
        print(f"\n💾 Results saved to: backtest_12_percent_{timestamp}.csv")

if __name__ == "__main__":
    main()