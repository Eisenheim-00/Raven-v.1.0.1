# main.py - Raven 2.0 Auto-Phase Evolution for 24/7 Operation
# Automatically progresses through evolution phases based on performance + time

import time
import schedule
import warnings
from datetime import datetime, timedelta
import logging
import sys
import os

# ============================================================================
# PATH SETUP - Fix Import Issues
# ============================================================================

# Get the current directory (core_system folder)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)  # RAVEN V.0.1.3 folder

# Add all necessary paths to Python path
sys.path.append(current_dir)  # core_system folder
sys.path.append(os.path.join(project_root, 'evolution_system'))  # evolution_system folder
sys.path.append(os.path.join(project_root, 'data_&_models'))     # data_&_models folder

print(f"🔧 Python paths configured:")
print(f"   Core system: {current_dir}")
print(f"   Evolution system: {os.path.join(project_root, 'evolution_system')}")
print(f"   Data & models: {os.path.join(project_root, 'data_&_models')}")

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
warnings.filterwarnings('ignore', category=FutureWarning, module='pandas')
warnings.filterwarnings('ignore', category=UserWarning, module='xgboost')

# ============================================================================
# IMPORTS - Now with fixed paths
# ============================================================================

# Core system imports (same folder)
from config import Config
from data_manager import DataManager
from signal_generator import SignalGenerator
from alpaca_trader import AlpacaTrader
from logger import RavenLogger

# Evolution system imports (evolution_system folder)
from evolution_system import RavenEvolutionSystem
from phase_controller import EvolutionPhase

class RavenAutoEvolutionBot:
    """
    Raven 2.0 - Fully Automated Evolution Bot for 24/7 Operation
    Automatically progresses through phases to achieve >60% win rate!
    """
    
    def __init__(self):
        print("🐦‍⬛ Starting Raven 2.0 - Auto-Evolution Trading Bot...")
        print("🎯 Target: >60% Win Rate Through Automatic Evolution")
        print("⏰ 24/7 Operation • Auto-Phase Progression • Emergency Fallbacks")
        print("="*80)
        
        # Initialize core components
        self.config = Config()
        self.logger = RavenLogger()
        self.data_manager = DataManager()
        
        # Initialize auto-phase evolution system
        base_signal_generator = SignalGenerator(self.config.MODEL_PATH)
        self.evolution = RavenEvolutionSystem(base_signal_generator)
        
        self.alpaca_trader = AlpacaTrader(
            self.config.APCA_API_KEY_ID,
            self.config.APCA_API_SECRET_KEY
        )
        
        # Performance tracking
        self.trades_today = 0
        self.trades_this_session = 0
        self.bot_start_time = datetime.now()
        self.last_status_report = datetime.now()
        self.status_report_interval = timedelta(hours=8)  # Report every 8 hours
        
        # Trade outcome tracking for learning
        self.pending_trades = {}  # Track trades waiting for outcome
        
        print("✅ Raven 2.0 Auto-Evolution System Ready!")
        print("🧠 Evolution will automatically progress through phases:")
        print("   Phase 1: Basic Learning (2+ days, 20+ trades, 55%+ accuracy)")
        print("   Phase 2: Regime Detection (4+ days, 40+ trades, 57%+ accuracy)")
        print("   Phase 3: Regime Specialists (7+ days, 75+ trades, 58%+ accuracy)")
        print("   Phase 4: Full Evolution (14+ days, 150+ trades, 59%+ accuracy)")
        print("   Phase 5: Advanced Learning (30+ days, 300+ trades, 60%+ accuracy)")
        
        self.logger.log_info("Raven 2.0 Auto-Evolution started successfully")
    
    def run_trading_cycle(self):
        """
        Enhanced trading cycle with automatic evolution and outcome tracking
        """
        try:
            current_time = datetime.now()
            print(f"\n⏰ Trading Cycle: {current_time.strftime('%Y-%m-%d %H:%M:%S')} UTC")
            
            # Check for pending trade outcomes first
            self.check_pending_trade_outcomes()
            
            # Step 1: Get latest Bitcoin data
            print("📊 Getting latest Bitcoin data...")
            latest_data = self.data_manager.get_latest_data()
            
            if latest_data is None:
                print("❌ No data available, skipping cycle")
                return
            
            # Step 2: Generate evolved signal with auto-phase management
            print("🧠 Generating signal with auto-phase evolution...")
            signal = self.evolution.generate_evolved_signal(latest_data)
            
            if signal is None:
                print("⚠️ No signal generated")
                return
            
            # Step 3: Display comprehensive signal info
            self.display_evolution_signal_info(signal)
            
            # Step 4: Execute trade if signal indicates
            if signal['trade_signal'] in ['BUY', 'SELL']:
                print(f"🎯 Executing {signal['trade_signal']} signal...")
                trade_result = self.alpaca_trader.execute_trade(signal)
                
                if trade_result:
                    self.trades_today += 1
                    self.trades_this_session += 1
                    
                    # Store trade for outcome tracking
                    trade_id = trade_result['order_id']
                    self.pending_trades[trade_id] = {
                        'signal': signal,
                        'trade_result': trade_result,
                        'timestamp': current_time,
                        'btc_price_at_trade': signal['current_price']
                    }
                    
                    self.logger.log_trade(signal, trade_result)
                    print(f"✅ Trade executed! Total today: {self.trades_today}, Session: {self.trades_this_session}")
                    
                else:
                    print("❌ Trade execution failed")
            else:
                print("⏸️ No trade signal - waiting for optimal opportunity")
                
                # Record non-trading decision for learning (simplified)
                if signal.get('trade_signal') == 'NO_TRADE':
                    self.evolution.record_trade_outcome(signal, 0.0, latest_data)
            
            # Step 5: Periodic status reporting
            if current_time - self.last_status_report > self.status_report_interval:
                self.print_comprehensive_status_report()
                self.last_status_report = current_time
            
            # Step 6: Log cycle completion
            self.logger.log_cycle(signal)
            print("✅ Trading cycle completed")
            
        except Exception as e:
            error_msg = f"Error in trading cycle: {str(e)}"
            print(f"❌ {error_msg}")
            self.logger.log_error(error_msg)
    
    def check_pending_trade_outcomes(self):
        """
        Check outcomes of pending trades and record them for learning
        This enables real learning from actual trade results!
        """
        if not self.pending_trades:
            return
        
        current_time = datetime.now()
        completed_trades = []
        
        for trade_id, trade_info in self.pending_trades.items():
            trade_time = trade_info['timestamp']
            time_elapsed = current_time - trade_time
            
            # Check trade outcome after 1+ hours
            if time_elapsed >= timedelta(hours=1):
                try:
                    # Get current Bitcoin price for return calculation
                    latest_data = self.data_manager.get_latest_data()
                    if latest_data is not None:
                        current_btc_price = latest_data['close'].iloc[-1]
                        original_price = trade_info['btc_price_at_trade']
                        
                        # Calculate actual return
                        actual_return = (current_btc_price - original_price) / original_price
                        
                        # Adjust return based on trade direction
                        trade_side = trade_info['trade_result']['side']
                        if trade_side.upper() == 'SELL':
                            actual_return = -actual_return  # Invert for short positions
                        
                        # Record outcome for learning
                        self.evolution.record_trade_outcome(
                            trade_info['signal'], 
                            actual_return, 
                            latest_data
                        )
                        
                        # Log the learning
                        outcome = "Profit" if actual_return > 0 else "Loss"
                        print(f"📚 Learning from trade {trade_id}: {outcome} ({actual_return:.3%})")
                        
                        completed_trades.append(trade_id)
                        
                except Exception as e:
                    print(f"⚠️ Failed to process trade outcome for {trade_id}: {e}")
                    completed_trades.append(trade_id)  # Remove problematic trades
        
        # Remove completed trades
        for trade_id in completed_trades:
            del self.pending_trades[trade_id]
    
    def display_evolution_signal_info(self, signal):
        """Display comprehensive signal information with evolution details"""
        print(f"\n📈 RAVEN 2.0 EVOLUTION SIGNAL:")
        print(f"   🎮 Phase: {signal.get('current_phase', 'Unknown')}")
        print(f"   🎯 Prediction: {signal['prediction']}")
        
        # Show confidence evolution
        base_conf = signal.get('base_confidence', signal['enhanced_confidence'])
        evolved_conf = signal['enhanced_confidence']
        
        if base_conf != evolved_conf:
            change_symbol = "↑" if evolved_conf > base_conf else "↓"
            print(f"   🧠 Confidence: {base_conf:.3f} {change_symbol} {evolved_conf:.3f}")
        else:
            print(f"   🧠 Confidence: {evolved_conf:.3f}")
        
        print(f"   🎯 Trade Signal: {signal['trade_signal']}")
        
        # Evolution-specific details
        if 'regime' in signal:
            print(f"   🎭 Market Regime: {signal['regime']}")
        if 'specialist' in signal:
            print(f"   🎭 Specialist: {signal['specialist']}")
        if 'evolution_features' in signal:
            features = ", ".join(signal['evolution_features'])
            print(f"   ⚙️ Active Features: {features}")
        if 'learning_adjustment' in signal:
            print(f"   🧠 Learning: {signal['learning_adjustment']}")
        
        print(f"   ⏰ Time Analysis: {signal['time_reason']} (Hour {signal['hour']} UTC)")
        
        # Confidence level indication
        if evolved_conf >= 0.15:
            print(f"   🌟 VERY HIGH CONFIDENCE - Excellent opportunity!")
        elif evolved_conf >= 0.12:
            print(f"   ✅ HIGH CONFIDENCE - Good trading opportunity")
        elif evolved_conf >= 0.10:
            print(f"   🟡 MEDIUM CONFIDENCE - Moderate opportunity")
        else:
            print(f"   🔴 LOW CONFIDENCE - Waiting for better setup")
    
    def print_comprehensive_status_report(self):
        """Print detailed status report every 8 hours"""
        print(f"\n" + "="*80)
        print(f"📊 RAVEN 2.0 - 8 HOUR STATUS REPORT")
        print(f"⏰ Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} UTC")
        print(f"="*80)
        
        # Basic stats
        uptime = datetime.now() - self.bot_start_time
        print(f"🤖 Bot Statistics:")
        print(f"   Uptime: {uptime}")
        print(f"   Trades Today: {self.trades_today}")
        print(f"   Trades This Session: {self.trades_this_session}")
        print(f"   Pending Trade Outcomes: {len(self.pending_trades)}")
        
        # Portfolio status
        portfolio = self.alpaca_trader.get_portfolio_status()
        if portfolio:
            print(f"\n💼 Portfolio Status:")
            print(f"   Value: ${portfolio['portfolio_value']:,.2f}")
            print(f"   Today's P&L: ${portfolio['todays_pnl']:,.2f}")
            
            if portfolio['current_position']:
                pos = portfolio['current_position']
                print(f"   Position: {pos['qty']} BTC (${pos['market_value']:,.2f})")
                print(f"   Unrealized P&L: ${pos['unrealized_pl']:,.2f}")
        
        # Evolution system status
        print(f"\n🧠 Evolution System Status:")
        self.evolution.print_evolution_summary()
        
        print(f"="*80)
    
    def daily_summary(self):
        """Enhanced daily summary with evolution metrics"""
        print(f"\n📊 DAILY SUMMARY - {datetime.now().strftime('%Y-%m-%d')}")
        print(f"="*60)
        
        print(f"📈 Trading Statistics:")
        print(f"   Trades executed: {self.trades_today}")
        print(f"   Session trades: {self.trades_this_session}")
        print(f"   Bot uptime: {datetime.now() - self.bot_start_time}")
        
        # Portfolio summary
        portfolio = self.alpaca_trader.get_portfolio_status()
        if portfolio:
            print(f"\n💰 Financial Performance:")
            print(f"   Portfolio value: ${portfolio['portfolio_value']:,.2f}")
            print(f"   Today's P&L: ${portfolio['todays_pnl']:,.2f}")
        
        # Evolution progress
        print(f"\n🧠 Evolution Progress:")
        status = self.evolution.get_evolution_status()
        phase_info = status.get('phase_info', {})
        
        print(f"   Current Phase: {phase_info.get('current_phase', 'Unknown')}")
        print(f"   Days in Phase: {phase_info.get('days_in_phase', 0):.1f}")
        
        if status.get('recent_performance'):
            perf = status['recent_performance']
            print(f"   Recent Accuracy: {perf['accuracy']:.1%}")
            print(f"   Recent Trend: {perf['recent_trend']:.1%}")
            
            # Check progress toward 60% target
            if perf['accuracy'] >= 0.60:
                print(f"   🎉 TARGET ACHIEVED: >60% win rate!")
            else:
                remaining = 0.60 - perf['accuracy']
                print(f"   🎯 Progress to 60%: {remaining:.1%} remaining")
        
        # Reset daily counter
        self.trades_today = 0
        
        print(f"="*60)
    
    def run_continuous(self):
        """
        Run Raven 2.0 continuously with auto-phase evolution
        """
        print(f"\n🚀 RAVEN 2.0 AUTO-EVOLUTION ACTIVATED!")
        print(f"🎯 Target: Achieve >60% win rate through automatic phase progression")
        print(f"⏰ 24/7 Operation: Trading every hour, evolving continuously")
        print(f"🧠 Auto-Evolution: Progresses through phases based on performance + time")
        print(f"🛡️ Safety Systems: Multiple fallback mechanisms active")
        print(f"\n🔄 Raven 2.0 is now hunting, learning, and evolving...")
        print(f"💡 Status reports every 8 hours • Daily summaries at midnight")
        print(f"⚡ Press Ctrl+C to stop\n")
        
        # Schedule automated operations
        schedule.every().hour.at(":00").do(self.run_trading_cycle)
        schedule.every().day.at("00:00").do(self.daily_summary)
        
        # Run immediate cycle for testing
        print("🚀 Running initial cycle...")
        self.run_trading_cycle()
        
        # Print initial status
        self.print_comprehensive_status_report()
        
        # Main operation loop
        try:
            while True:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
                
        except KeyboardInterrupt:
            print(f"\n🛑 Raven 2.0 stopped by user")
            self.logger.log_info("Raven 2.0 stopped by user")
            
            # Final status report
            print(f"\n📊 FINAL SESSION SUMMARY:")
            print(f"   Session Duration: {datetime.now() - self.bot_start_time}")
            print(f"   Total Trades: {self.trades_this_session}")
            print(f"   Pending Outcomes: {len(self.pending_trades)}")
            
            # Complete evolution summary
            self.evolution.print_evolution_summary()
            
        except Exception as e:
            print(f"\n❌ Raven 2.0 crashed: {e}")
            self.logger.log_error(f"Raven 2.0 crashed: {str(e)}")
    
    # Manual control methods for advanced users
    def force_phase(self, phase_name: str):
        """Manually force progression to specific phase (for testing)"""
        try:
            phase = EvolutionPhase(phase_name)
            success = self.evolution.force_phase_progression(phase)
            if success:
                print(f"⚡ Manual progression to {phase_name} completed!")
            return success
        except Exception as e:
            print(f"❌ Manual phase progression failed: {e}")
            return False
    
    def get_evolution_report(self):
        """Get comprehensive evolution report"""
        return self.evolution.get_evolution_status()

def main():
    """
    Launch Raven 2.0 Auto-Evolution System
    """
    print("🐦‍⬛ RAVEN 2.0 - AUTO-EVOLUTION TRADING BOT")
    print("🎯 Automatic Phase Progression • 24/7 Operation • >60% Win Rate Target")
    print("💎 Paper Trading • Emergency Fallbacks • Comprehensive Learning")
    print("="*80)
    
    try:
        # Validate configuration first
        if not Config.validate_settings():
            print("\n❌ Configuration validation failed!")
            print("💡 Please fix the configuration errors before starting Raven.")
            return
        
        # Create and launch Raven 2.0
        raven = RavenAutoEvolutionBot()
        raven.run_continuous()
        
    except Exception as e:
        print(f"❌ Failed to start Raven 2.0: {e}")
        import traceback
        print(f"🔍 Full error details:")
        traceback.print_exc()

if __name__ == "__main__":
    main()

# =============================================================================
# AUTOMATIC PHASE PROGRESSION SYSTEM
# =============================================================================

"""
🎮 RAVEN 2.0 AUTO-PHASE EVOLUTION:

Your bot will automatically progress through these phases:

📅 TIMELINE:
   Day 0-2:    Phase 1 (Basic Learning)
   Day 2-4:    Phase 2 (Regime Detection) 
   Day 4-7:    Phase 3 (Regime Specialists)
   Day 7-14:   Phase 4 (Full Evolution)
   Day 14-30:  Phase 5 (Advanced Learning)
   Day 30+:    Maintain optimal performance

🎯 PROGRESSION CRITERIA:
Each phase requires BOTH time elapsed AND performance thresholds:
   - Minimum days in current phase
   - Minimum number of trades completed
   - Minimum accuracy achievement

🛡️ SAFETY SYSTEMS:
   - Emergency fallback if performance drops significantly
   - Phase-appropriate safety thresholds
   - Multiple fallback mechanisms
   - Comprehensive logging and monitoring

🚀 EXPECTED RESULTS:
   Week 1:  Base performance (60.8% accuracy)
   Week 2:  Learning improvements (+2-5%)
   Week 3:  Regime specialists online (+3-7%)
   Week 4:  Full evolution active (+5-10%)
   Month 2: TARGET ACHIEVED (>60% win rate)

💡 NO MANUAL INTERVENTION REQUIRED:
The system automatically:
   ✅ Progresses through phases
   ✅ Adapts to market conditions
   ✅ Falls back if needed
   ✅ Reports progress every 8 hours
   ✅ Learns from every trade outcome

🎯 PERFECT FOR 24/7 OPERATION!
"""