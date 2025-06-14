# evolution_system.py - Raven 2.0 Auto-Phase Evolution Controller
# Automatically progresses through evolution phases for 24/7 operation

import sys
import os

# ============================================================================
# PATH SETUP - Fix Import Issues
# ============================================================================

# Get current directory (evolution_system folder)
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)  # RAVEN V.0.1.3 folder

# Add core_system path for SignalGenerator import
core_system_path = os.path.join(project_root, 'core_system')
sys.path.append(core_system_path)

# ============================================================================
# IMPORTS - Now with fixed paths
# ============================================================================

# Evolution system imports (same folder)
from evolution_tracker import RavenEvolutionTracker
from regime_specialist import RavenRegimeSpecialist
from phase_controller import RavenPhaseController, EvolutionPhase

# Core system imports (core_system folder) - FIXED IMPORTS
from signal_generator import SignalGenerator

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Optional

class RavenEvolutionSystem:
    """
    Raven 2.0 - Auto-Phase Evolution System for 24/7 Operation
    Automatically progresses through phases based on time + performance
    """
    
    def __init__(self, base_signal_generator: SignalGenerator):
        # Core components
        self.base_signal_generator = base_signal_generator
        self.evolution_tracker = RavenEvolutionTracker()
        self.regime_specialist = RavenRegimeSpecialist()
        self.phase_controller = RavenPhaseController()  # NEW: Automatic phase management
        
        # Performance monitoring
        self.consecutive_losses = 0
        self.last_phase_check = datetime.now()
        self.phase_check_interval = timedelta(hours=6)  # Check every 6 hours
        
        # Get initial phase configuration
        self.update_configuration_from_phase()
        
        print("🚀 RAVEN 2.0 AUTO-PHASE EVOLUTION SYSTEM ACTIVATED!")
        print("   🧠 Memory system: Online")
        print("   🎭 Regime specialists: Phase-controlled") 
        print("   📈 Learning engine: Phase-controlled")
        print("   🎮 Phase controller: Auto-progression enabled")
        print("   🛡️ Safety fallbacks: Multi-level")
        
        # Print initial phase status
        self.phase_controller.print_phase_status()
    
    def update_configuration_from_phase(self):
        """Update system configuration based on current phase"""
        config = self.phase_controller.get_phase_configuration()
        
        # Apply phase-specific settings
        self.learning_enabled = config['learning_enabled']
        self.regime_detection_enabled = config['regime_detection_enabled']
        self.regime_specialists_enabled = config['regime_specialists_enabled']
        self.confidence_adaptation_enabled = config['confidence_adaptation_enabled']
        self.advanced_features_enabled = config['advanced_features_enabled']
        self.max_consecutive_losses = config['safety_threshold']
        
        # Update evolution tracker settings
        self.evolution_tracker.confidence_adjustment_factor = config['confidence_adjustment_factor']
        
        print(f"⚙️ Configuration updated for phase: {self.phase_controller.current_phase.value}")
    
    def generate_evolved_signal(self, data_with_features: pd.DataFrame) -> Optional[Dict]:
        """
        Generate evolved signal with automatic phase management
        """
        try:
            # Check for automatic phase progression (every 6 hours)
            if datetime.now() - self.last_phase_check > self.phase_check_interval:
                self.check_automatic_phase_progression()
                self.last_phase_check = datetime.now()
            
            # Step 1: Generate base signal using original 60.8% system
            base_signal = self.base_signal_generator.generate_signal(data_with_features)
            
            if not base_signal:
                return None
            
            # Step 2: Apply phase-appropriate evolution
            evolved_signal = self.apply_phase_evolution(base_signal, data_with_features)
            
            # Step 3: Apply safety checks
            evolved_signal = self._apply_safety_checks(evolved_signal, base_signal)
            
            # Add evolution metadata
            evolved_signal['evolution_version'] = '2.0-auto'
            evolved_signal['current_phase'] = self.phase_controller.current_phase.value
            evolved_signal['base_confidence'] = base_signal['enhanced_confidence']
            evolved_signal['learning_active'] = self.learning_enabled
            
            return evolved_signal
            
        except Exception as e:
            print(f"❌ Evolution signal generation failed: {e}")
            # Fallback to base signal
            if 'base_signal' in locals() and base_signal:
                base_signal['evolution_fallback'] = True
                return base_signal
            return None
    
    def apply_phase_evolution(self, base_signal: Dict, data_with_features: pd.DataFrame) -> Dict:
        """Apply evolution features based on current phase"""
        evolved_signal = base_signal.copy()
        current_phase = self.phase_controller.current_phase
        
        # Phase 1: Basic Learning Only
        if current_phase == EvolutionPhase.PHASE_1_BASIC_LEARNING:
            if self.learning_enabled:
                evolved_signal = self._apply_basic_learning(evolved_signal)
            evolved_signal['evolution_features'] = ['basic_learning']
        
        # Phase 2: Add Regime Detection
        elif current_phase == EvolutionPhase.PHASE_2_REGIME_DETECTION:
            if self.learning_enabled:
                evolved_signal = self._apply_basic_learning(evolved_signal)
            if self.regime_detection_enabled:
                current_regime = self.evolution_tracker.detect_market_regime(data_with_features)
                evolved_signal['regime'] = current_regime
                evolved_signal = self._apply_regime_awareness(evolved_signal, current_regime)
            evolved_signal['evolution_features'] = ['basic_learning', 'regime_detection']
        
        # Phase 3: Add Regime Specialists
        elif current_phase == EvolutionPhase.PHASE_3_REGIME_SPECIALISTS:
            if self.learning_enabled:
                evolved_signal = self._apply_basic_learning(evolved_signal)
            if self.regime_detection_enabled:
                current_regime = self.evolution_tracker.detect_market_regime(data_with_features)
                evolved_signal['regime'] = current_regime
                evolved_signal = self._apply_regime_awareness(evolved_signal, current_regime)
            if self.regime_specialists_enabled:
                current_regime = evolved_signal.get('regime', 'UNKNOWN')
                evolved_signal = self.regime_specialist.adjust_signal_for_regime(
                    evolved_signal, current_regime, data_with_features
                )
            evolved_signal['evolution_features'] = ['basic_learning', 'regime_detection', 'regime_specialists']
        
        # Phase 4: Full Evolution
        elif current_phase == EvolutionPhase.PHASE_4_FULL_EVOLUTION:
            # Apply all features
            evolved_signal = self._apply_full_evolution(evolved_signal, data_with_features)
            evolved_signal['evolution_features'] = ['basic_learning', 'regime_detection', 'regime_specialists', 'advanced_adaptation']
        
        # Phase 5: Advanced Learning
        elif current_phase == EvolutionPhase.PHASE_5_ADVANCED_LEARNING:
            # Apply all features + advanced optimizations
            evolved_signal = self._apply_full_evolution(evolved_signal, data_with_features)
            evolved_signal = self._apply_advanced_features(evolved_signal, data_with_features)
            evolved_signal['evolution_features'] = ['all_features', 'advanced_optimization']
        
        return evolved_signal
    
    def _apply_basic_learning(self, signal: Dict) -> Dict:
        """Apply basic learning adjustments (Phase 1+)"""
        if not self.confidence_adaptation_enabled:
            return signal
        
        try:
            performance = self.evolution_tracker.get_recent_performance()
            
            if performance and performance['total_trades'] >= 5:
                recent_accuracy = performance['accuracy']
                
                # Simple confidence adjustment
                if recent_accuracy >= 0.65:
                    adjustment = 0.01
                    signal['learning_adjustment'] = 'confidence_boost'
                elif recent_accuracy <= 0.50:
                    adjustment = -0.02
                    signal['learning_adjustment'] = 'confidence_reduction'
                else:
                    adjustment = 0
                    signal['learning_adjustment'] = 'none'
                
                if adjustment != 0:
                    original = signal['enhanced_confidence']
                    signal['enhanced_confidence'] = max(0.05, min(0.25, original + adjustment))
                    signal['learning_confidence_delta'] = adjustment
            
            return signal
            
        except Exception as e:
            print(f"⚠️ Basic learning failed: {e}")
            return signal
    
    def _apply_regime_awareness(self, signal: Dict, regime: str) -> Dict:
        """Apply regime awareness without full specialists (Phase 2)"""
        # Simple regime-based confidence adjustment
        regime_multipliers = {
            'WILD_TRENDING': 1.1,    # Slightly more confident in volatile trends
            'WILD_SIDEWAYS': 0.9,    # Less confident in chaotic markets
            'CHILL_TRENDING': 1.05,  # Slightly more confident in steady trends
            'CHILL_SIDEWAYS': 0.95   # Slightly less confident in boring markets
        }
        
        multiplier = regime_multipliers.get(regime, 1.0)
        signal['enhanced_confidence'] *= multiplier
        signal['regime_adjustment'] = f"{regime}: {multiplier}x"
        
        return signal
    
    def _apply_full_evolution(self, signal: Dict, data_with_features: pd.DataFrame) -> Dict:
        """Apply full evolution system (Phase 4+)"""
        # All features enabled
        if self.learning_enabled:
            signal = self._apply_basic_learning(signal)
        
        if self.regime_detection_enabled:
            current_regime = self.evolution_tracker.detect_market_regime(data_with_features)
            signal['regime'] = current_regime
            
            if self.regime_specialists_enabled:
                signal = self.regime_specialist.adjust_signal_for_regime(
                    signal, current_regime, data_with_features
                )
        
        return signal
    
    def _apply_advanced_features(self, signal: Dict, data_with_features: pd.DataFrame) -> Dict:
        """Apply advanced features (Phase 5 only)"""
        # Advanced confidence scaling based on recent regime performance
        regime = signal.get('regime', 'UNKNOWN')
        regime_perf = self.evolution_tracker.get_regime_specific_performance(regime)
        
        if regime_perf and regime_perf['total_trades'] >= 10:
            regime_accuracy = regime_perf['accuracy']
            
            if regime_accuracy >= 0.70:
                # This regime is performing excellently - boost confidence
                signal['enhanced_confidence'] *= 1.15
                signal['advanced_boost'] = 'excellent_regime_performance'
            elif regime_accuracy <= 0.45:
                # This regime is struggling - reduce confidence
                signal['enhanced_confidence'] *= 0.85
                signal['advanced_boost'] = 'poor_regime_performance'
        
        return signal
    
    def _apply_safety_checks(self, evolved_signal: Dict, base_signal: Dict) -> Dict:
        """Apply safety checks with phase-aware thresholds"""
        # Use phase-appropriate safety threshold
        max_losses = self.max_consecutive_losses
        
        # Safety Check 1: Consecutive losses
        if self.consecutive_losses >= max_losses:
            print(f"🛡️ Phase {self.phase_controller.current_phase.value} safety: {self.consecutive_losses} consecutive losses")
            evolved_signal['trade_signal'] = 'NO_TRADE'
            evolved_signal['safety_fallback'] = f'consecutive_losses_phase_{self.phase_controller.current_phase.value}'
            return evolved_signal
        
        # Safety Check 2: Extreme confidence changes (phase-aware)
        base_conf = base_signal['enhanced_confidence']
        evolved_conf = evolved_signal['enhanced_confidence']
        
        max_multiplier = 2.0 if self.phase_controller.current_phase in [
            EvolutionPhase.PHASE_4_FULL_EVOLUTION, 
            EvolutionPhase.PHASE_5_ADVANCED_LEARNING
        ] else 1.5  # More conservative in early phases
        
        if evolved_conf > base_conf * max_multiplier:
            evolved_signal['enhanced_confidence'] = base_conf * max_multiplier
            evolved_signal['safety_adjustment'] = f'confidence_limited_{max_multiplier}x'
        
        return evolved_signal
    
    def check_automatic_phase_progression(self):
        """Check and execute automatic phase progression"""
        try:
            # Check if progression criteria are met
            should_progress, next_phase = self.phase_controller.check_phase_progression(
                self.evolution_tracker, self.regime_specialist
            )
            
            if should_progress and next_phase:
                # Get performance data for progression record
                performance = self.evolution_tracker.get_recent_performance(hours=24*30)
                
                # Execute progression
                success = self.phase_controller.progress_to_next_phase(next_phase, performance or {})
                
                if success:
                    # Update configuration for new phase
                    self.update_configuration_from_phase()
                    print(f"🎉 Automatic progression to {next_phase.value} completed!")
                    
                    # Log the phase change
                    self.evolution_tracker.store_regime_data(
                        f"PHASE_CHANGE_{next_phase.value}",
                        0.0, 0.0, 1.0  # Dummy values for regime storage
                    )
            
            # Check for emergency fallback
            performance = self.evolution_tracker.get_recent_performance()
            if performance and self.phase_controller.check_emergency_fallback(performance):
                print("🚨 Executing emergency phase fallback...")
                if self.phase_controller.emergency_fallback():
                    self.update_configuration_from_phase()
                    self.consecutive_losses = 0  # Reset on fallback
                    
        except Exception as e:
            print(f"⚠️ Automatic phase progression check failed: {e}")
    
    def record_trade_outcome(self, signal: Dict, actual_return: float, market_data: pd.DataFrame):
        """Record trade outcome with phase tracking"""
        try:
            # Record in evolution tracker
            self.evolution_tracker.record_trade_outcome(signal, actual_return, market_data)
            
            # Update regime specialist performance (if enabled)
            if self.regime_specialists_enabled:
                is_correct = (signal['prediction'] == 'Up' and actual_return >= 0) or \
                            (signal['prediction'] == 'Down' and actual_return < 0)
                
                regime = signal.get('regime', 'UNKNOWN')
                self.regime_specialist.update_specialist_performance(regime, is_correct)
                
                # Track consecutive losses for safety
                if not is_correct:
                    self.consecutive_losses += 1
                else:
                    self.consecutive_losses = 0
            
            # Trigger learning cycle
            if self.learning_enabled:
                self._trigger_learning_cycle()
            
            phase = self.phase_controller.current_phase.value
            outcome = '✅' if (signal['prediction'] == 'Up' and actual_return >= 0) or \
                            (signal['prediction'] == 'Down' and actual_return < 0) else '❌'
            
            print(f"📊 Trade outcome recorded: {outcome} (Phase: {phase})")
            
        except Exception as e:
            print(f"⚠️ Failed to record trade outcome: {e}")
    
    def _trigger_learning_cycle(self):
        """Phase-aware learning cycle"""
        try:
            # Only trigger regime specialist adaptation in appropriate phases
            if self.regime_specialists_enabled:
                total_trades = sum(p['trades'] for p in self.regime_specialist.specialist_performance.values())
                
                if total_trades > 0 and total_trades % 20 == 0:
                    print("🧠 Triggering phase-aware learning cycle...")
                    
                    # Adapt regime specialists
                    adaptations = self.regime_specialist.adapt_specialists_based_on_performance()
                    
                    # Learn from recent trades
                    learning_result = self.evolution_tracker.learn_from_recent_trades()
                    
                    if learning_result:
                        print(f"📈 Learning result: {learning_result['reason']}")
                        
        except Exception as e:
            print(f"⚠️ Learning cycle failed: {e}")
    
    def get_evolution_status(self) -> Dict:
        """Get comprehensive evolution status with phase info"""
        try:
            # Get phase status
            phase_status = self.phase_controller.get_phase_status()
            
            # Get performance data
            performance = self.evolution_tracker.get_recent_performance()
            
            # Get regime performance (if enabled)
            best_regime = None
            if self.regime_specialists_enabled:
                best_regime = self.regime_specialist.get_best_performing_specialist()
            
            return {
                'phase_info': phase_status,
                'learning_enabled': self.learning_enabled,
                'regime_specialists_enabled': self.regime_specialists_enabled,
                'consecutive_losses': self.consecutive_losses,
                'max_consecutive_losses': self.max_consecutive_losses,
                'recent_performance': performance,
                'best_regime': best_regime,
                'features_active': phase_status['configuration'].get('description', 'Unknown')
            }
            
        except Exception as e:
            print(f"⚠️ Failed to get evolution status: {e}")
            return {'error': str(e)}
    
    def print_evolution_summary(self):
        """Print comprehensive evolution summary with phase details"""
        print(f"\n🚀 RAVEN 2.0 AUTO-PHASE EVOLUTION SUMMARY")
        print(f"="*70)
        
        # Phase status
        self.phase_controller.print_phase_status(
            self.evolution_tracker.get_recent_performance()
        )
        
        # Overall status
        status = self.get_evolution_status()
        
        print(f"\n⚙️ Current Configuration:")
        print(f"   Learning: {'ACTIVE' if self.learning_enabled else 'DISABLED'}")
        print(f"   Regime Detection: {'ACTIVE' if self.regime_detection_enabled else 'DISABLED'}")
        print(f"   Regime Specialists: {'ACTIVE' if self.regime_specialists_enabled else 'DISABLED'}")
        print(f"   Advanced Features: {'ACTIVE' if self.advanced_features_enabled else 'DISABLED'}")
        
        print(f"\n🛡️ Safety Status:")
        print(f"   Consecutive Losses: {self.consecutive_losses}/{self.max_consecutive_losses}")
        print(f"   Safety Level: Phase {self.phase_controller.current_phase.value}")
        
        if status['recent_performance']:
            perf = status['recent_performance']
            print(f"\n📈 Recent Performance:")
            print(f"   Trades: {perf['total_trades']}")
            print(f"   Accuracy: {perf['accuracy']:.1%}")
            print(f"   Recent Trend: {perf['recent_trend']:.1%}")
        
        # Regime performance (if available)
        if self.regime_specialists_enabled and status['best_regime']:
            print(f"\n🏆 Best Performing Regime: {status['best_regime']}")
            self.regime_specialist.print_performance_summary()
        
        print(f"="*70)
    
    def force_phase_progression(self, target_phase: EvolutionPhase) -> bool:
        """Manually force progression to specific phase (for testing)"""
        try:
            print(f"⚡ Manual phase progression: {self.phase_controller.current_phase.value} → {target_phase.value}")
            
            # Record manual progression
            performance = self.evolution_tracker.get_recent_performance() or {}
            
            old_phase = self.phase_controller.current_phase
            self.phase_controller.current_phase = target_phase
            self.phase_controller.phase_start_time = datetime.now()
            self.phase_controller.save_current_phase()
            
            # Update configuration
            self.update_configuration_from_phase()
            
            print(f"✅ Manual progression completed!")
            return True
            
        except Exception as e:
            print(f"❌ Manual phase progression failed: {e}")
            return False