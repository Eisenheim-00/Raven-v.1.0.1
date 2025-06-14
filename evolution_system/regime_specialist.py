# regime_specialist.py - Raven's Market Mood Specialists
# Different strategies for different market conditions

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

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Optional

# Import Config directly (NOT from core_system module)
from config import Config

class RavenRegimeSpecialist:
    """
    Raven's specialist strategies for different market regimes
    Think of these as different versions of Raven optimized for specific market moods!
    """
    
    def __init__(self):
        # Base parameters from original 60.8% system
        self.base_confidence_threshold = Config.ENHANCED_CONFIDENCE_THRESHOLD
        self.base_best_hours = Config.BEST_HOURS.copy()
        self.base_worst_hours = Config.WORST_HOURS.copy()
        
        # Regime-specific adjustments (learned through testing)
        self.regime_specialists = {
            'WILD_TRENDING': {
                'name': 'Wild Trend Rider',
                'confidence_threshold': 0.10,  # More aggressive in trending volatile markets
                'hour_multiplier': 1.2,        # Boost confidence during good hours
                'volatility_preference': 'high',
                'best_hours': [23, 19, 20, 4, 1, 7, 14, 15],  # Extended hours for volatile markets
                'strategy': 'Ride strong trends in volatile conditions'
            },
            
            'WILD_SIDEWAYS': {
                'name': 'Wild Scalper',
                'confidence_threshold': 0.15,  # More conservative in chaotic markets
                'hour_multiplier': 0.9,        # Reduce confidence in uncertainty
                'volatility_preference': 'high',
                'best_hours': [23, 19, 20, 4],  # Only best hours
                'strategy': 'Quick scalps in high volatility sideways markets'
            },
            
            'CHILL_TRENDING': {
                'name': 'Steady Trend Follower',
                'confidence_threshold': 0.11,  # Moderate confidence in steady trends
                'hour_multiplier': 1.1,        # Slight boost for steady conditions
                'volatility_preference': 'low',
                'best_hours': [23, 19, 20, 4, 1, 7, 8, 9],  # Extended hours for steady trends
                'strategy': 'Follow steady trends with confidence'
            },
            
            'CHILL_SIDEWAYS': {
                'name': 'Patient Hunter',
                'confidence_threshold': 0.13,  # Conservative in boring markets
                'hour_multiplier': 1.0,        # Standard confidence
                'volatility_preference': 'low',
                'best_hours': [23, 19, 20, 4, 1],  # Core best hours only
                'strategy': 'Wait for high-confidence opportunities'
            }
        }
        
        # Performance tracking for each specialist
        self.specialist_performance = {regime: {'trades': 0, 'wins': 0, 'accuracy': 0.0} 
                                     for regime in self.regime_specialists.keys()}
        
        print("🎭 Raven's regime specialists activated!")
        self.print_specialists()
    
    def get_specialist_for_regime(self, regime: str) -> Dict:
        """Get the specialist configuration for current market regime"""
        if regime in self.regime_specialists:
            return self.regime_specialists[regime]
        else:
            # Default to the most conservative specialist for unknown regimes
            return self.regime_specialists['CHILL_SIDEWAYS']
    
    def adjust_signal_for_regime(self, base_signal: Dict, regime: str, market_data: pd.DataFrame) -> Dict:
        """
        Adjust Raven's trading signal based on current market regime
        This is where regime specialists modify the base prediction!
        """
        specialist = self.get_specialist_for_regime(regime)
        
        # Copy base signal
        adjusted_signal = base_signal.copy()
        adjusted_signal['regime'] = regime
        adjusted_signal['specialist'] = specialist['name']
        adjusted_signal['original_confidence'] = base_signal['enhanced_confidence']
        
        # Regime-specific confidence adjustment
        regime_confidence_threshold = specialist['confidence_threshold']
        hour_multiplier = specialist['hour_multiplier']
        
        # Adjust confidence based on regime and time
        current_hour = base_signal['hour']
        is_regime_best_hour = current_hour in specialist['best_hours']
        
        # Calculate regime-adjusted confidence
        base_confidence = base_signal['enhanced_confidence']
        
        if is_regime_best_hour:
            regime_adjusted_confidence = base_confidence * hour_multiplier
        else:
            regime_adjusted_confidence = base_confidence * 0.9  # Slight penalty for non-optimal hours
        
        # Apply regime-specific logic
        if regime == 'WILD_TRENDING':
            # In volatile trending markets, boost confidence if trend aligns with prediction
            trend_strength = abs(market_data['trend_strength'].iloc[-1]) if 'trend_strength' in market_data else 0
            if trend_strength > 0.02:  # Strong trend
                regime_adjusted_confidence *= 1.15
                adjusted_signal['regime_boost'] = 'trend_alignment'
        
        elif regime == 'WILD_SIDEWAYS':
            # In volatile sideways markets, reduce confidence slightly
            regime_adjusted_confidence *= 0.95
            adjusted_signal['regime_boost'] = 'volatility_caution'
        
        elif regime == 'CHILL_TRENDING':
            # In steady trending markets, moderate boost
            regime_adjusted_confidence *= 1.05
            adjusted_signal['regime_boost'] = 'steady_trend'
        
        elif regime == 'CHILL_SIDEWAYS':
            # In boring markets, no adjustment (base confidence)
            adjusted_signal['regime_boost'] = 'none'
        
        # Update signal with regime adjustments
        adjusted_signal['enhanced_confidence'] = regime_adjusted_confidence
        adjusted_signal['regime_threshold'] = regime_confidence_threshold
        
        # Determine if signal meets regime-specific threshold
        if regime_adjusted_confidence >= regime_confidence_threshold:
            if base_signal['trade_signal'] == 'NO_TRADE':
                # Regime specialist might activate a trade that base system rejected
                adjusted_signal['trade_signal'] = "BUY" if base_signal['prediction'] == 'Up' else "SELL"
                adjusted_signal['activation_reason'] = f"Regime specialist ({specialist['name']}) activated trade"
        else:
            if base_signal['trade_signal'] in ['BUY', 'SELL']:
                # Regime specialist might deactivate a trade that base system wanted
                adjusted_signal['trade_signal'] = 'NO_TRADE'
                adjusted_signal['deactivation_reason'] = f"Regime specialist ({specialist['name']}) rejected trade"
        
        # Update confidence reason
        adjusted_signal['confidence_reason'] = f"{specialist['name']}: {regime_adjusted_confidence:.3f} vs {regime_confidence_threshold:.3f}"
        
        return adjusted_signal
    
    def update_specialist_performance(self, regime: str, was_correct: bool):
        """Update performance tracking for regime specialist"""
        if regime in self.specialist_performance:
            self.specialist_performance[regime]['trades'] += 1
            if was_correct:
                self.specialist_performance[regime]['wins'] += 1
            
            # Recalculate accuracy
            trades = self.specialist_performance[regime]['trades']
            wins = self.specialist_performance[regime]['wins']
            self.specialist_performance[regime]['accuracy'] = wins / trades if trades > 0 else 0.0
    
    def get_best_performing_specialist(self) -> Optional[str]:
        """Find which regime specialist is performing best"""
        best_regime = None
        best_accuracy = 0.0
        min_trades = 5  # Need at least 5 trades to be considered
        
        for regime, performance in self.specialist_performance.items():
            if (performance['trades'] >= min_trades and 
                performance['accuracy'] > best_accuracy):
                best_accuracy = performance['accuracy']
                best_regime = regime
        
        return best_regime
    
    def get_regime_specific_performance(self, regime: str) -> Optional[Dict]:
        """Get performance data for specific regime"""
        if regime in self.specialist_performance:
            performance = self.specialist_performance[regime]
            return {
                'regime': regime,
                'total_trades': performance['trades'],
                'accuracy': performance['accuracy'],
                'wins': performance['wins']
            }
        return None
    
    def adapt_specialists_based_on_performance(self):
        """
        Evolve specialist parameters based on their performance
        This is the learning component!
        """
        adaptations = []
        
        for regime, performance in self.specialist_performance.items():
            if performance['trades'] >= 10:  # Need sufficient data
                accuracy = performance['accuracy']
                specialist = self.regime_specialists[regime]
                
                if accuracy >= 0.70:
                    # Performing very well - be more aggressive
                    old_threshold = specialist['confidence_threshold']
                    specialist['confidence_threshold'] = max(0.08, old_threshold - 0.01)
                    adaptations.append(f"{specialist['name']}: Increased aggression ({old_threshold:.3f} → {specialist['confidence_threshold']:.3f})")
                
                elif accuracy <= 0.50:
                    # Performing poorly - be more conservative
                    old_threshold = specialist['confidence_threshold']
                    specialist['confidence_threshold'] = min(0.20, old_threshold + 0.01)
                    adaptations.append(f"{specialist['name']}: Increased caution ({old_threshold:.3f} → {specialist['confidence_threshold']:.3f})")
        
        if adaptations:
            print("🧠 Raven's specialists evolved:")
            for adaptation in adaptations:
                print(f"   {adaptation}")
        
        return adaptations
    
    def get_regime_recommendation(self, current_regime: str) -> Dict:
        """Get trading recommendation based on current regime"""
        specialist = self.get_specialist_for_regime(current_regime)
        performance = self.specialist_performance.get(current_regime, {'accuracy': 0.0, 'trades': 0})
        
        return {
            'regime': current_regime,
            'specialist_name': specialist['name'],
            'strategy': specialist['strategy'],
            'confidence_threshold': specialist['confidence_threshold'],
            'best_hours': specialist['best_hours'],
            'performance': performance,
            'recommendation': self._get_regime_advice(current_regime, performance)
        }
    
    def _get_regime_advice(self, regime: str, performance: Dict) -> str:
        """Generate advice based on regime and performance"""
        if performance['trades'] < 5:
            return "Learning phase - gathering data for this regime"
        
        accuracy = performance['accuracy']
        if accuracy >= 0.65:
            return f"Excellent performance in {regime} - trade with confidence!"
        elif accuracy >= 0.55:
            return f"Good performance in {regime} - standard trading approach"
        else:
            return f"Challenging performance in {regime} - extra caution recommended"
    
    def print_specialists(self):
        """Print information about all regime specialists"""
        print(f"\n🎭 RAVEN'S REGIME SPECIALISTS")
        print(f"="*50)
        
        for regime, specialist in self.regime_specialists.items():
            performance = self.specialist_performance[regime]
            print(f"\n📊 {specialist['name']} ({regime}):")
            print(f"   Strategy: {specialist['strategy']}")
            print(f"   Confidence Threshold: {specialist['confidence_threshold']:.3f}")
            print(f"   Best Hours: {specialist['best_hours']}")
            print(f"   Performance: {performance['wins']}/{performance['trades']} ({performance['accuracy']:.1%})")
        
        print(f"="*50)
    
    def print_performance_summary(self):
        """Print performance summary for all specialists"""
        print(f"\n📈 REGIME SPECIALISTS PERFORMANCE")
        print(f"="*40)
        
        total_trades = sum(p['trades'] for p in self.specialist_performance.values())
        total_wins = sum(p['wins'] for p in self.specialist_performance.values())
        overall_accuracy = total_wins / total_trades if total_trades > 0 else 0.0
        
        print(f"Overall Performance: {total_wins}/{total_trades} ({overall_accuracy:.1%})")
        print(f"")
        
        # Sort by accuracy for display
        sorted_specialists = sorted(
            self.specialist_performance.items(),
            key=lambda x: x[1]['accuracy'],
            reverse=True
        )
        
        for regime, performance in sorted_specialists:
            specialist_name = self.regime_specialists[regime]['name']
            if performance['trades'] > 0:
                print(f"🎭 {specialist_name}:")
                print(f"   {performance['wins']}/{performance['trades']} trades ({performance['accuracy']:.1%})")
            else:
                print(f"🎭 {specialist_name}: No trades yet")
        
        # Identify best performer
        best_regime = self.get_best_performing_specialist()
        if best_regime:
            best_specialist = self.regime_specialists[best_regime]['name']
            best_accuracy = self.specialist_performance[best_regime]['accuracy']
            print(f"\n🏆 Top Performer: {best_specialist} ({best_accuracy:.1%})")
        
        print(f"="*40)