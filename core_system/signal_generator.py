# signal_generator.py - 60.8% Accuracy Signal Generator
# This is your super brain that predicts Bitcoin direction with 60.8% accuracy!

import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from config import Config

class SignalGenerator:
    """
    Generates trading signals using your validated 60.8% accuracy model
    This is like having a crystal ball that's right 6 out of 10 times!
    """
    
    def __init__(self, model_path):
        self.model_path = model_path
        self.model_data = None
        
        # Validated parameters from your analysis
        self.best_hours = Config.BEST_HOURS      # [23,19,20,4,1,7] UTC
        self.worst_hours = Config.WORST_HOURS    # [5,18,6,21,8,15] UTC  
        self.best_days = Config.BEST_DAYS        # [5,3,4] Sat,Thu,Fri
        self.confidence_threshold = Config.ENHANCED_CONFIDENCE_THRESHOLD  # 0.12
        
        self.load_model()
    
    def load_model(self):
        """
        Load your trained 60.8% accuracy model
        (Like loading your super smart brain)
        """
        try:
            # Suppress XGBoost version warnings
            import warnings
            warnings.filterwarnings('ignore', category=UserWarning, module='pickle')
            warnings.filterwarnings('ignore', category=UserWarning, module='xgboost')
            
            self.model_data = joblib.load(self.model_path)
            print(f"✅ Loaded 60.8% accuracy model")
            print(f"   Features: {len(self.model_data['features'])}")
            print(f"   Model type: {type(self.model_data['model']).__name__}")
            return True
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            return False
    
    def is_optimal_trading_time(self, timestamp):
        """
        Check if it's a good time to trade based on your analysis
        (Like checking if it's the right time to play your favorite game)
        """
        hour = timestamp.hour
        day_of_week = timestamp.dayofweek
        
        # Check against worst hours first
        if hour in self.worst_hours:
            return False, f"worst_hour_{hour}"
        
        # Check for best combinations
        is_best_hour = hour in self.best_hours
        is_best_day = day_of_week in self.best_days
        
        if is_best_hour and is_best_day:
            return True, f"optimal_time_h{hour}_d{day_of_week}"
        elif is_best_hour:
            return True, f"good_hour_{hour}"
        elif is_best_day:
            return True, f"good_day_{day_of_week}"
        else:
            return False, f"suboptimal_time_h{hour}_d{day_of_week}"
    
    def calculate_enhanced_confidence(self, probabilities, timestamps, volatility, trend_strength):
        """
        Calculate enhanced confidence using your discovered multipliers
        (Like making your crystal ball even more accurate)
        """
        # Base confidence (how far from 50/50)
        base_confidence = np.abs(probabilities - 0.5)
        
        # Time-based multipliers from your analysis
        hour_multipliers = {
            23: 1.3, 19: 1.3, 20: 1.3, 4: 1.3, 1: 1.3, 7: 1.3,  # Best hours
            5: 0.7, 18: 0.7, 6: 0.7, 21: 0.7, 8: 0.7, 15: 0.7   # Worst hours
        }
        
        # Get multiplier for current hour
        if hasattr(timestamps, 'hour'):
            hour = timestamps.hour
        else:
            hour = timestamps.iloc[-1].hour if hasattr(timestamps, 'iloc') else timestamps.hour
            
        time_multiplier = hour_multipliers.get(hour, 1.0)
        
        # Day-of-week multipliers
        dow_multipliers = {5: 1.2, 3: 1.1, 4: 1.1}  # Sat, Thu, Fri
        if hasattr(timestamps, 'dayofweek'):
            dow = timestamps.dayofweek
        else:
            dow = timestamps.iloc[-1].dayofweek if hasattr(timestamps, 'iloc') else timestamps.dayofweek
            
        dow_multiplier = dow_multipliers.get(dow, 0.95)
        
        # Volatility multiplier (medium volatility is best)
        if hasattr(volatility, 'quantile'):
            vol_q25 = volatility.quantile(0.25)
            vol_q75 = volatility.quantile(0.75)
            current_vol = volatility.iloc[-1] if hasattr(volatility, 'iloc') else volatility
        else:
            # Single value
            vol_q25, vol_q75 = 0.01, 0.03  # Default quantiles
            current_vol = volatility
            
        if vol_q25 < current_vol < vol_q75:
            vol_multiplier = 1.1  # Medium volatility
        elif current_vol >= vol_q75:
            vol_multiplier = 1.05  # High volatility
        else:
            vol_multiplier = 0.9  # Low volatility
        
        # Trend strength multiplier
        if hasattr(trend_strength, 'iloc'):
            current_trend = trend_strength.iloc[-1]
        else:
            current_trend = trend_strength
            
        trend_multiplier = 1.1 if current_trend > 0.02 else 0.95
        
        # Combined enhanced confidence
        enhanced_confidence = (base_confidence * time_multiplier * dow_multiplier * 
                             vol_multiplier * trend_multiplier)
        
        return enhanced_confidence
    
    def generate_signal(self, data_with_features):
        """
        Generate trading signal using your 60.8% accuracy system
        (This is where the magic happens!)
        """
        if not self.model_data:
            print("❌ No model loaded")
            return None
        
        if data_with_features is None or len(data_with_features) == 0:
            print("❌ No data provided")
            return None
        
        try:
            # Get current timestamp
            current_timestamp = data_with_features.index[-1]
            
            # Check if it's optimal trading time
            should_trade, time_reason = self.is_optimal_trading_time(current_timestamp)
            
            # Get model components
            model = self.model_data['model']
            scaler = self.model_data['scaler']
            features = self.model_data['features']
            
            # Prepare features for prediction (convert to numpy to avoid feature name conflicts)
            latest_features = data_with_features[features].tail(1).fillna(0)
            
            # Scale features and convert to numpy array (models were trained without feature names)
            features_scaled = scaler.transform(latest_features)
            
            # Get base prediction using numpy array (avoids sklearn warnings)
            probabilities = model.predict_proba(features_scaled)[0]
            probability_up = probabilities[1]  # Probability of going up
            base_prediction = model.predict(features_scaled)[0]
            
            # Calculate enhanced confidence
            enhanced_confidence = self.calculate_enhanced_confidence(
                np.array([probability_up]),
                current_timestamp,
                data_with_features['volatility_lag'].tail(5),
                data_with_features['trend_strength'].tail(1)
            )
            
            if hasattr(enhanced_confidence, '__len__'):
                enhanced_confidence = enhanced_confidence[0]
            
            # Determine trade signal using super-optimized logic
            if not should_trade:
                trade_signal = "NO_TRADE"
                confidence_reason = f"Suboptimal time: {time_reason}"
            elif enhanced_confidence >= self.confidence_threshold:
                trade_signal = "BUY" if base_prediction == 1 else "SELL"
                confidence_reason = f"High confidence: {enhanced_confidence:.3f} >= {self.confidence_threshold}"
            else:
                trade_signal = "NO_TRADE"
                confidence_reason = f"Low confidence: {enhanced_confidence:.3f} < {self.confidence_threshold}"
            
            # Create comprehensive signal
            signal = {
                'timestamp': current_timestamp,
                'prediction': "Up" if base_prediction == 1 else "Down",
                'probability': probability_up,
                'enhanced_confidence': enhanced_confidence,
                'trade_signal': trade_signal,
                'should_trade_time': should_trade,
                'time_reason': time_reason,
                'confidence_reason': confidence_reason,
                'hour': current_timestamp.hour,
                'day_of_week': current_timestamp.dayofweek,
                'current_price': data_with_features['close'].iloc[-1],
                
                # Additional context for logging
                'base_prediction': base_prediction,
                'volatility': data_with_features['volatility_lag'].iloc[-1],
                'trend_strength': data_with_features['trend_strength'].iloc[-1],
                'rsi': data_with_features['rsi_lag'].iloc[-1],
                'is_best_hour': current_timestamp.hour in self.best_hours,
                'is_best_day': current_timestamp.dayofweek in self.best_days,
                'is_worst_hour': current_timestamp.hour in self.worst_hours
            }
            
            return signal
            
        except Exception as e:
            print(f"❌ Signal generation failed: {e}")
            return None
    
    def get_signal_summary(self, signal):
        """
        Create a nice summary of the signal
        (Like explaining what your crystal ball is telling you)
        """
        if not signal:
            return "No signal generated"
        
        summary = []
        summary.append(f"🎯 TRADING SIGNAL SUMMARY")
        summary.append(f"   Time: {signal['timestamp'].strftime('%Y-%m-%d %H:%M')} UTC")
        summary.append(f"   Current Price: ${signal['current_price']:,.2f}")
        summary.append(f"   Prediction: {signal['prediction']} ({signal['probability']:.1%} confidence)")
        summary.append(f"   Enhanced Confidence: {signal['enhanced_confidence']:.3f}")
        summary.append(f"   Trade Signal: {signal['trade_signal']}")
        summary.append(f"   Reason: {signal['confidence_reason']}")
        
        # Add time analysis
        if signal['is_best_hour']:
            summary.append(f"   ⭐ Trading during BEST hour ({signal['hour']})")
        elif signal['is_worst_hour']:
            summary.append(f"   ⚠️ Trading during WORST hour ({signal['hour']})")
        else:
            summary.append(f"   ⏰ Trading during neutral hour ({signal['hour']})")
        
        if signal['is_best_day']:
            summary.append(f"   ⭐ Trading on BEST day ({signal['day_of_week']})")
        
        # Add market context
        summary.append(f"   📊 Market Context:")
        summary.append(f"      RSI: {signal['rsi']:.1f}")
        summary.append(f"      Volatility: {signal['volatility']:.4f}")
        summary.append(f"      Trend Strength: {signal['trend_strength']:.4f}")
        
        return "\n".join(summary)
    
    def validate_signal_quality(self, signal):
        """
        Validate that the signal meets your quality standards
        (Like double-checking your homework)
        """
        if not signal:
            return False, "No signal provided"
        
        quality_checks = []
        
        # Check 1: Enhanced confidence threshold
        if signal['enhanced_confidence'] >= self.confidence_threshold:
            quality_checks.append("✅ Confidence above threshold")
        else:
            quality_checks.append(f"❌ Confidence too low: {signal['enhanced_confidence']:.3f}")
        
        # Check 2: Time optimality
        if signal['should_trade_time']:
            quality_checks.append("✅ Optimal trading time")
        else:
            quality_checks.append("❌ Suboptimal trading time")
        
        # Check 3: Not worst hour
        if not signal['is_worst_hour']:
            quality_checks.append("✅ Not worst trading hour")
        else:
            quality_checks.append("❌ Trading during worst hour")
        
        # Check 4: Valid trade signal
        if signal['trade_signal'] in ['BUY', 'SELL']:
            quality_checks.append("✅ Valid trade signal")
        else:
            quality_checks.append("❌ No trade signal generated")
        
        # Overall assessment
        passed_checks = sum(1 for check in quality_checks if check.startswith("✅"))
        total_checks = len(quality_checks)
        
        is_high_quality = passed_checks >= 3  # Need at least 3/4 checks to pass
        
        return is_high_quality, quality_checks
    
    def get_model_info(self):
        """
        Get information about the loaded model
        (Like checking the specs of your super brain)
        """
        if not self.model_data:
            return "No model loaded"
        
        info = {
            'model_type': type(self.model_data['model']).__name__,
            'feature_count': len(self.model_data['features']),
            'features': self.model_data['features'],
            'expected_accuracy': '60.8%',
            'expected_sharpe': '13.20',
            'confidence_threshold': self.confidence_threshold,
            'optimal_hours': self.best_hours,
            'optimal_days': self.best_days
        }
        
        return info