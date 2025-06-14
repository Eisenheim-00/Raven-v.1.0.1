# phase_controller.py - Automated Evolution Phase Manager
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

import sqlite3
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Optional, Tuple
from enum import Enum
import json

# Import Config for centralized database paths
from config import Config

class EvolutionPhase(Enum):
    """Evolution phases for automated progression"""
    PHASE_0_BASELINE = "baseline"           # Original 60.8% system only
    PHASE_1_BASIC_LEARNING = "basic_learning"     # Add memory and basic learning
    PHASE_2_REGIME_DETECTION = "regime_detection"  # Add market regime detection
    PHASE_3_REGIME_SPECIALISTS = "regime_specialists"  # Add specialist strategies
    PHASE_4_FULL_EVOLUTION = "full_evolution"      # Full adaptive system
    PHASE_5_ADVANCED_LEARNING = "advanced_learning"  # Advanced optimizations

class RavenPhaseController:
    """
    Automatically manages Raven's evolution through phases
    Perfect for 24/7 operation with no manual intervention needed!
    Updated to use centralized database paths.
    """
    
    def __init__(self, db_path=None):
        # Use centralized database path from Config
        if db_path is None:
            self.db_path = Config.get_phases_db_path()
        else:
            self.db_path = db_path
            
        self.current_phase = EvolutionPhase.PHASE_1_BASIC_LEARNING  # Start with basic learning
        self.phase_start_time = datetime.now()
        self.setup_phase_database()
        
        # Phase progression criteria
        self.phase_criteria = {
            EvolutionPhase.PHASE_1_BASIC_LEARNING: {
                'min_days': 2,           # At least 2 days
                'min_trades': 20,        # At least 20 trades
                'min_accuracy': 0.55,    # At least 55% accuracy
                'description': 'Basic learning and memory system'
            },
            EvolutionPhase.PHASE_2_REGIME_DETECTION: {
                'min_days': 4,           # At least 4 days total
                'min_trades': 40,        # At least 40 trades
                'min_accuracy': 0.57,    # At least 57% accuracy
                'description': 'Market regime detection activated'
            },
            EvolutionPhase.PHASE_3_REGIME_SPECIALISTS: {
                'min_days': 7,           # At least 1 week
                'min_trades': 75,        # At least 75 trades
                'min_accuracy': 0.58,    # At least 58% accuracy
                'description': 'Regime specialists activated'
            },
            EvolutionPhase.PHASE_4_FULL_EVOLUTION: {
                'min_days': 14,          # At least 2 weeks
                'min_trades': 150,       # At least 150 trades
                'min_accuracy': 0.59,    # At least 59% accuracy
                'description': 'Full evolution system with adaptation'
            },
            EvolutionPhase.PHASE_5_ADVANCED_LEARNING: {
                'min_days': 30,          # At least 1 month
                'min_trades': 300,       # At least 300 trades
                'min_accuracy': 0.60,    # At least 60% accuracy (TARGET!)
                'description': 'Advanced learning and optimization'
            }
        }
        
        # Load current phase from centralized database
        self.load_current_phase()
        
        print(f"🎮 Phase Controller initialized - Current: {self.current_phase.value}")
        print(f"   📁 Phase database: {self.db_path}")
        print(f"   📁 Centralized in: {os.path.dirname(os.path.dirname(self.db_path))}")
    
    def setup_phase_database(self):
        """Setup centralized database to track phase progression"""
        try:
            # Ensure database directory exists
            db_dir = os.path.dirname(self.db_path)
            os.makedirs(db_dir, exist_ok=True)
            
            conn = sqlite3.connect(self.db_path)
            
            # Phase progression history
            conn.execute('''
                CREATE TABLE IF NOT EXISTS phase_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    phase_name TEXT,
                    start_time TEXT,
                    end_time TEXT,
                    duration_hours REAL,
                    trades_completed INTEGER,
                    accuracy_achieved REAL,
                    reason_for_change TEXT,
                    performance_data TEXT,
                    created_at TEXT
                )
            ''')
            
            # Current phase status
            conn.execute('''
                CREATE TABLE IF NOT EXISTS current_phase (
                    id INTEGER PRIMARY KEY,
                    phase_name TEXT,
                    start_time TEXT,
                    trades_since_start INTEGER,
                    last_updated TEXT
                )
            ''')
            
            conn.commit()
            conn.close()
            
            print("✅ Centralized phase database initialized")
            
        except Exception as e:
            print(f"❌ Phase database setup failed: {e}")
    
    def load_current_phase(self):
        """Load current phase from centralized database"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            result = conn.execute('''
                SELECT phase_name, start_time FROM current_phase WHERE id = 1
            ''').fetchone()
            
            if result:
                phase_name, start_time_str = result
                self.current_phase = EvolutionPhase(phase_name)
                self.phase_start_time = datetime.fromisoformat(start_time_str)
                print(f"📖 Loaded phase from centralized DB: {self.current_phase.value} (started {start_time_str})")
            else:
                # First time - insert initial phase
                self.save_current_phase()
                print(f"🆕 Starting with Phase 1: {self.current_phase.value}")
            
            conn.close()
            
        except Exception as e:
            print(f"⚠️ Failed to load current phase from centralized database: {e}")
    
    def save_current_phase(self):
        """Save current phase to centralized database"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            conn.execute('''
                INSERT OR REPLACE INTO current_phase (id, phase_name, start_time, trades_since_start, last_updated)
                VALUES (1, ?, ?, 0, ?)
            ''', (
                self.current_phase.value,
                self.phase_start_time.isoformat(),
                datetime.now().isoformat()
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            print(f"⚠️ Failed to save current phase to centralized database: {e}")
    
    def check_phase_progression(self, evolution_tracker, regime_specialist) -> Tuple[bool, Optional[EvolutionPhase]]:
        """
        Check if it's time to progress to the next phase
        Returns: (should_progress, next_phase)
        """
        try:
            # Calculate time since phase start
            time_since_start = datetime.now() - self.phase_start_time
            days_elapsed = time_since_start.total_seconds() / (24 * 3600)
            
            # Get performance data
            performance = evolution_tracker.get_recent_performance(hours=24*30)  # Last 30 days
            
            if not performance:
                return False, None
            
            total_trades = performance['total_trades']
            current_accuracy = performance['accuracy']
            
            # Determine next phase
            next_phase = self._get_next_phase()
            
            if not next_phase:
                return False, None  # Already at highest phase
            
            # Check progression criteria
            criteria = self.phase_criteria[next_phase]
            
            meets_time = days_elapsed >= criteria['min_days']
            meets_trades = total_trades >= criteria['min_trades']
            meets_accuracy = current_accuracy >= criteria['min_accuracy']
            
            print(f"🔍 Phase Progression Check ({self.current_phase.value} → {next_phase.value}):")
            print(f"   Time: {days_elapsed:.1f}/{criteria['min_days']} days {'✅' if meets_time else '❌'}")
            print(f"   Trades: {total_trades}/{criteria['min_trades']} {'✅' if meets_trades else '❌'}")
            print(f"   Accuracy: {current_accuracy:.1%}/{criteria['min_accuracy']:.1%} {'✅' if meets_accuracy else '❌'}")
            print(f"   📁 Checking from: {self.db_path}")
            
            # All criteria must be met
            should_progress = meets_time and meets_trades and meets_accuracy
            
            if should_progress:
                return True, next_phase
            else:
                return False, None
                
        except Exception as e:
            print(f"⚠️ Phase progression check failed: {e}")
            return False, None
    
    def _get_next_phase(self) -> Optional[EvolutionPhase]:
        """Get the next phase in sequence"""
        phase_order = [
            EvolutionPhase.PHASE_1_BASIC_LEARNING,
            EvolutionPhase.PHASE_2_REGIME_DETECTION,
            EvolutionPhase.PHASE_3_REGIME_SPECIALISTS,
            EvolutionPhase.PHASE_4_FULL_EVOLUTION,
            EvolutionPhase.PHASE_5_ADVANCED_LEARNING
        ]
        
        try:
            current_index = phase_order.index(self.current_phase)
            if current_index < len(phase_order) - 1:
                return phase_order[current_index + 1]
            else:
                return None  # Already at highest phase
        except ValueError:
            return EvolutionPhase.PHASE_1_BASIC_LEARNING  # Default fallback
    
    def progress_to_next_phase(self, next_phase: EvolutionPhase, performance_data: Dict):
        """Progress to the next evolution phase"""
        try:
            # Record phase completion in centralized database
            old_phase = self.current_phase
            time_elapsed = datetime.now() - self.phase_start_time
            hours_elapsed = time_elapsed.total_seconds() / 3600
            
            # Save to history
            conn = sqlite3.connect(self.db_path)
            
            conn.execute('''
                INSERT INTO phase_history 
                (phase_name, start_time, end_time, duration_hours, trades_completed, 
                 accuracy_achieved, reason_for_change, performance_data, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                old_phase.value,
                self.phase_start_time.isoformat(),
                datetime.now().isoformat(),
                hours_elapsed,
                performance_data.get('total_trades', 0),
                performance_data.get('accuracy', 0.0),
                'Automatic progression - criteria met',
                json.dumps(performance_data),
                datetime.now().isoformat()
            ))
            
            conn.commit()
            conn.close()
            
            # Update current phase
            self.current_phase = next_phase
            self.phase_start_time = datetime.now()
            self.save_current_phase()
            
            print(f"\n🎉 PHASE PROGRESSION COMPLETED!")
            print(f"   {old_phase.value} → {next_phase.value}")
            print(f"   Duration: {hours_elapsed:.1f} hours")
            print(f"   Trades: {performance_data.get('total_trades', 0)}")
            print(f"   Accuracy: {performance_data.get('accuracy', 0.0):.1%}")
            print(f"   New phase: {self.phase_criteria[next_phase]['description']}")
            print(f"   📁 Recorded in: {self.db_path}")
            
            return True
            
        except Exception as e:
            print(f"❌ Phase progression failed: {e}")
            return False
    
    def get_phase_configuration(self) -> Dict:
        """Get configuration settings for current phase"""
        phase_configs = {
            EvolutionPhase.PHASE_1_BASIC_LEARNING: {
                'learning_enabled': True,
                'regime_detection_enabled': False,
                'regime_specialists_enabled': False,
                'confidence_adaptation_enabled': True,
                'advanced_features_enabled': False,
                'safety_threshold': 3,  # More conservative safety
                'confidence_adjustment_factor': 0.01,  # Small adjustments
                'description': 'Basic learning with memory tracking'
            },
            EvolutionPhase.PHASE_2_REGIME_DETECTION: {
                'learning_enabled': True,
                'regime_detection_enabled': True,
                'regime_specialists_enabled': False,
                'confidence_adaptation_enabled': True,
                'advanced_features_enabled': False,
                'safety_threshold': 3,
                'confidence_adjustment_factor': 0.015,
                'description': 'Learning + regime detection'
            },
            EvolutionPhase.PHASE_3_REGIME_SPECIALISTS: {
                'learning_enabled': True,
                'regime_detection_enabled': True,
                'regime_specialists_enabled': True,
                'confidence_adaptation_enabled': True,
                'advanced_features_enabled': False,
                'safety_threshold': 4,
                'confidence_adjustment_factor': 0.02,
                'description': 'Full regime specialist system'
            },
            EvolutionPhase.PHASE_4_FULL_EVOLUTION: {
                'learning_enabled': True,
                'regime_detection_enabled': True,
                'regime_specialists_enabled': True,
                'confidence_adaptation_enabled': True,
                'advanced_features_enabled': True,
                'safety_threshold': 5,  # Less conservative (more confident)
                'confidence_adjustment_factor': 0.025,
                'description': 'Full evolution with all features'
            },
            EvolutionPhase.PHASE_5_ADVANCED_LEARNING: {
                'learning_enabled': True,
                'regime_detection_enabled': True,
                'regime_specialists_enabled': True,
                'confidence_adaptation_enabled': True,
                'advanced_features_enabled': True,
                'safety_threshold': 5,
                'confidence_adjustment_factor': 0.03,  # Maximum adaptation
                'description': 'Advanced optimization mode'
            }
        }
        
        return phase_configs.get(self.current_phase, phase_configs[EvolutionPhase.PHASE_1_BASIC_LEARNING])
    
    def check_emergency_fallback(self, performance_data: Dict) -> bool:
        """Check if emergency fallback to previous phase is needed"""
        if not performance_data:
            return False
        
        recent_accuracy = performance_data.get('recent_trend', performance_data.get('accuracy', 1.0))
        total_trades = performance_data.get('total_trades', 0)
        
        # Emergency fallback conditions
        emergency_conditions = [
            (recent_accuracy < 0.45 and total_trades >= 20),  # Very poor recent performance
            (recent_accuracy < 0.50 and total_trades >= 50),  # Poor sustained performance
        ]
        
        if any(emergency_conditions):
            print(f"🚨 EMERGENCY FALLBACK TRIGGERED!")
            print(f"   Recent accuracy: {recent_accuracy:.1%}")
            print(f"   Total trades: {total_trades}")
            print(f"   📁 From centralized database: {self.db_path}")
            return True
        
        return False
    
    def emergency_fallback(self):
        """Emergency fallback to previous phase or baseline"""
        try:
            # Get previous phase
            phase_order = [
                EvolutionPhase.PHASE_1_BASIC_LEARNING,
                EvolutionPhase.PHASE_2_REGIME_DETECTION,
                EvolutionPhase.PHASE_3_REGIME_SPECIALISTS,
                EvolutionPhase.PHASE_4_FULL_EVOLUTION,
                EvolutionPhase.PHASE_5_ADVANCED_LEARNING
            ]
            
            current_index = phase_order.index(self.current_phase)
            
            if current_index > 0:
                fallback_phase = phase_order[current_index - 1]
            else:
                fallback_phase = EvolutionPhase.PHASE_1_BASIC_LEARNING
            
            print(f"🔄 Emergency fallback: {self.current_phase.value} → {fallback_phase.value}")
            
            # Record emergency fallback in centralized database
            conn = sqlite3.connect(self.db_path)
            time_elapsed = datetime.now() - self.phase_start_time
            hours_elapsed = time_elapsed.total_seconds() / 3600
            
            conn.execute('''
                INSERT INTO phase_history 
                (phase_name, start_time, end_time, duration_hours, trades_completed, 
                 accuracy_achieved, reason_for_change, performance_data, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                self.current_phase.value,
                self.phase_start_time.isoformat(),
                datetime.now().isoformat(),
                hours_elapsed,
                0,  # Emergency fallback
                0.0,  # Poor performance
                'Emergency fallback - poor performance',
                '{"emergency": true}',
                datetime.now().isoformat()
            ))
            
            conn.commit()
            conn.close()
            
            # Update to fallback phase
            self.current_phase = fallback_phase
            self.phase_start_time = datetime.now()
            self.save_current_phase()
            
            print(f"   📁 Emergency fallback recorded in: {self.db_path}")
            
            return True
            
        except Exception as e:
            print(f"❌ Emergency fallback failed: {e}")
            return False
    
    def get_phase_status(self) -> Dict:
        """Get comprehensive phase status from centralized database"""
        time_since_start = datetime.now() - self.phase_start_time
        days_elapsed = time_since_start.total_seconds() / (24 * 3600)
        
        # Get next phase info
        next_phase = self._get_next_phase()
        next_criteria = self.phase_criteria.get(next_phase) if next_phase else None
        
        return {
            'current_phase': self.current_phase.value,
            'phase_description': self.phase_criteria.get(self.current_phase, {}).get('description', 'Unknown'),
            'days_in_phase': days_elapsed,
            'phase_start_time': self.phase_start_time.isoformat(),
            'next_phase': next_phase.value if next_phase else None,
            'next_phase_criteria': next_criteria,
            'configuration': self.get_phase_configuration(),
            'database_location': self.db_path
        }
    
    def print_phase_status(self, performance_data: Optional[Dict] = None):
        """Print detailed phase status"""
        status = self.get_phase_status()
        
        print(f"\n🎮 RAVEN EVOLUTION PHASE STATUS")
        print(f"📁 Data source: {status['database_location']}")
        print(f"="*50)
        print(f"Current Phase: {status['current_phase'].upper()}")
        print(f"Description: {status['phase_description']}")
        print(f"Time in Phase: {status['days_in_phase']:.1f} days")
        
        if performance_data:
            print(f"\nPerformance in Phase:")
            print(f"   Trades: {performance_data.get('total_trades', 0)}")
            print(f"   Accuracy: {performance_data.get('accuracy', 0.0):.1%}")
            print(f"   Recent Trend: {performance_data.get('recent_trend', 0.0):.1%}")
        
        if status['next_phase']:
            criteria = status['next_phase_criteria']
            print(f"\nNext Phase: {status['next_phase'].upper()}")
            if criteria:
                print(f"   Requirements:")
                print(f"   - Min Days: {criteria['min_days']}")
                print(f"   - Min Trades: {criteria['min_trades']}")
                print(f"   - Min Accuracy: {criteria['min_accuracy']:.1%}")
        else:
            print(f"\n🏆 Maximum Phase Reached!")
        
        print(f"="*50)
    
    def export_phase_data(self, output_dir=None):
        """Export phase progression data for analysis"""
        if not output_dir:
            # Export to data_&_models/exports/ by default
            data_dir = Config.get_data_models_dir()
            output_dir = os.path.join(data_dir, "exports")
            os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Export each table
            tables = ['phase_history', 'current_phase']
            
            for table in tables:
                df = pd.read_sql_query(f"SELECT * FROM {table}", conn)
                csv_file = os.path.join(output_dir, f"raven_phases_{table}_{timestamp}.csv")
                df.to_csv(csv_file, index=False)
                print(f"✅ Exported {len(df)} phase records to {csv_file}")
            
            conn.close()
            print(f"📁 All phase exports saved to: {output_dir}")
            
        except Exception as e:
            print(f"❌ Failed to export phase data: {e}")
    
    def get_database_info(self):
        """Get information about the centralized phase database"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Get table sizes
            tables = ['phase_history', 'current_phase']
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
            print(f"Failed to get phase database info: {e}")
            return None