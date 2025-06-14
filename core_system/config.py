# config.py - Bot Configuration Settings
# Updated with centralized database path management

import os
from datetime import datetime

class Config:
    """
    Configuration settings for Raven - AI Trading Bot
    Updated with centralized database path management
    """
    
    # =============================================================================
    # PROJECT STRUCTURE MANAGEMENT - CENTRALIZED DATABASE PATHS
    # =============================================================================
    
    @classmethod
    def get_project_root(cls):
        """Get the project root directory (RAVEN V.0.1.3)"""
        # Get current file directory (core_system)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Go up one level to project root
        project_root = os.path.dirname(current_dir)
        return project_root
    
    @classmethod
    def get_data_models_dir(cls):
        """Get the data_&_models directory path"""
        project_root = cls.get_project_root()
        data_models_dir = os.path.join(project_root, "data_&_models")
        
        # Create directory if it doesn't exist
        os.makedirs(data_models_dir, exist_ok=True)
        
        # Create subdirectories for organization
        subdirs = ["databases", "models", "logs", "exports"]
        for subdir in subdirs:
            os.makedirs(os.path.join(data_models_dir, subdir), exist_ok=True)
        
        return data_models_dir
    
    @classmethod
    def get_database_path(cls, db_name):
        """Get centralized database path in data_&_models/databases/"""
        data_dir = cls.get_data_models_dir()
        databases_dir = os.path.join(data_dir, "databases")
        return os.path.join(databases_dir, db_name)
    
    # =============================================================================
    # ALPACA API SETTINGS - CORRECT APCA FORMAT
    # =============================================================================
    
    # Get these from: https://app.alpaca.markets/paper/dashboard/overview
    # These correspond to APCA-API-KEY-ID and APCA-API-SECRET-KEY headers
    APCA_API_KEY_ID = "PKHT65PAYTGMXKU5PSRZ"        # Your API Key ID
    APCA_API_SECRET_KEY = "coiQELlSctfL14WsMcLVKrfYooyQnEZ0ejuBV1Rr"    # Your API Secret Key
    
    # Paper trading URL (for safe testing)
    ALPACA_BASE_URL = "https://paper-api.alpaca.markets"
    
    # =============================================================================
    # TRADING SETTINGS (Your validated strategy parameters)
    # =============================================================================
    
    # Bitcoin symbol on Alpaca
    SYMBOL = "BTCUSD"
    
    # Optimal trading hours (from your 60.8% accuracy analysis)
    BEST_HOURS = [23, 19, 20, 4, 1, 7]  # UTC hours with best performance
    WORST_HOURS = [5, 18, 6, 21, 8, 15]  # Avoid these hours
    BEST_DAYS = [5, 3, 4]  # Saturday, Thursday, Friday (0=Monday, 6=Sunday)
    
    # Confidence threshold (from your analysis)
    ENHANCED_CONFIDENCE_THRESHOLD = 0.12  # 59.4% accuracy threshold
    
    # Position sizing (how much money to risk per trade)
    POSITION_SIZE_USD = 1000  # Start with $1000 per trade (paper money)
    MAX_POSITIONS = 1  # Only hold 1 Bitcoin position at a time
    
    # =============================================================================
    # CENTRALIZED DATABASE PATHS - ALL DATABASES IN data_&_models/databases/
    # =============================================================================
    
    # Main Bitcoin data database
    DATABASE_PATH = None  # Will be set dynamically
    
    # All database files will be in data_&_models/databases/
    @classmethod
    def get_bitcoin_data_db_path(cls):
        return cls.get_database_path("bitcoin_data.db")
    
    @classmethod
    def get_logs_db_path(cls):
        return cls.get_database_path("raven_logs.db")
    
    @classmethod  
    def get_evolution_db_path(cls):
        return cls.get_database_path("raven_evolution.db")
    
    @classmethod
    def get_phases_db_path(cls):
        return cls.get_database_path("raven_phases.db")
    
    # =============================================================================
    # MODEL AND DATA SETTINGS - CENTRALIZED PATHS
    # =============================================================================
    
    # Path to your trained model (Will be set dynamically)
    MODEL_PATH = None  # Will be found automatically in validate_settings()
    
    # How often to check for new data (minutes)
    DATA_UPDATE_INTERVAL = 60  # Every hour
    
    # =============================================================================
    # LOGGING AND MONITORING - CENTRALIZED IN data_&_models/logs/
    # =============================================================================
    
    @classmethod
    def get_logs_dir(cls):
        """Get centralized logs directory"""
        data_dir = cls.get_data_models_dir()
        logs_dir = os.path.join(data_dir, "logs")
        os.makedirs(logs_dir, exist_ok=True)
        return logs_dir
    
    @classmethod
    def get_log_file_path(cls):
        """Get centralized log file path"""
        logs_dir = cls.get_logs_dir()
        return os.path.join(logs_dir, f"bot_{datetime.now().strftime('%Y%m%d')}.log")
    
    # Email notifications (optional - for advanced users)
    ENABLE_EMAIL_ALERTS = False
    EMAIL_FOR_ALERTS = "your_email@gmail.com"
    
    # =============================================================================
    # RISK MANAGEMENT SETTINGS
    # =============================================================================
    
    # Maximum daily trades (safety limit)
    MAX_DAILY_TRADES = 5
    
    # Stop trading if losses exceed this amount (paper money)
    DAILY_LOSS_LIMIT = 2000  # Stop if lose more than $2000 in one day
    
    # Minimum time between trades (minutes)
    MIN_TIME_BETWEEN_TRADES = 60  # Don't trade more than once per hour
    
    # =============================================================================
    # DISPLAY SETTINGS
    # =============================================================================
    
    # Console output preferences
    SHOW_DETAILED_LOGS = True
    SHOW_SIGNAL_DETAILS = True
    SHOW_PORTFOLIO_UPDATES = True
    
    # =============================================================================
    # HELPER METHODS
    # =============================================================================
    
    @classmethod
    def validate_settings(cls):
        """
        Check if all important settings are configured
        Using correct APCA authentication format and centralized paths
        """
        errors = []
        
        # Initialize centralized paths
        cls.DATABASE_PATH = cls.get_bitcoin_data_db_path()
        
        print(f"📁 Centralizing database files in: {cls.get_data_models_dir()}")
        print(f"   Bitcoin data: {cls.get_bitcoin_data_db_path()}")
        print(f"   Logs: {cls.get_logs_db_path()}")
        print(f"   Evolution: {cls.get_evolution_db_path()}")
        print(f"   Phases: {cls.get_phases_db_path()}")
        
        # Check APCA API keys (correct format)
        if cls.APCA_API_KEY_ID in ["YOUR_ALPACA_API_KEY_ID_HERE", ""]:
            errors.append("❌ Please set your APCA_API_KEY_ID in config.py")
            errors.append("   Get from: https://app.alpaca.markets/paper/dashboard/overview")
        
        if cls.APCA_API_SECRET_KEY in ["YOUR_ALPACA_SECRET_KEY_HERE", ""]:
            errors.append("❌ Please set your APCA_API_SECRET_KEY in config.py")
            errors.append("   Get from: https://app.alpaca.markets/paper/dashboard/overview")
        
        # Smart model path detection for your structure
        data_models_dir = cls.get_data_models_dir()
        
        # Try multiple possible paths for your model
        model_paths_to_try = [
            os.path.join(data_models_dir, "models", "optimized_bitcoin_model.pkl"),
            os.path.join(data_models_dir, "optimized_bitcoin_model.pkl"),
            os.path.join(cls.get_project_root(), "models", "optimized_bitcoin_model.pkl"),
        ]
        
        model_found = False
        for model_path in model_paths_to_try:
            if os.path.exists(model_path):
                cls.MODEL_PATH = model_path
                model_found = True
                print(f"✅ Model found at: {model_path}")
                break
        
        if not model_found:
            errors.append(f"❌ Model file not found. Searched:")
            for path in model_paths_to_try:
                errors.append(f"   - {path}")
            errors.append(f"   Expected location: data_&_models/models/optimized_bitcoin_model.pkl")
        
        # Check position size is reasonable
        if cls.POSITION_SIZE_USD <= 0:
            errors.append("❌ POSITION_SIZE_USD must be greater than 0")
        
        if errors:
            print("\n🚨 CONFIGURATION ERRORS:")
            for error in errors:
                print(f"   {error}")
            print("\n💡 Fix these errors before running the bot!")
            print("📖 Alpaca Authentication Docs: https://docs.alpaca.markets/reference/authentication-2")
            return False
        else:
            print("✅ All configuration settings are valid!")
            print(f"🔑 Using APCA authentication format")
            print(f"🎯 Paper trading mode: {cls.ALPACA_BASE_URL}")
            print(f"📁 All databases centralized in: data_&_models/databases/")
            return True
    
    @classmethod
    def print_settings(cls):
        """
        Display current settings with centralized paths
        """
        print("\n⚙️ CURRENT BOT SETTINGS:")
        print(f"   Trading Symbol: {cls.SYMBOL}")
        print(f"   Position Size: ${cls.POSITION_SIZE_USD:,}")
        print(f"   Max Daily Trades: {cls.MAX_DAILY_TRADES}")
        print(f"   Confidence Threshold: {cls.ENHANCED_CONFIDENCE_THRESHOLD}")
        print(f"   Best Trading Hours: {cls.BEST_HOURS}")
        print(f"   Model Path: {cls.MODEL_PATH}")
        print(f"   Paper Trading: {cls.ALPACA_BASE_URL}")
        print(f"   Auth Format: APCA-API-KEY-ID / APCA-API-SECRET-KEY")
        print(f"\n📁 CENTRALIZED DATABASE PATHS:")
        print(f"   Data Directory: {cls.get_data_models_dir()}")
        print(f"   Bitcoin Data DB: {cls.get_bitcoin_data_db_path()}")
        print(f"   Logs DB: {cls.get_logs_db_path()}")
        print(f"   Evolution DB: {cls.get_evolution_db_path()}")
        print(f"   Phases DB: {cls.get_phases_db_path()}")
        print(f"   Log Files: {cls.get_logs_dir()}")
    
    @classmethod
    def cleanup_old_db_files(cls):
        """Clean up old database files from project root and other directories"""
        project_root = cls.get_project_root()
        core_system = os.path.join(project_root, "core_system")
        evolution_system = os.path.join(project_root, "evolution_system")
        
        # Database files that might exist in wrong locations
        old_db_files = [
            "bitcoin_data.db",
            "raven_logs.db", 
            "raven_evolution.db",
            "raven_phases.db"
        ]
        
        locations_to_check = [project_root, core_system, evolution_system]
        
        cleaned_files = []
        for location in locations_to_check:
            for db_file in old_db_files:
                old_path = os.path.join(location, db_file)
                if os.path.exists(old_path):
                    try:
                        os.remove(old_path)
                        cleaned_files.append(old_path)
                        print(f"🧹 Cleaned up: {old_path}")
                    except Exception as e:
                        print(f"⚠️ Could not remove {old_path}: {e}")
        
        if cleaned_files:
            print(f"✅ Cleaned up {len(cleaned_files)} old database files")
            print("📁 All databases are now centralized in data_&_models/databases/")
        else:
            print("✅ No old database files found to clean up")

# =============================================================================
# SETUP INSTRUCTIONS - UPDATED FOR CENTRALIZED DATABASES
# =============================================================================

"""
🚀 SETUP CHECKLIST FOR YOUR RAVEN BOT (UPDATED FOR CENTRALIZED DATABASES):

📁 STEP 1: CENTRALIZED DATABASE MANAGEMENT (AUTOMATIC!)
✅ All databases now go to: data_&_models/databases/
✅ All logs now go to: data_&_models/logs/
✅ Directory structure created automatically
✅ Old scattered .db files can be cleaned up automatically

🔑 STEP 2: APCA API KEYS (YOU NEED TO DO THIS!)
1. Go to: https://app.alpaca.markets/paper/dashboard/overview
2. Click "Generate New Keys" 
3. Copy your API Key ID and Secret Key
4. Replace the placeholder values above:
   
   APCA_API_KEY_ID = "PK..."        # Your actual API Key ID
   APCA_API_SECRET_KEY = "..."      # Your actual Secret Key

🚀 STEP 3: RUN THE BOT WITH CENTRALIZED DATABASES
1. Navigate to core_system folder: cd core_system
2. Run: python main.py
3. All databases will be automatically created in data_&_models/databases/
4. All logs will go to data_&_models/logs/

🧹 STEP 4: CLEAN UP OLD SCATTERED DATABASE FILES (OPTIONAL)
Run this in Python to clean up old .db files:
```python
from config import Config
Config.cleanup_old_db_files()
```

📁 YOUR NEW ORGANIZED STRUCTURE:
RAVEN V.0.1.3/
├── core_system/
├── evolution_system/
├── data_&_models/
│   ├── databases/          # ALL .db files go here
│   │   ├── bitcoin_data.db
│   │   ├── raven_logs.db
│   │   ├── raven_evolution.db
│   │   └── raven_phases.db
│   ├── logs/              # ALL log files go here
│   ├── models/            # Your ML model files
│   └── exports/           # Data exports
└── deployment/

🎯 BENEFITS:
✅ No more scattered .db files!
✅ Easy backup (just copy data_&_models folder)
✅ Clean project structure
✅ All data in one organized location
✅ Automatic directory creation
"""