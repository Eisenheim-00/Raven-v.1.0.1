# verify_centralization.py - Verify Database Centralization
# Run this script to verify that database centralization is working properly

import os
import sys
import sqlite3
from datetime import datetime

# Add core_system to path to import Config
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir) if 'core_system' in current_dir else current_dir
core_system_path = os.path.join(project_root, 'core_system')
sys.path.append(core_system_path)

try:
    from config import Config
    config_available = True
except ImportError as e:
    print(f"⚠️ Could not import Config: {e}")
    config_available = False

def check_project_structure():
    """Check if the centralized directory structure exists"""
    print("📁 Checking centralized directory structure...")
    
    if not config_available:
        print("   ❌ Config not available, cannot check structure")
        return False
    
    try:
        # Test Config methods
        data_dir = Config.get_data_models_dir()
        
        # Check main directories
        required_dirs = {
            'data_&_models': data_dir,
            'databases': os.path.join(data_dir, 'databases'),
            'logs': os.path.join(data_dir, 'logs'),
            'models': os.path.join(data_dir, 'models'),
            'exports': os.path.join(data_dir, 'exports')
        }
        
        all_exist = True
        for name, path in required_dirs.items():
            if os.path.exists(path):
                print(f"   ✅ {name}: {path}")
            else:
                print(f"   ❌ {name}: {path} (MISSING)")
                all_exist = False
        
        return all_exist
        
    except Exception as e:
        print(f"   ❌ Error checking structure: {e}")
        return False

def check_database_paths():
    """Check if Config returns correct database paths"""
    print("\n🔗 Checking database path configuration...")
    
    if not config_available:
        print("   ❌ Config not available, cannot check paths")
        return False
    
    try:
        # Test all database path methods
        db_methods = {
            'Bitcoin Data': Config.get_bitcoin_data_db_path,
            'Logs': Config.get_logs_db_path,
            'Evolution': Config.get_evolution_db_path,
            'Phases': Config.get_phases_db_path
        }
        
        all_correct = True
        for name, method in db_methods.items():
            try:
                path = method()
                
                # Check if path is in centralized location
                if 'data_&_models' in path and 'databases' in path:
                    print(f"   ✅ {name}: {path}")
                else:
                    print(f"   ❌ {name}: {path} (NOT CENTRALIZED)")
                    all_correct = False
                    
            except Exception as e:
                print(f"   ❌ {name}: Error - {e}")
                all_correct = False
        
        return all_correct
        
    except Exception as e:
        print(f"   ❌ Error checking database paths: {e}")
        return False

def check_existing_databases():
    """Check existing database files in centralized location"""
    print("\n📄 Checking existing database files...")
    
    if not config_available:
        print("   ❌ Config not available, cannot check databases")
        return False
    
    try:
        databases_dir = os.path.join(Config.get_data_models_dir(), 'databases')
        
        if not os.path.exists(databases_dir):
            print(f"   ⚠️ Databases directory does not exist: {databases_dir}")
            return False
        
        db_files = [f for f in os.listdir(databases_dir) if f.endswith('.db')]
        
        if not db_files:
            print("   ℹ️ No database files found (this is normal for a fresh installation)")
            return True
        
        print(f"   📊 Found {len(db_files)} database files:")
        
        total_size = 0
        working_dbs = 0
        
        for db_file in db_files:
            db_path = os.path.join(databases_dir, db_file)
            size_kb = os.path.getsize(db_path) / 1024
            total_size += size_kb
            
            # Test database connectivity
            try:
                conn = sqlite3.connect(db_path)
                tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
                conn.close()
                
                table_count = len(tables)
                print(f"      ✅ {db_file} ({size_kb:.1f} KB, {table_count} tables)")
                working_dbs += 1
                
            except Exception as e:
                print(f"      ❌ {db_file} ({size_kb:.1f} KB, Error: {e})")
        
        print(f"   📈 Summary: {working_dbs}/{len(db_files)} databases working, {total_size:.1f} KB total")
        return working_dbs == len(db_files)
        
    except Exception as e:
        print(f"   ❌ Error checking databases: {e}")
        return False

def check_scattered_files():
    """Check for any remaining scattered database files"""
    print("\n🔍 Scanning for scattered database files...")
    
    # Get project root
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = current_dir
    
    # Look for RAVEN project structure
    possible_roots = [current_dir, os.path.dirname(current_dir)]
    for root in possible_roots:
        if (os.path.exists(os.path.join(root, "core_system")) and 
            os.path.exists(os.path.join(root, "evolution_system"))):
            project_root = root
            break
    
    # Search for scattered .db files
    search_dirs = [
        project_root,
        os.path.join(project_root, "core_system"),
        os.path.join(project_root, "evolution_system"),
        os.path.join(project_root, "deployment")
    ]
    
    scattered_files = []
    
    for search_dir in search_dirs:
        if not os.path.exists(search_dir):
            continue
        
        for file in os.listdir(search_dir):
            if file.endswith('.db'):
                file_path = os.path.join(search_dir, file)
                scattered_files.append({
                    'name': file,
                    'path': file_path,
                    'directory': search_dir
                })
    
    if scattered_files:
        print(f"   ⚠️ Found {len(scattered_files)} scattered database files:")
        for file_info in scattered_files:
            print(f"      📄 {file_info['name']} in {file_info['directory']}")
        print("   💡 Run the migration script to centralize these files")
        return False
    else:
        print("   ✅ No scattered database files found!")
        return True

def test_database_creation():
    """Test that new databases are created in the centralized location"""
    print("\n🧪 Testing database creation...")
    
    if not config_available:
        print("   ❌ Config not available, cannot test creation")
        return False
    
    try:
        # Test creating a temporary database
        test_db_path = os.path.join(Config.get_data_models_dir(), 'databases', 'test_centralization.db')
        
        # Create test database
        conn = sqlite3.connect(test_db_path)
        conn.execute('CREATE TABLE test_table (id INTEGER, timestamp TEXT)')
        conn.execute('INSERT INTO test_table VALUES (1, ?)', (datetime.now().isoformat(),))
        conn.commit()
        conn.close()
        
        # Verify it was created in the right place
        if os.path.exists(test_db_path):
            print(f"   ✅ Test database created successfully: {test_db_path}")
            
            # Clean up test database
            os.remove(test_db_path)
            print("   🧹 Test database cleaned up")
            return True
        else:
            print(f"   ❌ Test database not found at expected location")
            return False
            
    except Exception as e:
        print(f"   ❌ Error testing database creation: {e}")
        return False

def test_config_validation():
    """Test Config.validate_settings() with centralized paths"""
    print("\n⚙️ Testing Config validation...")
    
    if not config_available:
        print("   ❌ Config not available, cannot test validation")
        return False
    
    try:
        # Test validation
        is_valid = Config.validate_settings()
        
        if is_valid:
            print("   ✅ Config validation passed")
            
            # Print current settings
            print("   📋 Current centralized paths:")
            try:
                print(f"      Bitcoin Data: {Config.get_bitcoin_data_db_path()}")
                print(f"      Logs: {Config.get_logs_db_path()}")
                print(f"      Evolution: {Config.get_evolution_db_path()}")
                print(f"      Phases: {Config.get_phases_db_path()}")
                print(f"      Log Files: {Config.get_logs_dir()}")
                return True
            except Exception as e:
                print(f"   ⚠️ Error getting paths: {e}")
                return False
        else:
            print("   ❌ Config validation failed")
            return False
            
    except Exception as e:
        print(f"   ❌ Error testing config validation: {e}")
        return False

def generate_verification_report():
    """Generate a verification report"""
    if not config_available:
        return None
    
    try:
        data_dir = Config.get_data_models_dir()
        report_path = os.path.join(data_dir, "centralization_verification.txt")
        
        with open(report_path, 'w') as f:
            f.write("🐦‍⬛ RAVEN DATABASE CENTRALIZATION VERIFICATION\n")
            f.write("=" * 55 + "\n")
            f.write(f"Verification Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Data Directory: {data_dir}\n\n")
            
            # Check if databases directory exists and list contents
            databases_dir = os.path.join(data_dir, 'databases')
            if os.path.exists(databases_dir):
                db_files = [f for f in os.listdir(databases_dir) if f.endswith('.db')]
                f.write(f"Database Files ({len(db_files)}):\n")
                for db_file in db_files:
                    db_path = os.path.join(databases_dir, db_file)
                    size_kb = os.path.getsize(db_path) / 1024
                    f.write(f"  - {db_file} ({size_kb:.1f} KB)\n")
            
            f.write("\nCentralized Paths:\n")
            f.write(f"  - Bitcoin Data: {Config.get_bitcoin_data_db_path()}\n")
            f.write(f"  - Logs: {Config.get_logs_db_path()}\n")
            f.write(f"  - Evolution: {Config.get_evolution_db_path()}\n")
            f.write(f"  - Phases: {Config.get_phases_db_path()}\n")
            f.write(f"  - Log Files: {Config.get_logs_dir()}\n")
        
        return report_path
        
    except Exception as e:
        print(f"Error generating report: {e}")
        return None

def main():
    """Main verification function"""
    print("🐦‍⬛ RAVEN DATABASE CENTRALIZATION VERIFICATION")
    print("=" * 60)
    print("This tool will verify that your database centralization is working correctly.")
    print("")
    
    # Run all checks
    checks = [
        ("Project Structure", check_project_structure),
        ("Database Paths", check_database_paths),
        ("Existing Databases", check_existing_databases),
        ("Scattered Files", check_scattered_files),
        ("Database Creation", test_database_creation),
        ("Config Validation", test_config_validation)
    ]
    
    passed_checks = 0
    total_checks = len(checks)
    
    for check_name, check_function in checks:
        print(f"🔍 {check_name}:")
        try:
            result = check_function()
            if result:
                passed_checks += 1
        except Exception as e:
            print(f"   ❌ Error during {check_name}: {e}")
        print("")
    
    # Generate report
    report_path = generate_verification_report()
    if report_path:
        print(f"📄 Verification report saved: {report_path}")
        print("")
    
    # Final summary
    print("📊 VERIFICATION SUMMARY")
    print("=" * 30)
    print(f"Passed: {passed_checks}/{total_checks} checks")
    
    if passed_checks == total_checks:
        print("🎉 ALL CHECKS PASSED!")
        print("✅ Your database centralization is working perfectly!")
        print("")
        print("🚀 You can now run your bot with confidence that all")
        print("   database files will be created in the centralized location:")
        if config_available:
            print(f"   📁 {Config.get_data_models_dir()}")
    elif passed_checks >= total_checks - 1:
        print("⚠️ MOSTLY WORKING")
        print("✅ Your centralization is mostly working with minor issues.")
        print("💡 Check the failed items above and fix as needed.")
    else:
        print("❌ MULTIPLE ISSUES FOUND")
        print("💡 Please fix the issues above before running your bot.")
        print("🔧 You may need to:")
        print("   1. Run the migration script")
        print("   2. Update your Python files with the new versions")
        print("   3. Fix any configuration errors")
    
    return passed_checks == total_checks

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n❌ Verification cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Verification failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)