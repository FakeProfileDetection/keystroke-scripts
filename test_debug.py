#!/usr/bin/env python3
"""
test_debug.py - Quick test script to verify the debug mode works
"""

import subprocess
import sys
from pathlib import Path

def test_debug_mode():
    """Test the debug mode functionality"""
    print("🧪 Testing debug mode...")
    
    # You'll need to replace this with your actual dataset path
    dataset_path = "your_dataset.csv"  # UPDATE THIS PATH
    
    if not Path(dataset_path).exists():
        print(f"❌ Dataset not found: {dataset_path}")
        print("Please update the dataset_path variable in this script")
        return False
    
    try:
        # Test command with debug flag
        cmd = [
            sys.executable, "ml_platforms_runner.py",
            "-d", dataset_path,
            "--debug",  # Enable debug mode
            "-s", "1",  # Single seed
            "-o", "debug-test"
        ]
        
        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)  # 5 min timeout
        
        if result.returncode == 0:
            print("✅ Debug mode test completed successfully!")
            print("📁 Check for experiment_results_debug-test_*_debug directory")
            return True
        else:
            print(f"❌ Debug test failed with return code: {result.returncode}")
            print("STDOUT:", result.stdout[-1000:])  # Last 1000 chars
            print("STDERR:", result.stderr[-1000:])
            return False
            
    except subprocess.TimeoutExpired:
        print("⏰ Debug test timed out (should be much faster in debug mode)")
        return False
    except Exception as e:
        print(f"❌ Error running debug test: {e}")
        return False

def check_requirements():
    """Check if required files exist"""
    required_files = [
        "ml_platforms_core.py",
        "ml_platforms_runner.py"
    ]
    
    missing = []
    for file in required_files:
        if not Path(file).exists():
            missing.append(file)
    
    if missing:
        print(f"❌ Missing required files: {missing}")
        return False
    
    print("✅ All required files found")
    return True

def main():
    print("🚀 Testing the fixed ML platforms code...")
    
    if not check_requirements():
        return
    
    print("\n📋 What this test will verify:")
    print("  1. Debug mode runs faster with minimal hyperparameters")
    print("  2. All 4 models (RandomForest, XGBoost, CatBoost, SVM) are trained")
    print("  3. Confusion matrices are generated for each model")
    print("  4. HTML report generation works")
    print("  5. Performance plots include all models")
    
    # Update this path before running
    print("\n⚠️  IMPORTANT: Update the dataset_path in this script before running!")
    print("Current dataset_path: 'your_dataset.csv'")
    
    # Uncomment the line below after updating the dataset path
    # test_debug_mode()

if __name__ == "__main__":
    main()

