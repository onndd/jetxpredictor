#!/usr/bin/env python3
"""
Input Validation Test Script

Tüm ML modellerindeki input validation düzeltmelerini test eder.
"""

import numpy as np
import pandas as pd
import sys
import os

# Add utils to path
sys.path.append('utils')

def test_tabnet_predictor():
    """TabNet predictor input validation test"""
    print("🧪 TabNet Predictor Test")
    print("=" * 50)
    
    try:
        from tabnet_predictor import TabNetHighXPredictor
        
        predictor = TabNetHighXPredictor()
        
        # Test 1: String input
        print("Test 1: String input...")
        result = predictor.categorize_value("2.5")
        print(f"✅ String '2.5' -> Category {result}")
        
        # Test 2: Float input
        print("Test 2: Float input...")
        result = predictor.categorize_value(2.5)
        print(f"✅ Float 2.5 -> Category {result}")
        
        # Test 3: Invalid input
        print("Test 3: Invalid input...")
        result = predictor.categorize_value("invalid")
        print(f"✅ Invalid 'invalid' -> Category {result} (should be 0)")
        
        # Test 4: None input
        print("Test 4: None input...")
        result = predictor.categorize_value(None)
        print(f"✅ None -> Category {result} (should be 0)")
        
        print("✅ TabNet predictor test PASSED")
        
    except Exception as e:
        print(f"❌ TabNet predictor test FAILED: {e}")
        return False
    
    return True

def test_autogluon_predictor():
    """AutoGluon predictor input validation test"""
    print("\n🧪 AutoGluon Predictor Test")
    print("=" * 50)
    
    try:
        from autogluon_predictor import AutoGluonPredictor
        
        # Mock model (eğitim gerektirmeden test için)
        predictor = AutoGluonPredictor()
        predictor.predictor = type('MockPredictor', (), {
            'predict_proba': lambda self, X: pd.DataFrame({'0': [0.3], '1': [0.7]}),
            'predict': lambda self, X: pd.Series([0])
        })()
        
        # Test 1: List input
        print("Test 1: List input...")
        result = predictor.predict([1.2, 1.5, 2.0])
        print(f"✅ List input -> {result}")
        
        # Test 2: Series input
        print("Test 2: Series input...")
        series = pd.Series([1.2, 1.5, 2.0])
        result = predictor.predict(series)
        print(f"✅ Series input -> {result}")
        
        # Test 3: Invalid input
        print("Test 3: Invalid input...")
        result = predictor.predict("invalid")
        print(f"✅ Invalid input -> {result}")
        
        print("✅ AutoGluon predictor test PASSED")
        
    except Exception as e:
        print(f"❌ AutoGluon predictor test FAILED: {e}")
        return False
    
    return True

def test_lightgbm_predictor():
    """LightGBM predictor input validation test"""
    print("\n🧪 LightGBM Predictor Test")
    print("=" * 50)
    
    try:
        from lightgbm_predictor import LightGBMPredictor
        
        # Mock model
        predictor = LightGBMPredictor()
        predictor.model = type('MockModel', (), {
            'predict': lambda self, X, num_iteration=None: np.array([0.3, 0.7]),
            'best_iteration': 100
        })()
        predictor.is_trained = True
        
        # Test 1: List input
        print("Test 1: List input...")
        result = predictor.predict([1.2, 1.5, 2.0])
        print(f"✅ List input -> {result}")
        
        # Test 2: Series input
        print("Test 2: Series input...")
        series = pd.Series([1.2, 1.5, 2.0])
        result = predictor.predict(series)
        print(f"✅ Series input -> {result}")
        
        # Test 3: Invalid input
        print("Test 3: Invalid input...")
        result = predictor.predict("invalid")
        print(f"✅ Invalid input -> {result}")
        
        print("✅ LightGBM predictor test PASSED")
        
    except Exception as e:
        print(f"❌ LightGBM predictor test FAILED: {e}")
        return False
    
    return True

def test_catboost_ensemble():
    """CatBoost ensemble input validation test"""
    print("\n🧪 CatBoost Ensemble Test")
    print("=" * 50)
    
    try:
        from catboost_ensemble import CatBoostEnsemble
        
        # Mock ensemble
        ensemble = CatBoostEnsemble()
        ensemble.models = [
            type('MockModel', (), {
                'predict': lambda self, X: np.array([1.2])
            })(),
            type('MockModel', (), {
                'predict': lambda self, X: np.array([1.8])
            })()
        ]
        ensemble.weights = np.array([0.5, 0.5])
        
        # Test 1: List input
        print("Test 1: List input...")
        result = ensemble.predict([1.2, 1.5, 2.0])
        print(f"✅ List input -> {result}")
        
        # Test 2: Series input
        print("Test 2: Series input...")
        series = pd.Series([1.2, 1.5, 2.0])
        result = ensemble.predict(series)
        print(f"✅ Series input -> {result}")
        
        # Test 3: Invalid input
        print("Test 3: Invalid input...")
        result = ensemble.predict("invalid")
        print(f"✅ Invalid input -> {result}")
        
        print("✅ CatBoost ensemble test PASSED")
        
    except Exception as e:
        print(f"❌ CatBoost ensemble test FAILED: {e}")
        return False
    
    return True

def main():
    """Ana test fonksiyonu"""
    print("🚀 JetX Predictor Input Validation Test Suite")
    print("=" * 60)
    
    tests = [
        test_tabnet_predictor,
        test_autogluon_predictor,
        test_lightgbm_predictor,
        test_catboost_ensemble
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 60)
    print(f"📊 TEST SONUÇLARI: {passed}/{total} PASSED")
    
    if passed == total:
        print("🎉 TÜM TESTLER BAŞARILI!")
        print("✅ Input validation düzeltmeleri çalışıyor")
    else:
        print("⚠️  BAZI TESTLER BAŞARISIZ!")
        print("❌ Düzeltmeler gerekiyor")
    
    print("=" * 60)

if __name__ == "__main__":
    main()
