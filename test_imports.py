"""
Test script to verify all modules can be imported correctly
"""

print("=" * 70)
print("JetX Predictor - Import Test")
print("=" * 70)

# Test 1: category_definitions
print("\n1. Testing category_definitions...")
try:
    from category_definitions import (
        CategoryDefinitions,
        FeatureEngineering,
        SEQUENCE_LENGTHS,
        CONFIDENCE_THRESHOLDS,
        ADVANCED_ANALYZERS_AVAILABLE
    )
    print("   ✅ category_definitions imported successfully")
    print(f"   - SEQUENCE_LENGTHS: {SEQUENCE_LENGTHS}")
    print(f"   - Advanced Analyzers Available: {ADVANCED_ANALYZERS_AVAILABLE}")
except Exception as e:
    print(f"   ❌ Error: {e}")

# Test 2: psychological_analyzer
print("\n2. Testing psychological_analyzer...")
try:
    from utils.psychological_analyzer import PsychologicalAnalyzer
    print("   ✅ PsychologicalAnalyzer imported successfully")
except Exception as e:
    print(f"   ❌ Error: {e}")

# Test 3: anomaly_streak_detector
print("\n3. Testing anomaly_streak_detector...")
try:
    from utils.anomaly_streak_detector import AnomalyStreakDetector
    print("   ✅ AnomalyStreakDetector imported successfully")
except Exception as e:
    print(f"   ❌ Error: {e}")

# Test 4: Feature extraction with sample data
print("\n4. Testing feature extraction...")
try:
    import numpy as np
    # Sample data (100 values)
    sample_data = list(np.random.uniform(1.0, 5.0, 100))
    
    features = FeatureEngineering.extract_all_features(sample_data)
    print(f"   ✅ Extracted {len(features)} features successfully")
    
    # Check if new features are present
    has_psych = any('bait' in key or 'trap' in key for key in features.keys())
    has_streak = any('streak' in key for key in features.keys())
    
    if has_psych:
        print("   ✅ Psikolojik analiz özellikleri mevcut")
    if has_streak:
        print("   ✅ Anomaly streak özellikleri mevcut")
        
except Exception as e:
    print(f"   ❌ Error: {e}")
    import traceback
    traceback.print_exc()

# Test 5: Window 1000 support
print("\n5. Testing Window 1000 support...")
try:
    import numpy as np
    # Create 1000+ data points
    large_data = list(np.random.uniform(1.0, 5.0, 1200))
    
    features = FeatureEngineering.extract_basic_features(large_data)
    
    # Check for 1000 window features
    has_1000_window = any('_1000' in key for key in features.keys())
    
    if has_1000_window:
        print("   ✅ Window 1000 özellikleri başarıyla çıkarıldı")
        print(f"   - mean_1000: {features.get('mean_1000', 'N/A')}")
        print(f"   - std_1000: {features.get('std_1000', 'N/A')}")
    else:
        print("   ⚠️  Window 1000 özellikleri bulunamadı")
        
except Exception as e:
    print(f"   ❌ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)
print("Test tamamlandı!")
print("=" * 70)
