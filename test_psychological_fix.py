#!/usr/bin/env python3
"""
Test script to verify the infinite recursion fix in psychological_analyzer.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.psychological_analyzer import PsychologicalAnalyzer

def test_psychological_analyzer():
    """Test the psychological analyzer with sample data"""
    print("🧪 Testing PsychologicalAnalyzer Fix...")
    
    # Create analyzer
    analyzer = PsychologicalAnalyzer(threshold=1.5)
    
    # Test with sample data (enough to trigger all features)
    sample_history = [
        1.2, 1.8, 2.1, 1.4, 1.6, 3.2, 1.1, 1.3, 2.8, 1.5,
        1.7, 2.4, 1.2, 1.9, 3.5, 1.1, 1.4, 2.1, 1.6, 1.8,
        2.3, 1.3, 1.7, 2.9, 1.2, 1.5, 2.2, 1.4, 1.8, 2.5
    ]
    
    print(f"📊 Testing with {len(sample_history)} sample values...")
    
    try:
        # Test analyze_psychological_patterns (this was causing infinite recursion)
        print("🔍 Testing analyze_psychological_patterns...")
        features = analyzer.analyze_psychological_patterns(sample_history)
        
        print("✅ SUCCESS: No infinite recursion detected!")
        print(f"📈 Generated {len(features)} features:")
        
        for key, value in features.items():
            print(f"   • {key}: {value:.3f}")
        
        # Test detect_manipulation_pattern specifically
        print("\n🎭 Testing detect_manipulation_pattern...")
        manipulation_score = analyzer.detect_manipulation_pattern(sample_history)
        print(f"✅ Manipulation score: {manipulation_score:.3f}")
        
        # Test warning generation
        print("\n⚠️ Testing warning generation...")
        warning = analyzer.get_psychological_warning(features)
        print(f"📢 Warning: {warning}")
        
        print("\n🎉 ALL TESTS PASSED! The infinite recursion fix is working correctly.")
        return True
        
    except RecursionError as e:
        print(f"❌ FAILED: Infinite recursion still detected: {e}")
        return False
    except Exception as e:
        print(f"❌ FAILED: Unexpected error: {e}")
        return False

if __name__ == "__main__":
    success = test_psychological_analyzer()
    sys.exit(0 if success else 1)
