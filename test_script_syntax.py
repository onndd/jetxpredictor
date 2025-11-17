#!/usr/bin/env python3
"""
Test script to verify all training scripts have correct syntax
"""

import ast
import os

def check_syntax(file_path):
    """Check if a Python file has valid syntax"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            source = f.read()
        
        # Try to parse the file
        ast.parse(source)
        print(f"✅ {file_path}: Syntax is valid")
        return True
    except SyntaxError as e:
        print(f"❌ {file_path}: Syntax error at line {e.lineno}: {e.msg}")
        return False
    except Exception as e:
        print(f"⚠️ {file_path}: Error checking syntax: {e}")
        return False

# Test the main training scripts
scripts_to_test = [
    'notebooks/jetx_PROGRESSIVE_TRAINING_MULTISCALE.py',
    'notebooks/jetx_CATBOOST_TRAINING_MULTISCALE.py',
    'notebooks/ImitationLearning_Agent.py'
]

print("="*80)
print("🔍 SYNTAX CHECK FOR TRAINING SCRIPTS")
print("="*80)

all_valid = True

for script in scripts_to_test:
    if os.path.exists(script):
        is_valid = check_syntax(script)
        if not is_valid:
            all_valid = False
    else:
        print(f"⚠️ {script}: File not found")

print("\n" + "="*80)
if all_valid:
    print("✅ ALL SCRIPTS HAVE VALID SYNTAX!")
else:
    print("❌ SOME SCRIPTS HAVE SYNTAX ERRORS!")
print("Please check the output above for details.")
print("="*80)
