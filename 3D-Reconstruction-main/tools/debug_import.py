#!/usr/bin/env python3
"""
Test script for debugging import issues
"""

import sys
import os
from pathlib import Path

print("=== Debug Info ===")
print(f"Python version: {sys.version}")
print(f"Current working directory: {os.getcwd()}")
print(f"Python path: {sys.path}")
print()

# Check if key files exist
files_to_check = [
    "scene_editing/insert_instance.py",
    "scene_editing/scene_editing.py",
    "utils/misc.py",
    "datasets/driving_dataset.py"
]

print("=== File Existence Check ===")
for file_path in files_to_check:
    exists = os.path.exists(file_path)
    print(f"{file_path}: {'✓' if exists else '✗'}")

print()

# Try importing key modules
print("=== Module Import Test ===")

try:
    from utils.misc import import_str
    print("✓ utils.misc import successful")
except Exception as e:
    print(f"✗ utils.misc import failed: {e}")

try:
    from datasets.driving_dataset import DrivingDataset
    print("✓ datasets.driving_dataset import successful")
except Exception as e:
    print(f"✗ datasets.driving_dataset import failed: {e}")

try:
    from omegaconf import OmegaConf
    print("✓ omegaconf import successful")
except Exception as e:
    print(f"✗ omegaconf import failed: {e}")

# Check scene_editing module
print()
print("=== scene_editing Module Check ===")
try:
    # Add current directory to path
    current_dir = Path(__file__).parent
    project_root = current_dir.parent if current_dir.name == "scene_editing" else current_dir
    sys.path.insert(0, str(project_root))
    
    import scene_editing.scene_editing as se
    print("✓ scene_editing.scene_editing import successful")
    
    # Check if key functions exist
    required_functions = [
        'save_smpl_instance',
        'save_rigid_instance', 
        'load_instance_data',
        'insert_smpl_instance',
        'insert_rigid_instance'
    ]
    
    for func_name in required_functions:
        if hasattr(se, func_name):
            print(f"  ✓ Function {func_name} exists")
        else:
            print(f"  ✗ Function {func_name} does not exist")
            
except Exception as e:
    print(f"✗ scene_editing.scene_editing import failed: {e}")
    import traceback
    traceback.print_exc()

print("\n=== Check Complete ===")