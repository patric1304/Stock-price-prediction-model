"""
Reset all training data and start fresh.
Use this if you want to begin the 14-day cycle from scratch.
"""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

accumulated_data_path = Path("data/processed/training_data/accumulated_training_data.pkl")

if accumulated_data_path.exists():
    accumulated_data_path.unlink()
    print("✓ Deleted accumulated training data")
    print("Next run will start fresh from Day 1")
else:
    print("No accumulated data found - already fresh!")
