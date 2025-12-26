import sys
from pathlib import Path

# Ensure the src package is importable during tests
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
