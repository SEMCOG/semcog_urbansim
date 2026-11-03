"""Put the repository root on sys.path, so `pytest scripts/testing/...` works without PYTHONPATH=."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
