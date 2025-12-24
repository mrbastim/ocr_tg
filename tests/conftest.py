import os
import sys

# Add project root to sys.path for imports like `from bot...`
ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
