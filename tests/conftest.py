import sys
from pathlib import Path

# Dynamically locate source directory.
# Start from the test file and search for 'src' or use the parent if it contains Python modules.
TEST_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TEST_DIR.parent

# Try to find src/ in the project root; if it doesn't exist, assume src is already on the path
SRC_DIR = PROJECT_ROOT / "src"
if SRC_DIR.exists() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
