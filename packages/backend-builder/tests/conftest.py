import sys
from pathlib import Path

# Make the fixture package importable by name, as an installed app would be.
sys.path.insert(0, str(Path(__file__).parent))
