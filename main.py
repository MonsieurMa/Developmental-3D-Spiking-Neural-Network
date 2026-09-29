"""Transitional CLI wrapper. New code belongs in src/bionic_brain."""
from pathlib import Path
import sys

SRC = Path(__file__).resolve().parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bionic_brain.interfaces.cli import main

if __name__ == "__main__":
    raise SystemExit(main())

