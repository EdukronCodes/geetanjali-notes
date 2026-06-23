"""Seed ledger database from sample CSV."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.database import seed_from_csv

if __name__ == "__main__":
    seed_from_csv()
    print("Ledger database seeded.")
