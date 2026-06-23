"""Seed SQLite database with sample finance data."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.database import seed_sample_data

if __name__ == "__main__":
    seed_sample_data()
    print("Database seeded successfully.")
