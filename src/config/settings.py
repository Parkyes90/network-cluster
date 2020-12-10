import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
PAPERS_DIR = os.path.join(BASE_DIR, "data", "papers")
OUTPUTS_DIR = os.path.join(BASE_DIR, "data", "outputs")
