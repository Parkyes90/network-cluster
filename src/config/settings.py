import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = os.path.join(BASE_DIR, "data")
PAPERS_DIR = os.path.join(DATA_DIR, "converted_pdfs")
OUTPUTS_DIR = os.path.join(DATA_DIR, "outputs")
