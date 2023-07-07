# config.py
from pathlib import Path

# Directories
BASE_DIR = Path(__file__).parent.parent.absolute()
CONFIG_DIR = Path(BASE_DIR, "config")
DATA_DIR = Path(BASE_DIR, "data")
TYPEWRITER_DIR = Path(DATA_DIR, "SamsungSQ-1000")
IMAGES_URL_DIR = Path(DATA_DIR, "trainingImagesURLs")
TRAINING_IMGS_DIR = Path(DATA_DIR, "trainingImages")

# Create dirs
DATA_DIR.mkdir(parents=True, exist_ok=True)
TYPEWRITER_DIR.mkdir(parents=True, exist_ok=True)
IMAGES_URL_DIR.mkdir(parents=True, exist_ok=True)
TRAINING_IMGS_DIR.mkdir(parents=True, exist_ok=True)

BLOB_STORE_URL = "gdrive://1xfnkAm5QsUc9l2H4pWZbaOO4P7DbGLv1"