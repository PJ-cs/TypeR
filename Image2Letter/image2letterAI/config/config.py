# config.py
from pathlib import Path
import json

TYPEWRITER_NAME = "SamsungSQ-1000"

# Directories
ROOT_DIR = Path(__file__).parent.parent.parent.absolute()
CONFIG_DIR = Path(ROOT_DIR, "image2letterAI", "config")
DATA_DIR = Path(ROOT_DIR, "data")
TYPEWRITER_DIR = Path(DATA_DIR, TYPEWRITER_NAME)
IMAGES_URL_DIR = Path(DATA_DIR, "trainingImagesURLs")
TRAINING_IMGS_DIR = Path(DATA_DIR, "trainingImages")

# Create dirs
DATA_DIR.mkdir(parents=True, exist_ok=True)
TYPEWRITER_DIR.mkdir(parents=True, exist_ok=True)
IMAGES_URL_DIR.mkdir(parents=True, exist_ok=True)
TRAINING_IMGS_DIR.mkdir(parents=True, exist_ok=True)

BLOB_STORE_URL = "gdrive://1xfnkAm5QsUc9l2H4pWZbaOO4P7DbGLv1"

# Paths
FONT_PATH = Path(TYPEWRITER_DIR, "font.otf")

# Typewriter config
f = open(Path(CONFIG_DIR, "typewriter_configs.json"))
TYPEWRITER_CONFIG = json.load(f)[TYPEWRITER_NAME]
f.close()