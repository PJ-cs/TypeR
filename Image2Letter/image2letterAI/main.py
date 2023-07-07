import os
from config import config
import requests
from pathlib import Path

def elt_data():
    """Extract, load and transform our data assets."""
    # Extract + Load
    url_file_paths: list[str] = [file.path for file in os.scandir(config.IMAGES_URL_DIR)]
    # courtesy to https://pyimagesearch.com/2017/12/04/how-to-create-a-deep-learning-dataset-using-google-images/
    for url_file in url_file_paths:
        links = open(url_file).read().strip().split("\n")
        output_path = Path(config.TRAINING_IMGS_DIR, os.path.basename(url_file)[:-4])
        total = 0
        for url in links:
            try:
                # try to download the image
                r = requests.get(url, timeout=60)
                # save the image to disk
                p = os.path.sep.join([output_path, "{}.jpg".format(
                    str(total).zfill(8))])
                f = open(p, "wb")
                f.write(r.content)
                f.close()
                # update the counter
                print("[INFO] downloaded: {}".format(p))
                total += 1
            # handle if any exceptions are thrown during the download process
            except:
                print("[INFO] error downloading {}...skipping".format(p))


elt_data()