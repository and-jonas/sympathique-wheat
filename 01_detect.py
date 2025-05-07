
# ======================================================================================================================
# Detects reference marks in images and exports corresponding image coordinates
# Author: Jonas Anderegg jonas.anderegg@usys.ethz.ch
# Last modified 2025-05-07
# ======================================================================================================================

from ultralytics import YOLO
from pathlib import Path
import glob
import math
import os

# load trained reference mark detection model
model = YOLO('models/best.pt')

# get all directories to process
dirs = [
    child.resolve()
    for year in ['2023', '2024']
    for child in (Path('data') / year).iterdir()
    if child.is_dir()
]

# Helper function: process a batch of images using the trained reference mark detection model
def process_batch(image_paths, model):
    results = model.predict(
        image_paths,
        conf=0.3,
        save_txt=True,
        save_conf=True,
    )

# batch images and process
def process_directory_in_batches(batch_size=32):
    image_paths = glob.glob('*.JPG')
    num_batches = math.ceil(len(image_paths) / batch_size)
    for i in range(num_batches):
        batch_paths = image_paths[i * batch_size:(i + 1) * batch_size]
        process_batch(batch_paths, model)

# ======================================================================================================================

def main():
    for d in dirs:
        print(f"Processing: {d}")
        os.chdir(d)
        process_directory_in_batches(batch_size=20)

if __name__ == "__main__":
    main()

# ======================================================================================================================