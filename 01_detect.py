
# ======================================================================================================================
# Detects reference marks in images and exports corresponding image coordinates
# Author: Jonas Anderegg jonas.anderegg@usys.ethz.ch
# Last modified 2024-02-15
# ======================================================================================================================


from ultralytics import YOLO
import os
from pathlib import Path

# load trained reference mark detection model
model = YOLO('models/best.pt')

# list all directories to process
dirs = [str(child.resolve()) for child in Path.iterdir(Path('data'))]

# process all directories
for d in dirs:
    os.chdir(d)
    res = model.predict(
        "./",
        conf=0.3,
        save_txt=True,
    )

