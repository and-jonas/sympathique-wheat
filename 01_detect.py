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
