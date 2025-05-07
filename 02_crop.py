import os

# ======================================================================================================================
# Aligns ROIs from images in a series
# Author: Jonas Anderegg jonas.anderegg@usys.ethz.ch
# Last modified 2024-02-15
# ======================================================================================================================


# Define the marking strategy used
sidelines = True  # True if 2024, False if 2023

# import the corresponding ROIAligner
if sidelines:
    from Processors.RoiAligner import RoiAligner2 as RoiAligner
else:
    from Processors.RoiAligner import RoiAligner

def run():
    roi_aligner = RoiAligner(
        path_labels='data/2024/*/runs/pose/predict/labels',
        path_images='data/2024/*',
        path_leaf_masks=None,  # not available at this stage
        path_output='Output2',
        n_cpus=1
    )
    roi_aligner.process_all()


if __name__ == "__main__":
    run()
