import os

# ======================================================================================================================
# Aligns ROIs from images in a series
# Author: Jonas Anderegg jonas.anderegg@usys.ethz.ch
# Last modified 2024-02-15
# ======================================================================================================================


# Define the year (marking strategy used)
year = 2024

def run():
    if year == 2024:
        from Processors.RoiAligner import RoiAligner2 as RoiAligner
        roi_aligner = RoiAligner(
            path_labels=f'data/{year}/*/runs/pose/predict3/labels',
            path_images=f'data/{year}/*',
            path_leaf_masks=None,  # not available at this stage
            path_output='Output',
            n_cpus=1
        )
        roi_aligner.process_all()
    elif year == 2023:
        print("Leaf masks not required. Skipping.")


if __name__ == "__main__":
    run()
