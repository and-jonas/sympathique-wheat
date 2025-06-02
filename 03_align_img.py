
# ======================================================================================================================
# Aligns ROIs from images in a series
# Author: Jonas Anderegg jonas.anderegg@usys.ethz.ch
# Last modified 2025-06-02
# ======================================================================================================================

# Define the year (marking strategy used)
year = 2023

def run():
    if year == 2024:
        from Processors.RoiAligner import RoiAligner2 as RoiAligner
        roi_aligner = RoiAligner(
            path_labels=f'raw/{year}/*/runs/pose/predict/labels',
            path_images=f'raw/{year}/*',
            path_leaf_masks=f'processed/*/predictions/symptoms_seg/pred',
            path_output='processed',
            n_cpus=4
    )
    elif year == 2023:
        from Processors.RoiAligner import RoiAligner
        roi_aligner = RoiAligner(
            path_labels=f'raw/{year}/*/runs/pose/predict/labels',
            path_images=f'raw/{year}/*',
            path_output='processed',
            n_cpus=4
        )
    roi_aligner.process_all()


if __name__ == "__main__":
    run()