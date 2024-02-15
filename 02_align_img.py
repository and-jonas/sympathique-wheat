
# ======================================================================================================================
# Aligns ROIs from images in a series
# Author: Jonas Anderegg jonas.anderegg@usys.ethz.ch
# Last modified 2024-02-15
# ======================================================================================================================


from Processors.RoiAligner import RoiAligner


def run():
    roi_aligner = RoiAligner(
        path_labels='data/*/runs/pose/predict/labels',
        path_images='data/*',
        path_output='Output',
        n_cpus=6
    )
    roi_aligner.process_all()


if __name__ == "__main__":
    run()
