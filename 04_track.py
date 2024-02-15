
from Processors.SymptomTracker import SymptomTracker


def run():
    symptom_tracker = SymptomTracker(
        path_aligned_masks='Output/*/mask_aligned/piecewise',
        path_images='Output/*/result/piecewise',
        path_kpts='Output/*/keypoints',
        path_output='Output/ts',
        n_cpus=6,
    )
    symptom_tracker.process_all()


if __name__ == "__main__":
    run()


