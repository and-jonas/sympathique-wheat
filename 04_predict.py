
# ======================================================================================================================
# Detects and segments disease symptoms in image crops
# Author: Jonas Anderegg jonas.anderegg@usys.ethz.ch
# Last modified 2025-06-02
# ======================================================================================================================

from leaf import models
from leaf.inference import Predictor
from leaf.visualization import FlattenedVisualizer

import glob

# models.test()

pred = Predictor(config_name='flattened_leaves',
                 symptoms_seg_params={'model_name': 'tracking_latest'},
                 symptoms_det_params={'model_name': 'tracking_latest',
                                      'keypoints_thresh': 0.18}
)

# list directories to process
dirs_to_process = glob.glob('processed/ESWW*')

# loop over directories
# predict and visualize
for d in dirs_to_process:
    print(d)
    pred.predict(images_src=f'{d}/crop', export_dst=f'{d}/predictions')
    # vis = FlattenedVisualizer(src_root=f'{d}/predictions', rgb_root=f'{d}/crop', export_root=f'{d}/predictions')
    # vis.visualize()
