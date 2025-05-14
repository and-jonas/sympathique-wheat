from leaf import models
from leaf.inference import Predictor

import glob

# models.test()

pred = Predictor(config_name='flattened_leaves',
                 symptoms_seg_params={'model_name': 'tracking_latest'},
                 symptoms_det_params={'model_name': 'latest',
                                      'keypoints_thresh': 0.18}
)

# list directories to process
dirs_to_process = glob.glob('Output/ESWW*')

# loop over directories
for d in dirs_to_process:
    print(d)
    pred.predict(images_src=f'{d}/crop', export_dst=f'{d}/predictions')
