
# ======================================================================================================================
# Aligns Masks using the estimated transformation from 02
# Author: Jonas Anderegg jonas.anderegg@usys.ethz.ch
# Last modified 2024-02-15
# ======================================================================================================================


from Processors.MaskAligner import transform_mask

import glob
import os
import multiprocessing


# if __name__ == '__main__':

# find all masks
base_dir = 'processed'
masks_seg = glob.glob(f'{base_dir}/*/predictions/symptoms_seg/pred/*.png')
masks_det = glob.glob(f'{base_dir}/*/predictions/symptoms_det/pred/*.png')

# list all samples for which the transformation was successful
existing_output = glob.glob(f'{base_dir}/*/result/piecewise/*.JPG')
b_names = [os.path.basename(x).replace(".JPG", "") for x in existing_output]

# list all that can be processed
masks_seg = [m for m in masks_seg if os.path.basename(m).replace(".png", "") in b_names]
masks_det = [m for m in masks_det if os.path.basename(m).replace(".png", "") in b_names]

# get number of samples to process
n = len(masks_seg)

# list tasks
base_dir = [base_dir] * n
n_classes = [6] * n
kpt_cls = [(5, 6)] * n
tasks = [*zip(base_dir, masks_seg, masks_det, n_classes, kpt_cls)]

# transform masks
num_processes = 1
with multiprocessing.Pool(processes=num_processes) as pool:
    pool.starmap(transform_mask, tasks)

