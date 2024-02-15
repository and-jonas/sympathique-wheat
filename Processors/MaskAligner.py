
# ======================================================================================================================
# Aligns Masks using the estimated transformation from 02
# Author: Jonas Anderegg jonas.anderegg@usys.ethz.ch
# Last modified 2024-02-15
# ======================================================================================================================

from PIL import Image
import numpy as np
import cv2
import skimage
import pickle
import imageio
from pathlib import Path
import os
import json
import utils.base as base_utils


def transform_mask(base_dir, path_to_mask, n_classes, kpt_cls):

    print(path_to_mask)

    # get names
    base_name = os.path.basename(path_to_mask)
    jpg_name = base_name.replace(".png", ".JPG")
    stem_name = base_name.replace(".png", "")
    leaf_name = "_".join(stem_name.split("_")[2:4])

    # get mask
    mask = Image.open(path_to_mask)
    mask = np.asarray(mask)

    # get image
    image = f'{base_dir}/{leaf_name}/crop/{jpg_name}'
    img = Image.open(image)
    img = np.asarray(img)

    # get target (warped roi)
    target = Image.open(f"{base_dir}/{leaf_name}/result/piecewise/{stem_name}.JPG")
    target = np.asarray(target)

    # get bounding box localization info
    output_p = f"{base_dir}/{leaf_name}/roi/{stem_name}.json"
    f = open(output_p)
    data = json.load(f)
    rot = np.asarray(data['rotation_matrix'])
    bbox = np.asarray(data['bounding_box'])
    box = np.intp(bbox)

    tform_piecewise = None
    try:
        with open(f'{base_dir}/{leaf_name}/roi/{stem_name}_tform_piecewise.pkl', 'rb') as file:
            tform_piecewise = pickle.load(file)
    except FileNotFoundError:
        pass

    # get image roi
    full_img = np.zeros((5464, 8192, 3)).astype("uint8")
    mw, mh = map(int, np.mean(box, axis=0))
    full_img[mh-1024:mh+1024, :] = img
    rows, cols = full_img.shape[0], full_img.shape[1]

    # full mask
    full_mask = np.zeros((5464, 8192)).astype("uint8")
    full_mask[mh - 1024:mh + 1024, :] = mask

    # Transform the mask ===============================================================================================

    # remove points
    segmentation_mask = base_utils.remove_points_from_mask(mask=full_mask, classes=kpt_cls)

    # rotate mask
    segmentation_mask_rot = cv2.warpAffine(segmentation_mask, rot, (cols, rows))

    # crop roi
    roi = segmentation_mask_rot[box[0][1]:box[2][1], box[0][0]:box[1][0]]

    # warp roi (except for the first image in the series)
    tform = tform_piecewise
    if tform is not None:
        lm = np.stack([roi, roi, roi], axis=2)
        warped = skimage.transform.warp(lm, tform, output_shape=target.shape)
        warped = skimage.util.img_as_ubyte(warped[:, :, 0])
    else:
        warped = roi

    # Transform points, add to mask ====================================================================================

    # warp points
    if tform is not None:
        complete = base_utils.rotate_translate_warp_points(
            mask=full_mask,
            classes=kpt_cls,
            rot=rot,
            box=box,
            tf=tform,
            target_shape=target.shape,
            warped=warped,
        )
    else:
        complete = warped

    # Output ===========================================================================================================

    # transform to ease inspection
    complete = (complete.astype("uint32")) * 255 / n_classes
    complete = complete.astype("uint8")

    # save
    out_dir = Path(f'{base_dir}/{leaf_name}/mask_aligned/')
    out_dir_pw = out_dir / "piecewise"
    out_dir_pw.mkdir(exist_ok=True, parents=True)
    mask_name = f'{out_dir_pw}/{base_name}'
    imageio.imwrite(mask_name, complete)