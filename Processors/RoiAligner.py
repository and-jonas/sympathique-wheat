
# ======================================================================================================================
# Aligns ROIs from images in a series
# Author: Jonas Anderegg jonas.anderegg@usys.ethz.ch
# Last modified 2024-02-15
# ======================================================================================================================

# import libraries
from utils import base as base_utils
from utils import Stitcher
import numpy as np
import json
import pandas as pd
import pickle
from PIL import Image
import glob
import os
from pathlib import Path
import copy
import skimage
from skimage import transform
import cv2
from natsort import natsorted
from multiprocessing import Manager, Process


class RoiAligner:

    def __init__(self, path_labels, path_images, path_output, n_cpus):
        self.path_labels = Path(path_labels)
        self.path_images = Path(path_images)
        self.path_output = Path(path_output)
        self.n_cpus = n_cpus

    def prepare_workspace(self):
        """
        Creates all required output directories
        """
        self.path_output.mkdir(parents=True, exist_ok=True)
        with open(f"{self.path_output}/unmatched_projective.txt", 'w') as file:
            pass
        with open(f"{self.path_output}/unmatched_piecewise.txt", 'w') as file:
            pass

    def get_output_paths(self, label_series, create_dirs):
        """
        Creates all required output paths and makes the corresponding directories
        :param label_series: The series to process
        :param create_dirs: Wether or not to create directories
        :return: All required paths
        """
        sample_id = "_".join(os.path.basename(label_series).split("_")[2:4]).replace(".txt", "")
        # generate output paths for each sample and create directories
        sample_output_path = self.path_output / sample_id
        kpts_path = sample_output_path / "keypoints"
        overlay_path = sample_output_path / "overlay"
        roi_path = sample_output_path / "roi"
        result_path = sample_output_path / "result"
        result_pw = result_path / "piecewise"
        result_proj = result_path / "projective"
        preview_path = sample_output_path / "preview"
        crop_path = sample_output_path / "crop"
        if create_dirs:
            for p in (kpts_path, overlay_path, roi_path, result_pw, result_proj, preview_path, crop_path):
                p.mkdir(parents=True, exist_ok=True)
        return kpts_path, overlay_path, roi_path, result_pw, result_proj, preview_path, crop_path

    def log_fail(self, image_id, t_type, reason):
        """
        Writes the image id of failure casees to a txt file
        :param image_id: if of the current image
        :param t_type: the type of transformation attempted ("piecewise" or "projective" or "polynomial")
        :param reason: Reason for the failure; chr string
        """
        f = open(f"{self.path_output}/unmatched_{t_type}.txt", 'a')
        f.writelines(image_id + " " + str(reason) + "\n")
        f.close()

    def get_series(self):
        """
        Creates two lists of file paths: to key point coordinate files and to images
        for each of the samples monitored over time, stored in date-wise folders.
        :return:
        """
        label_series = []
        image_series = []

        labels = glob.glob(f'{self.path_labels}/*.txt')
        images = glob.glob(f'{self.path_images}/*.JPG')
        label_image_id = ["_".join(os.path.basename(l).split("_")[2:4]).replace(".txt", "") for l in labels]
        image_image_id = ["_".join(os.path.basename(l).split("_")[2:4]).replace(".JPG", "") for l in images]
        uniques = np.unique(label_image_id)

        if len(images) != len(labels):
            raise Exception("list of images and list of coordinate files are not of equal length.")

        print("found " + str(len(uniques)) + " unique sample names")

        for unique_sample in uniques:
            image_idx = [index for index, image_id in enumerate(image_image_id) if unique_sample == image_id]
            label_idx = [index for index, label_id in enumerate(label_image_id) if unique_sample == label_id]
            sample_image_names = [images[i] for i in image_idx]
            sample_labels = [labels[i] for i in label_idx]
            # sort to ensure sequential processing of subsequent images
            sample_image_names = sorted(sample_image_names, key=lambda i: os.path.splitext(os.path.basename(i))[0])
            sample_labels = sorted(sample_labels, key=lambda i: os.path.splitext(os.path.basename(i))[0])
            label_series.append(sample_labels)
            image_series.append(sample_image_names)

        return label_series, image_series

    def process_series(self, work_queue, result):
        """
        Processes the image series for one sample.
        :param work_queue:
        :param result:
        """
        for job in iter(work_queue.get, 'STOP'):

            l_series = job["lseries"]
            i_series = job["iseries"]

            # check that there are an equal number of images and coordinate files
            if len(l_series) != len(i_series):
                print("label series and image series are not of equal length!")
                break

            # iterate over all samples in the series
            roi_widths = []
            for j in range(len(l_series)):

                # prepare sample work space
                image_id = os.path.basename(l_series[j]).replace(".txt", "")
                out_paths = self.get_output_paths(label_series=l_series[j], create_dirs=True)
                kpts_path, overlay_path, roi_path, result_pw, result_proj, preview_path, crop_path = out_paths

                print(image_id)

                # get key point coordinates from YOLO output
                coords = pd.read_table(l_series[j], header=None, sep=" ")
                x = coords.iloc[:, 5] * 8192
                y = coords.iloc[:, 6] * 5464

                # get image
                img = Image.open(i_series[j])
                img = np.array(img)

                # remove double detections
                # TODO can this be done during inference via non maximum suppression
                point_list, x, y = base_utils.remove_double_detections(x=x, y=y, tol=50)

                # remove outliers in the key point detections from YOLO errors,
                outliers_x = base_utils.reject_outliers(x, tol=None, m=3.)  # larger extension, larger variation
                outliers_y = base_utils.reject_outliers(y, tol=None, m=2.5)  # smaller extension, smaller variation
                outliers = outliers_x + outliers_y
                point_list = np.delete(point_list, outliers, 0)

                # if too few points detected, skip
                if len(point_list) < 7:
                    print("Insufficient marks detected. Skipping. ")
                    continue

                # get minimum area rectangle around retained key points
                rect = cv2.minAreaRect(point_list)

                # rotate the image about its center
                (center, (w, h), angle) = rect
                if angle > 45:
                    angle = angle - 90
                rows, cols = img.shape[0], img.shape[1]
                M_img = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
                img_rot = cv2.warpAffine(img, M_img, (cols, rows))

                # rotate the bounding box about the image's center
                M_box = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
                box = cv2.boxPoints(rect)
                pts = np.intp(cv2.transform(np.array([box]), M_box))[0]
                pts[pts < 0] = 0

                # order bbox points clockwise
                pts = base_utils.order_points(pts)

                # record roi localization
                roi_loc = {'rotation_matrix': M_img.tolist(), 'bounding_box': pts.tolist()}

                # make crop to run inference on
                img_cropped = base_utils.make_inference_crop(pts, img)
                cv2.imwrite(f'{crop_path}/{image_id}.JPG', cv2.cvtColor(img_cropped, cv2.COLOR_BGR2RGB))

                # draw key points and bounding box on overlay image as check
                overlay = base_utils.make_bbox_overlay(img, point_list, box)
                cv2.imwrite(f'{overlay_path}/{image_id}.JPG', cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))

                # crop the roi from the rotated image
                img_crop = img_rot[pts[0][1]:pts[2][1], pts[0][0]:pts[1][0]]

                # log roi width to detect outliers in series
                roi_widths.append(img_crop.shape[1])
                if j == 0:
                    init_roi_height = img_crop.shape[0]

                # detect size outliers and remove from the logged bbox width values
                if j != 0:
                    size_outliers = base_utils.reject_size_outliers(roi_widths, max_diff=125)
                    if size_outliers:
                        del roi_widths[-1]

                # copy image for later use
                save_img = copy.copy(img_crop)

                # rotate and translate key point coordinates
                kpts = np.intp(cv2.transform(np.array([point_list]), M_img))[0]

                # get tx and ty values for key point translation
                tx, ty = (-pts[0][0], -pts[0][1])
                translation_matrix = np.array([
                    [1, 0, tx],
                    [0, 1, ty]
                ], dtype=np.float32)

                # apply  translation to key points
                kpts = np.intp(cv2.transform(np.array([kpts]), translation_matrix))[0]

                # remove outliers from rotated coordinates,
                # separately for top and bottom row of marks
                t, b = base_utils.identify_outliers_2d(pts=kpts, tol=100, m=3)
                kpts = np.vstack([t, b])

                # export a preview
                preview = base_utils.make_bbox_overlay(img_crop, kpts, box=None)
                cv2.imwrite(f'{preview_path}/{image_id}.JPG', cv2.cvtColor(preview, cv2.COLOR_BGR2RGB))

                # match all images in the series to the first image where possible
                # For the first image of each series, no further processing is needed
                if j == 0:
                    kpts_ref = kpts
                    # kpts_norm_ref = kpts_norm
                    cv2.imwrite(f'{result_pw}/{image_id}.JPG', cv2.cvtColor(save_img, cv2.COLOR_BGR2RGB))
                    cv2.imwrite(f'{result_proj}/{image_id}.JPG', cv2.cvtColor(save_img, cv2.COLOR_BGR2RGB))
                # For all subsequent images, perform image registration with the first image as target
                elif j > 0:
                    # match key points with those on the first image of the series
                    # by searching for the closest points
                    # if non is found in proximity, eliminate from both images
                    # this must be done separately for top and bottom marks to avoid cross-associations
                    kpts_top, kpts_bottom = base_utils.separate_marks(pts=kpts)
                    kpts_ref_top, kpts_ref_bottom = base_utils.separate_marks(kpts_ref)
                    src_top, dst_top = base_utils.find_keypoint_matches(
                        current=kpts_top, current_orig=kpts_top, ref=kpts_ref_top, dist_limit=250)
                    src_bot, dst_bot = base_utils.find_keypoint_matches(
                        current=kpts_bottom, current_orig=kpts_bottom, ref=kpts_ref_bottom, dist_limit=250)
                    src = src_top + src_bot
                    dst = dst_top + dst_bot

                    # verify that matches are spatially reasonable; remove outlier associations
                    # if too few points detected, skip
                    src, dst = base_utils.check_keypoint_matches(src=src, dst=dst, mdev=50, tol=100, m=2.5)

                    # if there are few matches, or if there is a different roi size from the expected,
                    # there is likely a translation due to key point detection errors
                    # try to match with the previous image in the series using SIFT + RANSAC
                    match_thresh = int(0.95*len(kpts))
                    if len(src) < match_thresh or size_outliers:
                        if len(src) < match_thresh:
                            print("Key point mis-match. Matching on last image in series.")
                            print(image_id)
                        if size_outliers:
                            print("Size outlier detected. Matching on last image in series.")

                        # go back (max 4 time steps) in the series until success
                        aligned = False
                        i = 1
                        while not aligned and i < 4:
                            try:
                                # load previous image
                                prev_image_id = os.path.basename(l_series[j - i]).replace(".txt", "")
                                previous_image = Image.open(f'{result_pw}/{prev_image_id}.JPG')
                            except FileNotFoundError:
                                i += 1
                                continue

                            previous_image = np.asarray(previous_image)
                            current_image = save_img

                            # adjust size by padding if needed; images must have equal height for stitching
                            (w1, h1, _) = previous_image.shape
                            (w2, h2, _) = current_image.shape
                            if w2 > w1:
                                previous_image = cv2.copyMakeBorder(previous_image, 0, w2 - w1, 0, 0, cv2.BORDER_CONSTANT)
                            elif w1 > w2:
                                current_image = cv2.copyMakeBorder(current_image, 0, w1 - w2, 0, 0, cv2.BORDER_CONSTANT)

                            # try stitching images using SIFT and RANSAC
                            stitcher = Stitcher.Stitcher()
                            try:
                                result, vis, H = stitcher.stitch(
                                    images=[copy.copy(previous_image), copy.copy(current_image)],
                                    masks=[None, None],
                                    showMatches=True,
                                )

                            except TypeError:
                                self.log_fail(image_id, t_type="projective", reason="no_H")
                                i += 1
                                continue

                            # warp key points by applying the inverse of the homography matrix
                            try:
                                kpts_warped = [base_utils.warp_point(x[0], x[1], np.linalg.inv(H)) for x in kpts]
                            except:
                                self.log_fail(image_id, t_type="projective", reason="no_warped")
                                i += 1
                                continue

                            # search for matches with sufficient tolerance
                            src, dst = base_utils.find_keypoint_matches(
                                current=kpts_warped, current_orig=kpts, ref=kpts_ref, dist_limit=200)

                            # verify that matches are spatially reasonable; remove outlier associations
                            src, dst = base_utils.check_keypoint_matches(src=src, dst=dst, mdev=None, tol=100, m=2.5)

                            i += 1

                            # check if now aligned and exit loop
                            if len(src) > 7:
                                aligned = True

                    # write warped key point coordinates to file for eventual later FINAL roi determination
                    try:
                        src = np.asarray(src)
                        data = {"x": src[:, 0], "y": src[:, 1]}
                        df = pd.DataFrame(data)
                        df.to_csv(f'{kpts_path}/{image_id}.txt', index=False)
                    except IndexError:
                        pass

                    # Transform with the first image of the series as destination
                    # PROJECTIVE
                    tform_projective = transform.ProjectiveTransform()
                    try:
                        tform_projective.estimate(src, dst)
                    except:
                        self.log_fail(image_id, t_type="projective", reason="transform_fail")
                        continue
                    # PIECEWISE
                    tform_piecewise = transform.PiecewiseAffineTransform()
                    try:
                        tform_piecewise.estimate(src, dst)
                    except:
                        self.log_fail(image_id, t_type="piecewise", reason="transform_fail")
                        continue

                    # Save
                    # PROJECTIVE
                    roi_loc['transformation_matrix'] = tform_projective.params.tolist()
                    # projective_warped = skimage.transform.warp(save_img, tform_projective,
                    #                                            output_shape=(init_roi_height, roi_widths[0]))
                    # projective_warped = skimage.util.img_as_ubyte(projective_warped)
                    # cv2.imwrite(f'{result_proj}/{image_id}.JPG', cv2.cvtColor(projective_warped, cv2.COLOR_BGR2RGB))
                    # PIECEWISE
                    with open(f'{roi_path}/{image_id}_tform_piecewise.pkl', 'wb') as file:
                        pickle.dump(tform_piecewise, file)
                    file.close()
                    piecewise_warped = transform.warp(save_img, tform_piecewise,
                                                      output_shape=(init_roi_height, roi_widths[0]))
                    piecewise_warped = skimage.util.img_as_ubyte(piecewise_warped)
                    cv2.imwrite(f'{result_pw}/{image_id}.JPG', cv2.cvtColor(piecewise_warped, cv2.COLOR_BGR2RGB))

                    del size_outliers

                    # add transformation matrix to the roi localization info
                    roi_loc['transformation_matrix'] = tform_projective.params.tolist()

                with open(f'{roi_path}/{image_id}.json', 'w') as outfile:
                    json.dump(roi_loc, outfile)

    def process_all(self):

        self.prepare_workspace()
        label_series, image_series = self.get_series()

        if len(label_series) > 0:
            # make job and results queue
            m = Manager()
            jobs = m.Queue()
            results = m.Queue()
            processes = []
            # Progress bar counter
            max_jobs = len(label_series)
            count = 0

            # Build up job queue
            for lseries, iseries in zip(label_series, image_series):
                print("to queue")
                job = dict()
                job['lseries'] = lseries
                job['iseries'] = iseries
                jobs.put(job)

            # Start processes
            for w in range(self.n_cpus):
                p = Process(target=self.process_series,
                            args=(jobs, results))
                p.daemon = True
                p.start()
                processes.append(p)
                jobs.put('STOP')

            print(str(len(label_series)) + " jobs started, " + str(self.n_cpus) + " workers")

            # Get results and increment counter along with it
            while count < max_jobs:
                img_names = results.get()
                count += 1
                print("processed " + str(count) + "/" + str(max_jobs))

            for p in processes:
                p.join()


class RoiAligner2:


    def __init__(self, path_labels, path_images, path_leaf_masks, path_output, n_cpus, path_sample_list=None):
        self.path_labels = Path(path_labels)
        self.path_images = Path(path_images)
        self.path_leaf_masks = Path(path_leaf_masks) if path_leaf_masks else None
        self.path_output = Path(path_output)
        self.path_sample_list = Path(path_sample_list) if path_sample_list else None
        self.n_cpus = n_cpus

    def prepare_workspace(self):
        """
        Creates all required output directories
        """
        self.path_output.mkdir(parents=True, exist_ok=True)
        with open(f"{self.path_output}/unmatched_projective.txt", 'w') as file:
            pass
        with open(f"{self.path_output}/unmatched_piecewise.txt", 'w') as file:
            pass

    def get_output_paths(self, label_series, create_dirs):
        """
        Creates all required output paths and makes the corresponding directories
        :param label_series: The series to process
        :param create_dirs: Wether or not to create directories
        :return: All required paths
        """
        sample_id = "_".join(os.path.basename(label_series).split("_")[2:4]).replace(".txt", "")
        # generate output paths for each sample and create directories
        sample_output_path = self.path_output / sample_id
        kpts_path = sample_output_path / "keypoints"
        overlay_path = sample_output_path / "overlay"
        roi_path = sample_output_path / "roi"
        result_path = sample_output_path / "result"
        result_pw = result_path / "piecewise"
        result_proj = result_path / "projective"
        preview_path = sample_output_path / "preview"
        crop_path = sample_output_path / "crop"
        leaf_mask_path = sample_output_path / "leaf_mask"
        if create_dirs:
            for p in (kpts_path, overlay_path, roi_path, result_pw, result_proj, preview_path, crop_path, leaf_mask_path):
                p.mkdir(parents=True, exist_ok=True)
        return kpts_path, overlay_path, roi_path, result_pw, result_proj, preview_path, crop_path, leaf_mask_path

    def log_fail(self, image_id, type, reason):
        """
        Writes the image id of failure casees to a txt file
        :param image_id: if of the current image
        :param type: the type of transformation attempted ("piecewise" of "projective")
        """
        f = open(f"{self.path_output}/unmatched_{type}.txt", 'a')
        f.writelines(image_id + " " + str(reason) + "\n")
        f.close()

    def get_series(self, sample_list=None):
        """
        Creates two lists of file paths: to key point coordinate files and to images
        for each of the samples monitored over time, stored in date-wise folders.
        :return:
        """
        label_series = []
        image_series = []

        labels = glob.glob(f'{self.path_labels}/*.txt')
        images = glob.glob(f'{self.path_images}/*.JPG')
        label_image_id = ["_".join(os.path.basename(l).split("_")[2:4]).replace(".txt", "") for l in labels]
        image_image_id = ["_".join(os.path.basename(l).split("_")[2:4]).replace(".JPG", "") for l in images]
        uniques = natsorted(np.unique(label_image_id))

        if len(images) != len(labels):
            raise Exception("list of images and list of coordinate files are not of equal length.")
        print("found " + str(len(uniques)) + " unique sample names")

        # print info on processing
        if not self.path_leaf_masks:
            print("No leaf masks provided. Generting inference crop and skipping Alignment")

        # if a file list is provided select the subset
        if self.path_sample_list is not None:
            file_list = pd.read_csv(self.path_sample_list, header=None, sep=" ")
            sample_list = file_list.iloc[:, 0]
            s_list = ["_".join(os.path.basename(l).split("_")[2:4]).replace(".txt", "") for l in sample_list]
            uniques = np.unique(s_list)
            print("processing " + str(len(uniques)) + " unique sample names")

        # compile the lists
        for unique_sample in uniques:
            image_idx = [index for index, image_id in enumerate(image_image_id) if unique_sample == image_id]
            label_idx = [index for index, label_id in enumerate(label_image_id) if unique_sample == label_id]
            sample_image_names = [images[i] for i in image_idx]
            sample_labels = [labels[i] for i in label_idx]
            # sort to ensure sequential processing of subsequent images
            sample_image_names = sorted(sample_image_names, key=lambda i: os.path.splitext(os.path.basename(i))[0])
            sample_labels = sorted(sample_labels, key=lambda i: os.path.splitext(os.path.basename(i))[0])
            label_series.append(sample_labels)
            image_series.append(sample_image_names)

        return label_series, image_series

    def process_series(self, work_queue, result):
        """
        Processes the image series for one sample.
        :param work_queue:
        :param result:
        """
        for job in iter(work_queue.get, 'STOP'):

            l_series = job["lseries"]
            i_series = job["iseries"]

            # check that there are an equal number of images and coordinate files
            if len(l_series) != len(i_series):
                print("label series and image series are not of equal length!")
                break

            # iterate over all samples in the series
            roi_widths = []
            for j in range(len(l_series)):


                try:

                    # prepare sample work space
                    image_id = os.path.basename(l_series[j]).replace(".txt", "")

                    # print(image_id)
                    # if image_id == "20240625_091351_ESWW0090023_11":
                    #     print("stop")

                    out_paths = self.get_output_paths(label_series=l_series[j], create_dirs=True)
                    kpts_path, overlay_path, roi_path, result_pw, result_proj, preview_path, crop_path, leaf_mask_path = out_paths

                    # get key point coordinates from YOLO output
                    coords = pd.read_table(l_series[j], header=None, sep=" ")
                    x = coords.iloc[:, 5] * 8192
                    y = coords.iloc[:, 6] * 5464

                    # get image
                    img = Image.open(i_series[j])
                    img = np.array(img)

                    # remove double detections
                    point_list, x, y = base_utils.remove_double_detections(x=x, y=y, tol=50)

                    # remove outliers in the key point detections from YOLO errors
                    outliers_x = base_utils.reject_outliers(x, tol=None, m=3.)  # larger extension, larger variation
                    outliers_y = base_utils.reject_outliers(y, tol=None, m=2.5)  # smaller extension, smaller variation
                    outliers = outliers_x + outliers_y
                    point_list = np.delete(point_list, outliers, 0)

                    # if too few points detected, skip
                    if len(point_list) < 7:
                        print("Insufficient marks detected. Skipping. ")
                        continue

                    # filter outliers in x
                    if j == 0:
                        point_list = base_utils.filter_points_x(point_list, image=img)

                    # get minimum area rectangle around retained key points
                    rect = cv2.minAreaRect(point_list)

                    # enlarge to enable feature extraction for 56 px square box around detected markers
                    (center, (w, h), angle) = rect
                    rect = (center, (w+224, h+224), angle)

                    # rotate the image about its center
                    if angle > 45:
                        angle = angle - 90
                    rows, cols = img.shape[0], img.shape[1]
                    M_img = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
                    img_rot = cv2.warpAffine(img, M_img, (cols, rows))

                    # rotate the bounding box about the image's center
                    M_box = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
                    box = cv2.boxPoints(rect)
                    pts = np.intp(cv2.transform(np.array([box]), M_box))[0]
                    pts[pts < 0] = 0

                    # order bbox points clockwise
                    pts = base_utils.order_points(pts)

                    # record roi localization
                    roi_loc = {'rotation_matrix': M_img.tolist(), 'bounding_box': pts.tolist()}

                    # make crop to run inference on
                    img_cropped = base_utils.make_inference_crop(pts, img)
                    cv2.imwrite(f'{crop_path}/{image_id}.JPG', cv2.cvtColor(img_cropped, cv2.COLOR_BGR2RGB))

                    # if no leaf masks are available, skip alignment
                    if not self.path_leaf_mask:
                        continue

                    # draw key points and bounding box on overlay image as check
                    overlay = base_utils.make_bbox_overlay(img, point_list, box)
                    cv2.imwrite(f'{overlay_path}/{image_id}.JPG', cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))

                    # crop the roi from the rotated image
                    img_crop = img_rot[pts[0][1]:pts[2][1], pts[0][0]:pts[1][0]]
                    current_roi_width = img_crop.shape[1]

                    # log roi width to detect outliers in series
                    size_outliers = None
                    roi_widths.append(current_roi_width)
                    if j == 0:
                        init_roi_height, init_roi_width = img_crop.shape[:2]

                    # detect size outliers and remove from the logged bbox width values
                    if j != 0:
                        size_outliers = base_utils.reject_size_outliers(data=roi_widths, max_diff=200)
                        if size_outliers is not None:
                            del roi_widths[-1]

                    # copy image for later use
                    save_img = copy.copy(img_crop)

                    # rotate and translate key point coordinates
                    kpts = np.intp(cv2.transform(np.array([point_list]), M_img))[0]

                    # get tx and ty values for key point translation
                    tx, ty = (-pts[0][0], -pts[0][1])
                    translation_matrix = np.array([
                        [1, 0, tx],
                        [0, 1, ty]
                    ], dtype=np.float32)

                    # apply  translation to key points
                    kpts = np.intp(cv2.transform(np.array([kpts]), translation_matrix))[0]

                    # remove outliers from rotated coordinates,
                    # separately for top, bottom and edge marks
                    reference = True if j == 0 else False
                    w_ref = None if j == 0 else w_ref
                    l, r, t, b, w = base_utils.identify_outliers_2d_(
                        pts=kpts,
                        tol=100,
                        m=3,
                        w_ref=w_ref,
                        reference=reference
                    )
                    w_ref = w if j == 0 else w_ref
                    kpts = l, r, t, b

                    # load leaf mask to measure distance of l, r points from leaf mask
                    path_leaf_mask = f'{self.path_leaf_masks}/{image_id}.png'
                    mask = base_utils.process_leaf_mask(path_leaf_mask=path_leaf_mask)
                    cv2.imwrite(f'{leaf_mask_path}/{image_id}.JPG', cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
                    # reconstruct full mask
                    mw, mh = map(int, np.mean(pts, axis=0))
                    full_mask = np.zeros((5464, 8192)).astype("uint8")
                    full_mask[mh - 1024:mh + 1024, :] = mask
                    mask_rot = cv2.warpAffine(full_mask, M_img, (cols, rows))  # rotate mask
                    mask = mask_rot[pts[0][1]:pts[2][1], pts[0][0]:pts[1][0]]  # crop to roi

                    # export a preview
                    preview = base_utils.make_bbox_overlay(img=img_crop, pts=(l, r, t, b), box=None)
                    cv2.imwrite(f'{preview_path}/{image_id}.JPG', cv2.cvtColor(preview, cv2.COLOR_BGR2RGB))

                    # get relative x-positions of the edge marks
                    dist = base_utils.get_leaf_edge_distances(pts=(l, r), leaf_mask=mask)

                    # match all images in the series to the first image where possible
                    # For the first image of each series, no further processing is needed
                    if j == 0:
                        kpts_ref = kpts
                        dist_ref = dist
                        w_ref = w
                        cv2.imwrite(f'{result_pw}/{image_id}.JPG', cv2.cvtColor(save_img, cv2.COLOR_BGR2RGB))
                        cv2.imwrite(f'{result_proj}/{image_id}.JPG', cv2.cvtColor(save_img, cv2.COLOR_BGR2RGB))
                        roi_widths[0] = int(w)

                    # For all subsequent images, perform image registration with the first image as target
                    elif j > 0:
                        # match key points with those on the first image of the series
                        # by searching for the closest points in the top and bottom rows
                        # if non is found in proximity, eliminate from both images
                        src1, dst1 = base_utils.find_keypoint_matches(
                            current=kpts,
                            current_orig=kpts,
                            ref=kpts_ref,
                            dist_limit=250,
                        )
                        # match key points with those on the first image of the series
                        # by searching for the closest points in the left and right marks
                        # if non is found in proximity, eliminate from both images

                        src2, dst2 = base_utils.find_distance_matches(
                            current=dist,
                            ref=dist_ref,
                            c_kpt=kpts,
                            r_kpt=kpts_ref,
                            rel_limit=0.15,
                        )
                        # verify that matches are spatially reasonable; remove outlier associations
                        # if too few points detected, skip
                        src1, dst1 = base_utils.check_keypoint_matches(src=src1, dst=dst1, mdev=50, tol=100, m=2.5)

                        # combine all (l, r, t, b)
                        src = src2 + src1
                        dst = dst2 + dst1

                        # if there are few matches, or if there is a different roi size from the expected,
                        # there is likely a translation due to key point detection errors
                        # try to match with the previous image in the series using SIFT + RANSAC
                        n_kpts = np.sum([len(kpts[i]) for i in range(len(kpts))])
                        n_matches = np.sum([len(src[i]) for i in range(len(src))])
                        match_thresh = int(0.95*n_kpts)
                        if n_matches < match_thresh or np.abs(w-w_ref) > 200:
                            if n_matches < match_thresh:
                                print(" ----- Key point mis-match. Matching on last image in series.")
                            if np.abs(w-w_ref)>200:
                                print(" ----- Size outlier detected. Matching on last image in series.")

                            # go back (max 4 time steps) in the series until success
                            aligned = False
                            i = 1
                            while not aligned and i < 4:
                                try:
                                    # load previous image
                                    prev_image_id = os.path.basename(l_series[j - i]).replace(".txt", "")
                                    previous_image = Image.open(f'{result_proj}/{prev_image_id}.JPG')
                                except FileNotFoundError:
                                    i += 1
                                    continue

                                previous_image = np.asarray(previous_image)
                                current_image = save_img

                                # adjust size by padding if needed; images must have equal height for stitching
                                (w1, h1, _) = previous_image.shape
                                (w2, h2, _) = current_image.shape
                                if w2 > w1:
                                    previous_image = cv2.copyMakeBorder(previous_image, 0, w2 - w1, 0, 0, cv2.BORDER_CONSTANT)
                                elif w1 > w2:
                                    current_image = cv2.copyMakeBorder(current_image, 0, w1 - w2, 0, 0, cv2.BORDER_CONSTANT)

                                # try stitching images using SIFT and RANSAC
                                stitcher = Stitcher()
                                try:
                                    res, vis, H = stitcher.stitch(
                                        images=[copy.copy(previous_image), copy.copy(current_image)],
                                        masks=[None, None],
                                        showMatches=True,
                                    )

                                except TypeError:
                                    self.log_fail(image_id, type="projective", reason="no_H")
                                    i += 1
                                    continue

                                # warp key points by applying the inverse of the homography matrix
                                try:
                                    kpts_warped = []
                                    for type in kpts:
                                        kpts_warped.append(np.array([base_utils.warp_point(x[0], x[1], np.linalg.inv(H)) for x in type]))
                                except:
                                    self.log_fail(image_id, type="projective", reason="no_warped")
                                    i += 1
                                    continue

                                # match key points with those on the first image of the series
                                # by searching for the closest points in the top and bottom rows
                                # if non is found in proximity, eliminate from both images
                                src1, dst1 = base_utils.find_keypoint_matches(
                                    current=kpts_warped,
                                    current_orig=kpts,
                                    ref=kpts_ref,
                                    dist_limit=250
                                )

                                # match key points with those on the first image of the series
                                # by searching for the closest points in the left and right marks
                                # if non is found in proximity, eliminate from both images
                                src2, dst2 = base_utils.find_distance_matches(
                                    current=dist,
                                    ref=dist_ref,
                                    c_kpt=kpts,
                                    r_kpt=kpts_ref,
                                    rel_limit=0.1
                                )
                                # verify that matches are spatially reasonable; remove outlier associations
                                # if too few points detected, skip
                                src1, dst1 = base_utils.check_keypoint_matches(src=src1, dst=dst1, mdev=50, tol=100, m=2.5)

                                # combine all (l, r, t, b)
                                src = src2 + src1
                                dst = dst2 + dst1

                                i += 1

                                # check if now aligned and exit loop
                                if np.sum([len(src[i]) for i in range(len(src))]) > 7:
                                    aligned = True

                        # write warped key point coordinates to file for eventual later FINAL roi determination
                        src_filtered = [np.array(lst) for lst in src if lst]
                        dst_filtered = [np.array(lst) for lst in dst if lst]
                        src_ = np.vstack(src_filtered)
                        dst_ = np.vstack(dst_filtered)
                        try:
                            data = {"x": src_[:, 0], "y": src_[:, 1]}
                            df = pd.DataFrame(data)
                            df.to_csv(f'{kpts_path}/{image_id}.txt', index=False)
                        except IndexError:
                            pass

                        # Transform with the first image of the series as destination
                        # PROJECTIVE
                        tform_projective = transform.ProjectiveTransform()
                        try:
                            tform_projective.estimate(src_, dst_)
                        except:
                            self.log_fail(image_id, type="projective", reason="transform_fail")
                            continue
                        # PIECEWISE
                        tform_piecewise = transform.PiecewiseAffineTransform()
                        try:
                            tform_piecewise.estimate(src_, dst_)
                        except:
                            self.log_fail(image_id, type="piecewise", reason="transform_fail")
                            continue

                        # Save
                        # PROJECTIVE
                        roi_loc['transformation_matrix'] = tform_projective.params.tolist()
                        projective_warped = skimage.transform.warp(save_img, tform_projective,
                                                                   output_shape=(init_roi_height, init_roi_width))
                        projective_warped = skimage.util.img_as_ubyte(projective_warped)
                        cv2.imwrite(f'{result_proj}/{image_id}.JPG', cv2.cvtColor(projective_warped, cv2.COLOR_BGR2RGB))
                        # PIECEWISE
                        with open(f'{roi_path}/{image_id}_tform_piecewise.pkl', 'wb') as file:
                            pickle.dump(tform_piecewise, file)
                        file.close()
                        piecewise_warped = transform.warp(save_img, tform_piecewise,
                                                          output_shape=(init_roi_height, init_roi_width))
                        piecewise_warped = skimage.util.img_as_ubyte(piecewise_warped)
                        cv2.imwrite(f'{result_pw}/{image_id}.JPG', cv2.cvtColor(piecewise_warped, cv2.COLOR_BGR2RGB))

                        size_outliers = None

                        # add transformation matrix to the roi localization info
                        roi_loc['transformation_matrix'] = tform_projective.params.tolist()

                    with open(f'{roi_path}/{image_id}.json', 'w') as outfile:
                        json.dump(roi_loc, outfile)

                except:
                    self.log_fail(image_id, type="all", reason="unknown")
                    continue

            # monitor progress
            sample_identifier = "_".join(os.path.basename(l_series[0]).replace(".txt", "").split("_")[-2:])
            result.put(sample_identifier)

    def process_all(self):

        self.prepare_workspace()
        label_series, image_series = self.get_series()

        if len(label_series) > 0:
            # make job and results queue
            m = Manager()
            jobs = m.Queue()
            results = m.Queue()
            processes = []
            # Progress bar counter
            max_jobs = len(label_series)
            count = 0

            # Build up job queue
            for lseries, iseries in zip(label_series, image_series):
                print("to queue")
                job = dict()
                job['lseries'] = lseries
                job['iseries'] = iseries
                jobs.put(job)

            # Start processes
            for w in range(self.n_cpus):
                p = Process(target=self.process_series,
                            args=(jobs, results))
                p.daemon = True
                p.start()
                processes.append(p)
                jobs.put('STOP')

            # print(str(len(label_series)) + " jobs started, " + str(self.n_cpus) + " workers")

            # Get results and increment counter along with it
            while count < max_jobs:
                img_names = results.get()
                count += 1
                print("processing " + str(count) + "/" + str(max_jobs))

            for p in processes:
                p.join()