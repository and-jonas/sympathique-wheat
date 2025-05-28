
# ======================================================================================================================
# Tracks developing lesions in time series of images and extracts data for each lesion and leaf
# Author: Jonas Anderegg jonas.anderegg@usys.ethz.ch
# Last modified 2024-02-15
# ======================================================================================================================

from pathlib import Path
import glob
import os.path
from utils import base as base_utils
from utils import lesion as lesion_utils
import imageio
import numpy as np
import pandas as pd
import cv2
import copy
from scipy.spatial.distance import cdist
from scipy import ndimage as ndi
from multiprocessing import Manager, Process
from matplotlib import path
import matplotlib.pyplot as plt
from tqdm import tqdm


class SymptomTracker:

    def __init__(self, path_aligned_masks,  path_images, path_kpts, path_output, n_cpus, indices=None):
        self.path_aligned_masks = Path(path_aligned_masks)
        self.path_images = Path(path_images)
        self.path_kpts = Path(path_kpts)
        self.path_output = Path(path_output)
        self.n_cpus = n_cpus
        self.indices = indices

    def prepare_workspace(self):
        """
        Creates all required output directories
        """
        self.path_output.mkdir(parents=True, exist_ok=True)
        with open(f"{self.path_output}/failed_samples.txt", 'w') as file:
            pass

    def log_fail(self, image_id):
        """
        Writes the image id of failure casees to a txt file
        :param image_id: if of the current image
        :param type: the type of transformation attempted ("piecewise" of "projective")
        """
        f = open(f"{self.path_output}/failed_samples.txt", 'a')
        f.writelines(image_id + "\n")
        f.close()

    def create_output_dirs(self, series_id):
        """
        Creates all required output directories for a specific series of images
        :param series_id:
        """
        sample_output_path = self.path_output / series_id
        init_m_path = sample_output_path / "init_mask"
        m_checker_path = sample_output_path / "mask"
        o_checker_path = sample_output_path / "overlay"
        pycn_density_map = sample_output_path / "pycn_density_mask"
        pycn_distance_map = sample_output_path / "pycn_distance_mask"
        lesion_data_path = sample_output_path / "lesion_data"
        leaf_data_path = sample_output_path / "leaf_data"
        leaf_mask = sample_output_path / "leaf_mask"
        instance_mask = sample_output_path / "instance_mask"
        for p in (m_checker_path, o_checker_path, lesion_data_path, leaf_data_path, init_m_path, leaf_mask,
                  pycn_distance_map, pycn_density_map, instance_mask):
            p.mkdir(parents=True, exist_ok=True)
        return m_checker_path, o_checker_path, lesion_data_path, leaf_data_path, init_m_path, leaf_mask, pycn_distance_map, pycn_density_map, instance_mask

    def get_series(self):
        """
        Creates two lists of file paths: to key point coordinate files and to images
        for each of the samples monitored over time, stored in date-wise folders.
        :return:
        """
        mask_series = []
        image_series = []

        masks = glob.glob(f'{self.path_aligned_masks}/*.png')
        images = glob.glob(f'{self.path_images}/*.JPG')
        mask_image_id = ["_".join(os.path.basename(l).split("_")[2:4]).replace(".png", "") for l in masks]
        image_image_id = ["_".join(os.path.basename(l).split("_")[2:4]).replace(".JPG", "") for l in images]
        uniques = np.unique(mask_image_id)

        # if len(images) != len(masks):
        #     raise Exception("list of images and list of coordinate files are not of equal length.")
        #     # warnings.warn("list of images and list of coordinate files are not of equal length."
        #     #               "Ignoring extra coordinate files.")

        print("-- found " + str(len(uniques)) + " unique sample names")

        for unique_sample in uniques:
            image_idx = [index for index, image_id in enumerate(image_image_id) if unique_sample == image_id]
            mask_idx = [index for index, mask_id in enumerate(mask_image_id) if unique_sample == mask_id]
            sample_image_names = [images[i] for i in image_idx]
            sample_masks = [masks[i] for i in mask_idx]
            # sort to ensure sequential processing of subsequent images
            sample_image_names = sorted(sample_image_names, key=lambda i: os.path.splitext(os.path.basename(i))[0])
            sample_masks = sorted(sample_masks, key=lambda i: os.path.splitext(os.path.basename(i))[0])
            mask_series.append(sample_masks)
            image_series.append(sample_image_names)


        if self.indices:
            start, end = self.indices
            m, i = mask_series[start:end], image_series[start:end]
        else:
            m, i = mask_series, image_series
        print("-- processing " + str(len(m)) + " series")

        return m, i

    def process_series(self, work_queue, result):
        """
        Processes the image series for one sample.
        :param work_queue:
        :param result:
        """
        for job in iter(work_queue.get, 'STOP'):

            m_series = job["mseries"]
            i_series = job["iseries"]

            # generate output directories for each series
            series_id = "_".join(os.path.basename(m_series[0]).split("_")[2:4]).replace(".png", "")

            # print("processing " + series_id)

            out_paths = self.create_output_dirs(series_id=series_id)

            # Initialize unique object labels
            next_label = 1
            labels = {}
            all_objects = {}
            num_frames = len(m_series)

            # Process each frame in the time seriesr
            for frame_number in range(1, num_frames + 1):

                # try:

                # print("--" + str(frame_number))

                # get sample identifiers
                png_name = os.path.basename(m_series[frame_number - 1])
                data_name = png_name.replace(".png", ".txt")
                sample_name = png_name.replace(".png", "")
                txt_name = sample_name + '.txt'

                print(data_name)

                # ==================================================================================================================
                # 1. Pre-processing
                # ==================================================================================================================

                # Load the multi-class segmentation mask
                frame_ = cv2.imread(m_series[frame_number - 1], cv2.IMREAD_GRAYSCALE)

                # get coordinates of white marks ("key points")
                if frame_number == 1:
                    kpts = [(0, 0), (frame_.shape[1], 0), (frame_.shape[1], frame_.shape[0]), (0, frame_.shape[0])]
                else:
                    kpts_fn = glob.glob(str(self.path_kpts / txt_name))[0]
                    kpts0 = pd.read_csv(kpts_fn)
                    kpts = base_utils.make_point_list(np.asarray(kpts0))

                # get leaf mask (without insect damage!)
                mask_leaf = np.where((frame_ >= 41) & (frame_ != 153), 1, 0).astype("uint8")
                mask_leaf = np.where(mask_leaf, 255, 0).astype("uint8")

                # get lesion mask
                frame = np.where(frame_ == 85, 255, 0).astype("uint8")
                frame = base_utils.filter_objects_size(mask=frame, size_th=500, dir="smaller")
                # fill small holes
                kernel = np.ones((3, 3), np.uint8)
                frame = cv2.morphologyEx(frame, cv2.MORPH_DILATE, kernel, iterations=2)
                frame = cv2.morphologyEx(frame, cv2.MORPH_ERODE, kernel, iterations=2)
                # remove some artifacts, e.g., around insect damage
                frame = cv2.morphologyEx(frame, cv2.MORPH_ERODE, kernel, iterations=1)
                frame = cv2.morphologyEx(frame, cv2.MORPH_DILATE, kernel, iterations=1)
                # reformat
                frame = np.where(frame, 255, 0).astype("uint8")

                # ==================================================================================================================
                # 2. Get leaf mask
                # ==================================================================================================================

                # Find the reference point (mean of x and y coordinates)
                ref_point = np.mean(kpts, axis=0)

                # Calculate polar angles and sort the points
                sorted_points = sorted(kpts, key=lambda p: np.arctan2(p[1] - ref_point[1], p[0] - ref_point[0]))

                # transform coordinates to a path
                grid_path = path.Path(sorted_points, closed=False)

                # create a mask of the image
                xcoords = np.arange(0, frame.shape[0])
                ycoords = np.arange(0, frame.shape[1])
                coords = np.transpose([np.repeat(ycoords, len(xcoords)), np.tile(xcoords, len(ycoords))])

                # Create mask
                leaf_mask = grid_path.contains_points(coords, radius=-0.5)
                leaf_mask = np.swapaxes(leaf_mask.reshape(frame.shape[1], frame.shape[0]), 0, 1)
                leaf_mask = np.where(leaf_mask, 1, 0).astype("uint8")
                leaf_mask = cv2.morphologyEx(leaf_mask, cv2.MORPH_DILATE, kernel, iterations=2)

                # reduce to roi delimited by the key points
                leaf_checker = mask_leaf * leaf_mask
                cv2.imwrite(f'{out_paths[5]}/{png_name}', leaf_checker)

                # ==================================================================================================================
                # 3. Watershed segmentation for object separation
                # ==================================================================================================================

                if frame_number == 1:
                    seg = frame
                else:
                    seg_lag = seg
                    if len(np.unique(markers)) > 1:
                        seg = lesion_utils.get_object_watershed_labels(current_mask=frame, markers=markers)
                    else:
                        seg = frame

                # important to avoid small water shed segments that cannot be processed
                # this is probably because of small shifts in frames over time (imperfect alignment)
                # removes small false positives
                seg = base_utils.filter_objects_size(mask=seg, size_th=1000, dir="smaller")

                # multiply with leaf mask
                seg = seg * leaf_mask

                if frame_number > 2:
                    try:
                        seg = lesion_utils.complement_mask(leaf_mask=leaf_mask, seg_lag=seg_lag, seg=seg, kpts=kpts0)
                    # if this fails, reset seg to seg_lag and skip the current frame
                    except IndexError:
                        seg = seg_lag
                        continue

                # ==================================================================================================================
                # 3. Identify and add undetected lesions from previous frame
                # ==================================================================================================================

                # get image
                img = cv2.imread(i_series[frame_number - 1], cv2.IMREAD_COLOR)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # init dict
                object_matches = {}  # matches
                objects = {}  # all objects

                # check if there are any missing objects from the last frame in the current frame
                # update the mask if needed
                for lab, (lag_x, lag_y, lag_w, lag_h) in all_objects.items():
                    out1 = seg_lag[lag_y:lag_y + lag_h, lag_x:lag_x + lag_w]
                    out2 = seg[lag_y:lag_y + lag_h, lag_x:lag_x + lag_w]
                    overlap = np.sum(np.bitwise_and(out1, out2)) / (255 * len(np.where(out1)[1]))
                    # if the object cannot be retrieved in the current mask,
                    # paste the object from the previous frame into the current one
                    if overlap < 0.1:
                        seg[lag_y:lag_y + lag_h, lag_x:lag_x + lag_w] = seg_lag[lag_y:lag_y + lag_h,
                                                                        lag_x:lag_x + lag_w]

                # check size again
                seg = base_utils.filter_objects_size(mask=seg, size_th=1000, dir="smaller")

                # generate complete watershed markers
                _, markers, _, _ = cv2.connectedComponentsWithStats(seg, connectivity=8)
                cv2.imwrite(f'{out_paths[4]}/{png_name}', seg)

                # ==================================================================================================================
                # 4. Analyze each lesion: label and extract data
                # ==================================================================================================================

                # find contours
                contours, _ = cv2.findContours(seg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

                # if not lesions are found, the original image without overlay is saved
                if len(contours) < 1:
                    # imageio.imwrite(f'{out_paths[1]}/{png_name}', img)
                    continue

                # Process each detected object in the current frame
                checker = copy.copy(img)
                lesion_data = []

                # prepare distance map
                mask_invert = np.bitwise_not(seg)
                distance_invert = ndi.distance_transform_edt(mask_invert)

                # for pycnidiation monitoring
                image_with_pycn = copy.copy(img)

                # prepare pycnidia density map
                pycn_distmap, pycn_dmap = base_utils.get_pycnidia_maps(
                    mask=frame_,
                    resize_factor=5, bandwidth=25, kernel='gaussian'
                )
                plt.imsave(f'{out_paths[6]}/{png_name}', pycn_distmap)
                plt.imsave(f'{out_paths[7]}/{png_name}', pycn_dmap)

                # if frame_number == 11:
                #     print("stop")
                instance_mask = np.zeros_like(seg)
                for idx, contour in enumerate(contours):

                    print("----" + str(idx))

                    # get the roi
                    x, y, w, h = map(int, cv2.boundingRect(contour))
                    objects[idx] = (x, y, w, h)
                    rect = cv2.boundingRect(contour)
                    roi = lesion_utils.select_roi_2(rect=rect, mask=seg)
                    

                    # check if is fully on the imaged leaf
                    in_leaf_checker = np.unique(leaf_mask[np.where(roi)[0], np.where(roi)[1]])[0]

                    # check if is a new object by comparing with each previously identified object
                    is_new_object = True
                    for lag_label, (lag_x, lag_y, lag_w, lag_h) in labels.items():

                        # print("------" + str(lag_label))

                        # get the mask of the lag object in context
                        rect_lag = (lag_x, lag_y, lag_w, lag_h)

                        # skip check if rectangles do not overlap
                        if not base_utils.rectangles_overlap(rect, rect_lag):
                            continue

                        # if rectangles do overlap, perform more detailed check
                        roi_lag = lesion_utils.select_roi_2(rect=rect_lag, mask=seg_lag)

                        # get areas and overlap
                        lag_area = np.sum(np.logical_and(roi_lag, roi_lag))
                        int_area = np.sum(np.logical_and(roi, roi_lag))
                        contour_overlap = int_area / lag_area

                        # if overlaps, then it is not a new lesion but an already tracked one
                        # --> add corresponding label and terminate search
                        if contour_overlap >= 0.2:  # <== CRITICAL =================================================
                            is_new_object = False
                            object_matches[lag_label] = (x, y, w, h)
                            current_label = lag_label  # Update the label to the existing object's label
                            break

                    # If the object is not sufficiently overlapped with any previous object, assign a new label
                    if is_new_object:
                        object_matches[next_label] = (x, y, w, h)
                        current_label = next_label  # Update the label to the newly assigned label
                        next_label += 1

                    instance_mask = np.where(roi, current_label, instance_mask)

                    # modify dimensions of the bounding rectangle
                    rect = lesion_utils.get_bounding_boxes(rect=rect)

                    # extract roi
                    _, empty_img, _ = lesion_utils.select_roi(rect=rect, img=img, mask=seg)

                    # extract RGB profile, checker image, spline normals, and spline base points
                    _, out_checker, spl, spl_points, cols = lesion_utils.spline_contours(
                        mask_obj=roi,
                        mask_leaf=leaf_checker,
                        img=empty_img,
                        checker=checker,
                        distance_invert=distance_invert,
                    )

                    # extract lesion data
                    if len(spl[0]) != 0 and in_leaf_checker != 0:

                        # extract perimeter lengths
                        analyzable_perimeter = len(spl[1]) / len(spl[0])
                        occluded_perimeter = len(spl[2]) / len(spl[0])

                        # Calculate total distance in x and y directions
                        total_x_distance = np.sum(np.abs(np.diff(spl_points[0])))
                        total_y_distance = np.sum(np.abs(np.diff(spl_points[1])))

                        # calculate expandable distance in x and y directions
                        remove_idx = np.unique(cols)
                        if len(remove_idx) != 0:
                            r_idxs = base_utils.split_consecutive_sets(remove_idx)
                            dx = sum(np.sum(np.abs(np.diff(spl_points[0][i]))) for i in r_idxs)
                            dy = sum(np.sum(np.abs(np.diff(spl_points[1][i]))) for i in r_idxs)
                        else:
                            dx = 0
                            dy = 0

                        # extract other lesion properties
                        # these are extracted from the original (un-smoothed) contour
                        contour_area = cv2.contourArea(contour)
                        contour_perimeter = cv2.arcLength(contour, True)
                        hull = cv2.convexHull(contour)
                        hull_area = cv2.contourArea(hull)
                        contour_solidity = float(contour_area) / hull_area
                        _, _, w, h = x, y, w, h = cv2.boundingRect(contour)

                        # extract pycnidia number
                        pycn_mask = np.where(roi, frame_, 0)
                        n_pycn = len(np.where(pycn_mask == 212)[0])
                        pycn_density_lesion = n_pycn / contour_area

                        # extract rust number
                        rust_mask = np.where(roi, frame_, 0)
                        n_rust = len(np.where(rust_mask == 255)[0])
                        rust_density_lesion = n_rust / contour_area

                        pycn_features, pycn_contour = base_utils.get_pycn_features(
                            mask=frame_,
                            lesion_mask=roi,
                            contour=contour,
                            max_dist=50, bandwidth=25, kernel='gaussian'
                        )

                        # collect output data
                        ldat = {'label': current_label,
                                'area': contour_area,
                                'perimeter': contour_perimeter,
                                'solidity': contour_solidity,
                                'analyzable_perimeter': analyzable_perimeter,
                                'occluded_perimeter': occluded_perimeter,
                                'x_perimeter': total_x_distance,
                                'y_perimeter': total_y_distance,
                                'x_perimeter_n': dx,
                                'y_perimeter_n': dy,
                                'max_width': w,
                                'max_height': h,
                                'n_pycn': n_pycn,
                                'pycn_density_lesion': pycn_density_lesion,
                                'rust_density_lesion': rust_density_lesion
                                }
                    else:
                        keys = ['area', 'perimeter', 'solidity', 'analyzable_perimeter',
                                'occluded_perimeter', 'x_perimeter', 'y_perimeter',
                                'x_perimeter_n', 'y_perimeter_n', 'max_width', 'max_height',
                                'n_pycn', 'pycn_density_lesion', 'rust_density_lesion']
                        ldat = ldat | {'label': current_label} | {key: np.nan for key in keys}

                    # draw pycnidiation contour
                    if pycn_contour is not None:
                        for p in pycn_contour:
                            cv2.drawContours(image_with_pycn, p, -1, (255, 255, 0), 2)
                    cv2.drawContours(image_with_pycn, contour, -1, (0, 0, 0), 2)

                    # data
                    ldat = ldat | pycn_features
                    lesion_data.append(ldat)

                # except:
                #     self.log_fail(sample_name)
                #     continue

                # export instance mask
                cv2.imwrite(f'{out_paths[8]}/{png_name}', instance_mask)

                # Update the labels with the new matches
                labels = object_matches
                all_objects = objects

                # ==================================================================================================================
                # 5. Analyze leaf
                # ==================================================================================================================

                # summary stats
                la_tot = (frame_.shape[0] * frame_.shape[1]) - len(
                    np.where(frame_ == 0)[0])  # roi area - background pixels
                la_damaged = len(np.where((frame_ != 0) & (frame_ != 42))[0])
                la_healthy = len(np.where(frame_ == 42)[0])
                la_damaged_f = la_damaged / la_tot
                la_healthy_f = la_healthy / la_tot
                la_insect = len(np.where(frame_ == 127)[0])
                n_pycn = len(np.where(frame_ == 212)[0])
                rust_idx = np.where(frame_ == 255)
                rust_point_list = base_utils.filter_points(x=rust_idx[1], y=rust_idx[0], min_distance=7)
                n_rust = len(rust_point_list)
                n_lesion = len(contours)
                placl = len(np.where((frame_ == 85) | (frame_ == 212))[0]) / (la_tot - la_insect)
                pycn_density = n_pycn / (la_tot - la_insect)
                rust_density = n_rust / (la_tot - la_insect)

                # distribution metrics
                out = ndi.distance_transform_edt(np.bitwise_not(seg))
                out[frame_ == 0] = np.nan
                out[out == 0] = np.nan
                mean_dist = np.nanmean(out)
                std_dist = np.nanstd(out)
                cv_dist = std_dist / mean_dist
                n_comps, output, stats, centroids = cv2.connectedComponentsWithStats(seg, connectivity=8)
                if n_comps > 2:
                    distance = cdist(centroids[1:], centroids[1:], metric='euclidean')
                    np.fill_diagonal(distance, np.nan)
                    shortest_dist = np.nanmin(distance, axis=1)
                    mean_shortest_dist = np.mean(shortest_dist)
                    std_shortest_dist = np.std(shortest_dist)
                    cv_shortest_dist = std_shortest_dist / mean_shortest_dist
                else:
                    mean_shortest_dist = np.nan
                    std_shortest_dist = np.nan
                    cv_shortest_dist = np.nan

                # grab data
                leaf_data = [
                    {
                        'la_tot': la_tot,
                        'la_damaged': la_damaged,
                        'la_healthy': la_healthy,
                        'la_damaged_f': la_damaged_f,
                        'la_healthy_f': la_healthy_f,
                        'la_insect': la_insect,
                        'n_pycn': n_pycn,
                        'n_rust': n_rust,
                        'n_lesion': n_lesion,
                        'placl': placl,
                        'pycn_density': pycn_density,
                        'rust_density': rust_density,
                        'mean_dist': mean_dist,
                        'std_dist': std_dist,
                        'cv_dist': cv_dist,
                        'mean_shortest_dist': mean_shortest_dist,
                        'std_shortest_dist': std_shortest_dist,
                        'cv_shortest_dist': cv_shortest_dist,
                    },
                ]

                # Create a DataFrame from the list of dictionaries
                df = pd.DataFrame(leaf_data)

                # Export the DataFrame to a CSV file
                df.to_csv(f'{out_paths[3]}/{data_name}', index=False)

                # ==================================================================================================================
                # 6. Create output
                # ==================================================================================================================

                # save lesion data
                df = pd.DataFrame(lesion_data, columns=lesion_data[0].keys())
                df.to_csv(f'{out_paths[2]}/{data_name}', index=False)

                # Draw and save the labeled objects on the frame
                frame_with_labels = cv2.cvtColor(seg, cv2.COLOR_GRAY2BGR)
                image_with_labels = copy.copy(out_checker)
                for label, (x, y, w, h) in labels.items():
                    cv2.rectangle(frame_with_labels, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame_with_labels, str(label), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    cv2.rectangle(image_with_labels, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    cv2.putText(image_with_labels, str(label), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                    # cv2.rectangle(image_with_pycn, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    # cv2.putText(image_with_pycn, str(label), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

                # Draw pycnidia and rust
                pycnidia = np.where(frame_ == 212)
                try:
                    p_y_coords, p_x_coords = pycnidia
                    # Draw circles at the specified coordinates
                    for x, y in zip(p_y_coords, p_x_coords):
                        cv2.circle(image_with_pycn, (y, x), 5, (255, 0, 0), 1)
                        cv2.circle(image_with_labels, (y, x), 5, (0, 0, 255), 1)
                except IndexError:
                    pass
                try:
                    r_y_coords, r_x_coords = rust_point_list[:, 1], rust_point_list[:, 0]
                    for x, y in zip(r_y_coords, r_x_coords):
                        cv2.circle(image_with_pycn, (y, x), 5, (0, 255, 0), 1)
                        cv2.circle(image_with_labels, (y, x), 5, (0, 0, 255), 1)
                except IndexError:
                    pass

                # cv2.imwrite(f'{out_paths[0]}/{png_name}', frame_with_labels)
                imageio.imwrite(f'{out_paths[1]}/{png_name}', image_with_labels)
                imageio.imwrite(f'{out_paths[0]}/{png_name}', image_with_pycn)

            result.put(series_id)

    # ==================================================================================================================

    def process_all(self):
        self.prepare_workspace()
        mask_series, image_series = self.get_series()

        if len(mask_series) > 0:
            # make job and results queue
            m = Manager()
            jobs = m.Queue()
            results = m.Queue()
            processes = []
            # Progress bar counter
            max_jobs = len(mask_series)
            count = 0

            # Build up job queue
            with tqdm(total=len(mask_series), desc="Processing jobs") as pbar:
                for mseries, iseries in zip(mask_series, image_series):
                    job = dict()
                    job['mseries'] = mseries
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

                print(str(len(mask_series)) + " jobs started, " + str(self.n_cpus) + " workers")

                # Get results and increment counter along with it
                while count < max_jobs:
                    count += 1
                    pbar.update(1)

                for p in processes:
                    p.join()