
import math
import scipy.interpolate as si
from skimage.draw import line
import cv2
import numpy as np
from utils import base as base_utils
from scipy import ndimage as ndi
from skimage.segmentation import watershed
from matplotlib import path


def get_bounding_boxes(rect):
    """
    Get bounding boxes of each maintained lesion in a full leaf image
    :param rect: the original rectangle
    :return: Coordinates of the bounding boxes as returned by cv2.boundingRect()
    """
    x, y, w, h = rect
    w = w + 30
    h = h + 30
    x = x - 15
    y = y - 15
    # boxes must not extend beyond the edges of the image
    if x < 0:
        w = w-np.abs(x)
        x = 0
    if y < 0:
        h = h-np.abs(y)
        y = 0
    coords = x, y, w, h

    return coords


def select_roi(rect, img, mask):
    """
    Selects part of an image defined by bounding box coordinates. The selected patch is pasted onto empty masks for
    processing in correct spatial context
    :param rect: bounding box coordinates (x, y, w, h) as returned by cv2.boundingRect()
    :param img: The image to process
    :param mask: The binary mask of the same image
    :return: Roi of masks and img, centroid coordinates of the lesion to process (required for clustering)
    """
    # get the coordinates of the rectangle to process
    x, y, w, h = rect

    # create empty files for processing in spatial context
    empty_img = np.ones(img.shape).astype('int8') * 255
    empty_mask = np.zeros(mask.shape)
    empty_mask_all = np.zeros(mask.shape)

    # filter out the central object (i.e. lesion of interest)
    # isolate the rectangle
    patch_mask_all = mask[y:y + h, x:x + w]

    # select object by size or by centroid position in the patch
    n_comps, output, stats, centroids = cv2.connectedComponentsWithStats(patch_mask_all, connectivity=8)

    # if there is more then one object in the roi, need to select the one of interest
    if n_comps > 2:
        sizes = list(stats[:, 4][1:])
        max_idx = np.argmax(sizes)
        lesion_mask = np.uint8(np.where(output == max_idx + 1, 255, 0))
        ctr_obj = [centroids[max_idx + 1][0] + x, centroids[max_idx + 1][1] + y]

    # if there is only one object, select it
    else:
        lesion_mask = np.uint8(np.where(output == 1, 255, 0))
        ctr_obj = [centroids[1][0] + x, centroids[1][1] + y]

    # paste the patches onto the empty files at the correct position
    empty_img[y:y + h, x:x + w, :] = img[y:y + h, x:x + w, :]
    empty_mask[y:y + h, x:x + w] = lesion_mask
    empty_mask_all[y:y + h, x:x + w] = patch_mask_all

    mask_all = empty_mask_all.astype("uint8")

    return mask_all, empty_img, ctr_obj


def select_roi_2(rect, mask):
    x, y, w, h = rect
    roi = np.zeros_like(mask)
    roi[y:y + h, x:x + w] = mask[y:y + h, x:x + w]
    c, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(c, key=cv2.contourArea)
    roi = np.zeros_like(roi)
    cv2.drawContours(roi, [largest_contour], 0, 255, thickness=cv2.FILLED)
    return roi


def check_color_profiles(color_profiles, dist_profiles_outer, leaf_profiles, spline_normals):
    """
    Removes spline normals (and corresponding color profiles) that (a) extend into the lesion sphere of the same lesion
    (convexity defects) and replaces values on the inner side of the spline normals that lie beyond the "center" of the
    lesion (i.e. extend too far inwards).
    :param color_profiles: A 3D array (an image), raw color profiles, sampled on the spline normals
    :param leaf_profiles: A 3D array (an image), binary, sampled on the leaf mask
    :param dist_profiles_outer: the euclidian distance map of the inverse binary image
    :param spline_normals: the spline normals in cv2 format
    :return: Cleaned color profiles (after removing normals in convexity defects and replacing values of normals
    extending too far inwards) and the cleaned list of spline normals in cv2 format.
    """

    dist_profiles_outer = dist_profiles_outer.astype("int32")
    diff_out = np.diff(dist_profiles_outer, axis=0)

    # identify lesion edge positions
    cols_0 = np.where(leaf_profiles != 255)[1]
    # separate the normals into complete and incomplete
    cols_1 = np.where(diff_out < 0)[1]
    # incomplete because extending outside the ROI (in extreme cases)
    cols_2 = np.where(np.all(diff_out == 0, axis=0))[0]
    # starting with a low distance value, in intermediate cases
    cols_3 = np.where(dist_profiles_outer[dist_profiles_outer.shape[0] - 1] < 5, )[0]
    # combine criteria
    all = [arr for arr in [cols_0, cols_1, cols_2, cols_3] if arr.size > 0]
    try:
        cols = np.concatenate(all)
    except ValueError:
        cols = []
    spl_n_full_length = [i for j, i in enumerate(spline_normals) if j not in np.unique(cols)]
    spl_n_red_length = [i for j, i in enumerate(spline_normals) if j in np.unique(cols)]
    checked_cprof = np.delete(color_profiles, cols, 1)

    return checked_cprof, spl_n_full_length, spl_n_red_length


def get_spline_normals(spline_points, length_in=0, length_out=15):
    """
    Gets spline normals (lines) in cv2 format
    :param spline_points: x- and y- coordinates of the spline base points, as returned by spline_approx_contour().
    :param length_in: Int, indicating how far splines should extend inwards
    :param length_out: Int, indicating how far splines should extend outwards
    :return: A list of the spline normals, each in cv2 format.
    """
    ps = np.vstack((spline_points[1], spline_points[0])).T

    x_i = spline_points[0]
    y_i = spline_points[1]

    # get endpoints of the normals
    endpoints = []
    for i in range(0, len(ps)-1):
        v_x = y_i[i] - y_i[i + 1]
        v_y = x_i[i] - x_i[i + 1]
        mag = math.sqrt(v_x * v_x + v_y * v_y)
        v_x = v_x / mag
        v_y = v_y / mag
        temp = v_x
        v_x = -v_y
        v_y = temp
        A_x = int(y_i[i] + v_x * length_in)
        A_y = int(x_i[i + 1] + v_y * length_in)
        B_x = int(y_i[i] - v_x * length_out)
        B_y = int(x_i[i + 1] - v_y * length_out)
        n = [A_x, A_y], [B_x, B_y]
        endpoints.append(n)

    # get normals (lines) connecting the endpoints
    normals = []
    for i in range(len(endpoints)):
        p1 = endpoints[i][0]
        p2 = endpoints[i][1]
        discrete_line = list(zip(*line(*p1, *p2)))
        discrete_line = [[list(ele)] for ele in discrete_line]
        cc = [np.array(discrete_line, dtype=np.int32)]  # cv2 contour format
        normals.append(cc)

    return normals


def extract_normals_pixel_values(img, normals, length_in=0, length_out=15):
    """
    Extracts the pixel values situated on the spline normals.
    :param length_out: Int, how long the contour normals should extend from the lesion
    :param length_in: Int, how long the contour normals should extend into the lesion
    :param img: The image, binary mask or edt to process
    :param normals: The normals extracted in cv2 format as resulting from utils.get_spline_normals()
    :return: The "scan", i.e. an image (binary, single-channel 8-bit, or RGB) with stacked extracted profiles
    """
    # check whether is multi-channel image or 2d array
    is_img = base_utils.is_multi_channel_img(img)

    # For a normal perfectly aligned with the image axes, length equals the number of inward and outward pixels defined
    # All normals (differing in "pixel-length" due to varying orientation in space, are interpolated to the same length
    max_length_contour = length_in + length_out + 1

    # iterate over normals
    profile_list = []
    for k, normal in enumerate(normals):

        # get contour pixel coordinates
        contour_points = base_utils.flatten_contour_data(normal, asarray=False)

        # extract pixel values
        values = []
        for i, point in enumerate(contour_points):

            x = point[1]
            y = point[0]

            # try to sample pixel values, continue if not possible (extending outside of the roi)
            try:
                value = img[x, y].tolist()
            except IndexError:
                continue
            values.append(value)

            # split channels (R,G,B)
            # if img is a 3d array:
            if len(img.shape) > 2:
                channels = []
                for channel in range(img.shape[2]):
                    channel = [item[channel] for item in values]
                    channels.append(channel)
            else:
                channels = [values]

        # interpolate pixel values on contours to ensure equal length of all contours
        # for each channel
        interpolated_contours = []
        for channel in channels:
            size = len(channel)
            xloc = np.arange(len(channel))
            new_size = max_length_contour
            new_xloc = np.linspace(0, size, new_size)
            new_data = np.interp(new_xloc, xloc, channel).tolist()
            interpolated_contours.extend(new_data)

        if is_img:
            # create list of arrays
            line_scan = np.zeros([max_length_contour, 1, 3], dtype=np.uint8)
            for i in range(max_length_contour):
                v = interpolated_contours[i::max_length_contour]
                line_scan[i, :] = v
        else:
            line_scan = np.zeros([max_length_contour, 1], dtype=np.uint8)
            for i in range(max_length_contour):
                v = interpolated_contours[i::max_length_contour]
                line_scan[i, :] = v

        profile_list.append(line_scan)

    # stack arrays
    try:
        scan = np.hstack(profile_list)
    except ValueError:
        scan = None

    return scan


def spline_approx_contour(contour, sf=0.25):
    """
    Approximates lesion edges by b-splines
    :param contour: Contours detected in a binary mask.
    :param sf: smoothing factor; correcting the standard m - sqrt(2*m)
    :return: x and y coordinates of pixels making up the smoothed contour, OR representing the spline normal base
    points.
    """
    # re-sample contour points
    contour_points = base_utils.flatten_contour_data(input_contour=contour, asarray=True)

    # find B-Spline representation of contour
    # control smoothing
    s = sf * len(contour_points) - np.sqrt(2*len(contour_points))
    try:
        tck, u = si.splprep(contour_points.T, u=None, s=s, per=1, quiet=1)
        # evaluate  B-spline
        u_new = np.linspace(u.min(), u.max(), int(len(contour_points)))
        y_new, x_new = si.splev(u_new, tck, der=0)
    except (ValueError, TypeError) as error:
        print(error)
        x_new, y_new = ([], [])
    return x_new, y_new


def spline_contours(mask_obj, mask_all, mask_leaf, img, checker, distance_invert):
    """
    Wrapper function for processing of contours via spline normals
    :param mask_leaf: a binary mask with 255 for leaf and 0 for background
    :param mask_obj: a binary mask containing only the lesion of interest.
    :param mask_all: a binary mask containing all the segmented objects in the patch
    :param img: the original patch image
    :param checker: A copy of the (full) image to process.
    :return: cleaned color profiles from contour normals in cv2 format, an image for evaluation
    """

    # get contour
    contour, _ = cv2.findContours(mask_obj, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # get spline points and smoothed contour
    spline_points = spline_approx_contour(contour, sf=0.25)

    # if the spline approximation was successful, process normally
    # otherwise make empty
    if not len(spline_points[0]) == 0:

        sm_contour = base_utils.make_cv2_formatted(spline_points)

        # get spline normals
        spl_n = get_spline_normals(spline_points=spline_points)

        # sample normals on image and on false object mask
        color_profiles = extract_normals_pixel_values(img=img, normals=spl_n)

        # # sample normals on distance maps
        dist_profiles_outer = extract_normals_pixel_values(img=distance_invert, normals=spl_n)

        # sample normals on leaf masks
        leaf_profiles = extract_normals_pixel_values(img=mask_leaf, normals=spl_n)

        # remove normals that extend into lesion
        final_profiles, spl_n_full, spl_n_red = check_color_profiles(
            color_profiles=color_profiles,
            dist_profiles_outer=dist_profiles_outer,
            leaf_profiles=leaf_profiles,
            spline_normals=spl_n,
        )

    else:
        spl_n = []
        sm_contour = []
        final_profiles, spl_n_full, spl_n_red, = None, [], []

    # create the check image
    # add analyzable normals
    for i in range(len(spl_n_full)):
        cv2.drawContours(checker, spl_n_full[i], -1, (255, 0, 0), 1)
    # add normals on leaf edges or extending into the lesion itself
    if spl_n_red is not None:
        for i in range(len(spl_n_red)):
            cv2.drawContours(checker, spl_n_red[i], -1, (0, 122, 0), 1)
    # add smoothed contour
    if sm_contour is not None:
        for c in [sm_contour]:
            cv2.drawContours(checker, c, -1, (0, 0, 255), 1)

    return final_profiles, checker, (spl_n, spl_n_full, spl_n_red), spline_points


def get_object_watershed_labels(current_mask, markers):
    """
    Performs watershed segmentation of merged objects
    :param current_mask: the current frame in the time series (a binary mask)
    :param markers: labelled components from the previous frame
    :return: the current frame with separated objects
    """
    # binarize the post-processed mask
    mask = np.where(current_mask == 125, 255, current_mask)
    # invert the mask
    mask_inv = np.bitwise_not(mask)
    # calculate the euclidian distance transform
    distance = ndi.distance_transform_edt(mask_inv)
    # watershed segmentation, using labelled components as markers
    # this must be done to ensure equal number of watershed segments and connected components!
    labels = watershed(distance, markers=markers, watershed_line=True)
    # make thicker watershed lines for separability of objects
    watershed_lines = np.ones(shape=np.shape(labels))
    watershed_lines[labels == 0] = 0  # ws lines are labeled as 0 in markers
    kernel = np.ones((2, 2), np.uint8)
    watershed_lines = cv2.erode(watershed_lines, kernel, iterations=1)
    labels = labels * watershed_lines
    labels = labels * current_mask/255
    separated = np.where(labels != 0, 255, 0)
    return np.uint8(separated)


def complement_mask(leaf_mask, seg, seg_lag, kpts):
    """
    Checks if a part of the leaf is missing due to undetected or lost markers. If a part of the leaf is missing,
    complements the current segmentation mask with objects from the preceding mask in the areas of interest.
    :param leaf_mask: the current leaf mask
    :param seg: the current segmentation mask
    :param seg_lag: the preceding segmentation mask
    :param kpts: the detected key points (marker coordinates)
    :return: the complemented segmentation mask
    """
    # sort the points based on their x-coordinates
    kpts = np.array(kpts)
    upper = kpts[kpts[:, 1] < 300]
    lower = kpts[kpts[:, 1] > 300]
    xSortedUpper = upper[np.argsort(upper[:, 0]), :]
    xSortedLower = lower[np.argsort(lower[:, 0]), :]

    # grab the left-most and right-most points from the sorted x-roodinate points
    leftMostUpper = xSortedUpper[:1, :][:, 0]
    rightMostUpper = xSortedUpper[-1:, :][:, 0]
    leftMostLower = xSortedLower[:1, :][:, 0]
    rightMostLower = xSortedLower[-1:, :][:, 0]

    leftMost = leftMostUpper[0], leftMostLower[0]
    rightMost = rightMostUpper[0], rightMostLower[0]

    # check if a part of the leaf is missing
    if any(num < leaf_mask.shape[1]-250 for num in rightMost):

        # transform coordinates to a path
        ur = tuple(xSortedUpper[-1:, :][0])
        u_corner = tuple([leaf_mask.shape[1], 0])
        l_corner = tuple([leaf_mask.shape[1], leaf_mask.shape[0]])
        lr = tuple(xSortedLower[-1:, :][0])
        grid_path = path.Path([ur, u_corner, l_corner, lr, ur], closed=False)

        # create a mask of the image
        xcoords = np.arange(0, leaf_mask.shape[0])
        ycoords = np.arange(0, leaf_mask.shape[1])
        coords = np.transpose([np.repeat(ycoords, len(xcoords)), np.tile(xcoords, len(ycoords))])

        # Create mask
        c_mask = grid_path.contains_points(coords, radius=-0.5)
        c_mask = np.swapaxes(c_mask.reshape(leaf_mask.shape[1], leaf_mask.shape[0]), 0, 1)
        c_mask = np.where(c_mask, 1, 0).astype("uint8")

        seg_lag_complement = c_mask * seg_lag
        seg = seg + seg_lag_complement

    if any(num > 250 for num in leftMost):
        # transform coordinates to a path
        ul = tuple(xSortedUpper[:1, :][0])
        ll = tuple(xSortedLower[:1, :][0])
        l_corner = tuple([0, leaf_mask.shape[0]])
        u_corner = tuple([0, 0])
        grid_path = path.Path([ul, ll, l_corner, u_corner, ul], closed=False)

        # create a mask of the image
        xcoords = np.arange(0, leaf_mask.shape[0])
        ycoords = np.arange(0, leaf_mask.shape[1])
        coords = np.transpose([np.repeat(ycoords, len(xcoords)), np.tile(xcoords, len(ycoords))])

        # Create mask
        c_mask = grid_path.contains_points(coords, radius=-0.5)
        c_mask = np.swapaxes(c_mask.reshape(leaf_mask.shape[1], leaf_mask.shape[0]), 0, 1)
        c_mask = np.where(c_mask, 1, 0).astype("uint8")

        seg_lag_complement = c_mask * seg_lag
        seg = seg + seg_lag_complement

    return seg
