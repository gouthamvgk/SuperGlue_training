import os
import numpy as np
import cv2
import torch

def get_rotmat(angle, as_3d=False, scale=1.0, center_x=0.0, center_y=0.0):
    cos_angle, sine_angle = np.cos(angle) * scale, np.sin(angle) * scale
    rotation = [cos_angle, -sine_angle, sine_angle, cos_angle]
    rotation = np.reshape(rotation, (2, 2)).T
    if as_3d:
        matrix_3d = np.eye(3)
        matrix_3d[:2, :2] = rotation
        matrix_3d[0, 2] = ((1 - cos_angle)*center_x) - (sine_angle*center_y)
        matrix_3d[1, 2] = (sine_angle*center_x) + ((1-cos_angle)*center_y)
        return matrix_3d
    return rotation

def get_translation_mat(image_height, image_width, trans, transformed_corners):
    left_top_min = np.min(transformed_corners, axis=0)
    right_bottom_min = np.min(np.array([image_width, image_height]) - transformed_corners, axis=0)
    trans_x_value = int(np.random.uniform(0, trans) * image_width)
    trans_y_value = int(np.random.uniform(0, trans) * image_height)
    if np.random.uniform() > 0.5: #translate x with respect to left axis
        trans_x = trans_x_value if left_top_min[0] < 0 else -trans_x_value
    else: #translate x with respect to right axis
        trans_x = trans_x_value if right_bottom_min[0] > 0 else -trans_x_value
    if np.random.uniform() > 0.5: #translate y with respect to top axis
        trans_y = trans_y_value if left_top_min[1] < 0 else -trans_y_value
    else: #translate y with respect to bottom axis
        trans_y = trans_y_value if right_bottom_min[1] > 0 else -trans_y_value
    translate_mat = np.eye(3)
    translate_mat[0, 2] = trans_x
    translate_mat[1, 2] = trans_y
    return translate_mat

def get_perspective_mat(patch_ratio, center_x, center_y, pers_x, pers_y, shear_ratio, shear_angle, rotation_angle, scale, trans):
    shear_angle, rotation_angle = np.deg2rad(shear_angle), np.deg2rad(rotation_angle)
    image_height, image_width = center_y * 2, center_x * 2
    patch_bound_w, patch_bound_h = int(patch_ratio * image_width), int(patch_ratio * image_height)
    patch_corners = np.array([[0,0], [0, patch_bound_h], [patch_bound_w, patch_bound_h], [patch_bound_w, 0]]).astype(np.float32)
    pers_value_x = np.random.normal(0, pers_x/2)
    pers_value_y = np.random.normal(0, pers_y/2)
    pers_matrix = np.array([[1, 0, 0], [0, 1, 0], [pers_value_x, pers_value_y, 1]])
    #shear_ratio is given by shear_x/shear_y
    if np.random.uniform() > 0.5:
        shear_ratio_value = np.random.uniform(1, 1+shear_ratio)
        shear_x, shear_y = 1, 1 / shear_ratio_value
    else:
        shear_ratio_value = np.random.uniform(1-shear_ratio, 1)
        shear_x, shear_y = shear_ratio_value, 1
    shear_angle_value = np.random.uniform(-shear_angle, shear_angle)
    shear_matrix = get_rotmat(-shear_angle_value, as_3d=True, center_x=center_x, center_y=center_y) @ np.diag([shear_x, shear_y, 1]) @ get_rotmat(shear_angle_value, as_3d=True, center_x=center_x, center_y=center_y)
    shear_perspective = shear_matrix @ pers_matrix
    rotation_angle_value = np.random.uniform(-rotation_angle, rotation_angle)
    scale_value = np.random.uniform(1, 1+(2*scale))
    #priotrising scaling up compared to scaling down
    scaled_rotation_matrix = get_rotmat(rotation_angle_value, as_3d=True, scale=scale_value, center_x=center_x, center_y=center_y)
    homography_matrix = scaled_rotation_matrix @ shear_perspective
    trans_patch_corners = cv2.perspectiveTransform(np.reshape(patch_corners, (-1, 1, 2)), homography_matrix).squeeze(1)
    translation_matrix = get_translation_mat(image_height, image_width, trans, trans_patch_corners)
    homography_matrix = translation_matrix @ homography_matrix
    return homography_matrix

def torch_cdist(keypoints1, keypoints2):
    diff = (keypoints1[:, None, :] - keypoints2[None, :, :]) ** 2
    summed = diff.sum(-1)
    distance = torch.sqrt(summed)
    return distance

def torch_setdiff1d(miss_index, match_index):
    combined = torch.cat((miss_index, match_index))
    unq, count = combined.unique(return_counts=True)
    diff = unq[count == 1]
    return diff

def warp_keypoints(keypoints, homography_mat):
    source = torch.cat([keypoints, torch.ones(len(keypoints), 1).to(keypoints.device)], dim=-1)
    dest = (homography_mat @ source.T).T
    dest /= dest[:, 2:3]
    return dest[:, :2]

def torch_find_matches(src_keypoints1, src_keypoints2, homography, dist_thresh=3, n_iters=1):
    match_list_1, match_list_2 = torch.empty(0,dtype=torch.int64,device=src_keypoints1.device), torch.empty(0,dtype=torch.int64,device=src_keypoints2.device) 
    missing_indices_1 = torch.arange(len(src_keypoints1), device=src_keypoints1.device, dtype=torch.long)
    missing_indices_2 = torch.arange(len(src_keypoints2), device=src_keypoints2.device, dtype=torch.long)
    src_keypoints1_projected = warp_keypoints(src_keypoints1, homography) #keypoint1 must be projected to keypoint2 space for comparsion
    for i in range(n_iters): #2nd iteration only recover 1 or 2 keypoint matches in most cases. So 1 should be enough.
        keypoints1 = src_keypoints1_projected[missing_indices_1, :]
        keypoints2 = src_keypoints2[missing_indices_2, :]
        distance = torch_cdist(keypoints1, keypoints2)
        min1 = torch.argmin(distance, 1)
        min2 = torch.argmin(distance, 0)
        intersect_indexes_2 = torch.where(min1[min2] == torch.arange(len(min2), device=min1.device))[0]
        intersect_indexes1 = min2[intersect_indexes_2]
        matched_distances = distance[intersect_indexes1, intersect_indexes_2] < dist_thresh
        intersect_indexes1 = intersect_indexes1[matched_distances]
        intersect_indexes_2 = intersect_indexes_2[matched_distances]
        matched_indexes_1 = missing_indices_1[intersect_indexes1]
        matched_indexes_2 = missing_indices_2[intersect_indexes_2]
        missing_indices_1 = torch_setdiff1d(missing_indices_1, matched_indexes_1)
        missing_indices_2 = torch_setdiff1d(missing_indices_2, matched_indexes_2)
        match_list_1 = torch.cat((match_list_1, matched_indexes_1))
        match_list_2 = torch.cat((match_list_2, matched_indexes_2))
    return match_list_1, match_list_2, missing_indices_1, missing_indices_2

def scale_homography(homo_matrix, src_height, src_width, dest_height, dest_width):
    """
    If src and warped image is scaled by same amount, then homography needs to changed according
    to the scale in x and y direction
    """
    scale_x = dest_width / src_width
    scale_y = dest_height / src_height
    scale_matrix = np.diag([scale_x, scale_y, 1.0])
    homo_matrix = scale_matrix @ homo_matrix @ np.linalg.inv(scale_matrix)
    return homo_matrix

def resize_aspect_ratio(image, resize_h, resize_w):
    h, w = image.shape[0:2]
    max_size = max(h, w)
    ratio_h, ratio_w = h / max_size, w / max_size
    new_height, new_width = int(resize_h * ratio_h), int(resize_w * ratio_w)
    resized = cv2.resize(image, (new_width, new_height))
    template = np.ones((resize_h, resize_w), dtype=np.uint8) * np.random.randint(0, 127)
    template[((resize_h - resized.shape[0])//2):((resize_h - resized.shape[0])//2) + resized.shape[0],
              ((resize_w - resized.shape[1])//2):((resize_w - resized.shape[1])//2) + resized.shape[1]] = resized
    return template