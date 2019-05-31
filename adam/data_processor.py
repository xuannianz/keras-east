# coding:utf-8
import glob
import sys
import csv
import cv2
import time
import os
import argparse
import itertools
from multiprocessing import Pool
import threading
import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt
import matplotlib.patches as Patches
from shapely.geometry import Polygon
import os.path as osp
import tensorflow as tf
import logging
import h5py
import random


def get_images(data_path):
    files = []
    idx = 0
    for ext in ['jpg', 'png', 'jpeg', 'JPG']:
        files.extend(glob.glob(
            os.path.join(data_path, '*.{}'.format(ext))))
        idx += 1
    return files


def check_annotations(images_dir):
    for image_path in glob.glob(osp.join(images_dir, '*.jpg')):
        image = cv2.imread(image_path)
        h, w = image.shape[:2]
        image_fname = osp.split(image_path)[-1]
        image_fname_noext = osp.splitext(image_fname)[0]
        label_fname = 'gt_' + image_fname_noext + '.txt'
        label_path = osp.join(images_dir, label_fname)
        if not os.path.exists(label_path):
            logger.error('{} does not exist'.format(label_path))
            sys.exit(-1)
        polys = []
        with open(label_path, 'r') as f:
            reader = csv.reader(f)
            for line in reader:
                # strip BOM. \ufeff for python3,  \xef\xbb\bf for python2
                line = [i.strip('\ufeff').strip('\xef\xbb\xbf') for i in line]
                x1, y1, x2, y2, x3, y3, x4, y4 = [float(coord) for coord in line[:8]]
                poly = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
                polys.append(poly)
        polys = np.array(polys)
        if np.any(polys < 0):
            logger.error('{}: negative coords {}'.format(label_path, polys[polys < 0]))
            sys.exit(-1)
        elif np.any(polys[:, :, 0] > w - 1):
            logger.error('{}: invalid width {} > {}'.format(label_path,
                                                            polys[polys[:, :, 0] > w - 1],
                                                            w - 1))
            sys.exit(-1)
        elif np.any(polys[:, :, 1] > h - 1):
            logger.error('{}: invalid height {} > {}'.format(label_path,
                                                             polys[polys[:, :, 1] > h - 1],
                                                             h - 1))
            sys.exit(-1)


def load_annotation(label_path, scale, pad, window):
    """
    load annotations from txt file
    首先确保所有坐标都在 image 的范围内, 0 <= x < w, 0 <= y < h
    Args:
        label_path:
        scale:
        pad:
        window:

    Returns:
        polys (np.array): dtype=np.float32

    """
    polys = []
    notext_polys = []
    small_area_polys = []
    small_height_polys = []
    small_width_polys = []
    if not os.path.exists(label_path):
        logger.error('{} does not exist'.format(label_path))
        sys.exit(-1)
    with open(label_path, 'r') as f:
        reader = csv.reader(f)
        for line in reader:
            # strip BOM. \ufeff for python3,  \xef\xbb\bf for python2
            line = [i.strip('\ufeff').strip('\xef\xbb\xbf') for i in line]
            x1, y1, x2, y2, x3, y3, x4, y4 = [round(scale * float(coord)) for coord in line[:8]]
            label = line[8]
            text_poly = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
            # + pad_left
            text_poly[:, 0] = text_poly[:, 0] + pad[1][0]
            # + pad_top
            text_poly[:, 1] = text_poly[:, 1] + pad[0][0]
            text_poly[:, 0] = np.clip(text_poly[:, 0], window[1], window[3])
            text_poly[:, 1] = np.clip(text_poly[:, 1], window[0], window[2])
            # if np.any(text_poly < 0):
            #     logger.error('{}: negative coords {}'.format(label_path, text_poly))
            #     sys.exit(-1)
            # elif np.any(text_poly[:, 0] > input_size - 1):
            #     logger.error('{}: invalid width {} > {}'.format(label_path,
            #                                                     text_poly[text_poly[:, 0] > input_size - 1],
            #                                                     input_size - 1))
            #     sys.exit(-1)
            # elif np.any(text_poly[:, 1] > input_size - 1):
            #     logger.error('{}: invalid height {} > {}'.format(label_path,
            #                                                      text_poly[text_poly[:, 1] > input_size - 1],
            #                                                      input_size - 1))
            #     sys.exit(-1)
            # else:
            counter_clockwise, clockwise = reorder_vertexes(text_poly)
            area = polygon_area(counter_clockwise)
            text_poly_h = min(np.linalg.norm(text_poly[0] - text_poly[3]),
                              np.linalg.norm(text_poly[1] - text_poly[2]))
            text_poly_w = min(np.linalg.norm(text_poly[0] - text_poly[1]),
                              np.linalg.norm(text_poly[2] - text_poly[3]))
            if label == '*' or label == '###':
                notext_polys.append(text_poly)
            elif area < 64:
                logger.warning('{}: small area of {} is {}'.format(label_path, text_poly, area))
                small_area_polys.append(text_poly)
            elif text_poly_h < 8:
                logger.warning('{}: small height of {} is {}'.format(label_path, text_poly, text_poly_h))
                small_height_polys.append(text_poly)
            elif text_poly_w < 8:
                logger.warning('{}: small width of {} is {}'.format(label_path, text_poly, text_poly_w))
                small_width_polys.append(text_poly)
            else:
                polys.append(text_poly)
        return np.array(polys, dtype=np.float32), np.array(notext_polys, dtype=np.float32), \
               np.array(small_area_polys, dtype=np.float32), \
               np.array(small_height_polys, dtype=np.float32), np.array(small_width_polys, dtype=np.float32)


def show_image(image, name):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.imshow(name, image)


def reorder_vertexes(xy_list):
    """
    先找最小 x 的点, 作为返回值的第 0 个元素
    然后找最小 x 的点对角线的点, 作为返回值的第 2 个元素
    然后对角线下方的点, 作为返回值的第 1 个元素
    然后对角线上方的点, 作为返回值的第 3 个元素
    最后调整把左上方的点作为返回值的第　0 个元素, 其他按逆时针移动
    总结起来就是: 先找到最小 x 的点, 然后按逆时针找到其他三个点, 最后调整把左上方的点作为返回值的第　0 个元素, 其他按逆时针移动
    :param xy_list: shape 为 (4, 2) 的数组,ICPR 的数据集中依次存放左下方的点和按逆时针方向的其他三个点
    :return:
    """
    # reorder_xy_list 的 shape 为 (4, 2)
    reorder_xy_list = np.zeros_like(xy_list)
    #######################################################################
    # 找最小 x 的点,其下标复制给 first_v
    #######################################################################
    # determine the first point with the smallest x, if two has same x, choose that with smallest y,
    # 四个点的坐标按 x 进行排序, 如果 x 相等, 按 y 排序
    # np.argsort 返回结果的 shape 和原数组一样, 每个元素的值表示该位置对应的原数组中的元素的序号
    # 参见 https://docs.scipy.org/doc/numpy-1.15.1/reference/generated/numpy.argsort.html
    ordered_idxes = np.argsort(xy_list, axis=0)
    # 有最小 x 的元素在原数组中的序号, 第一个 0 能获得最小 x 和最小 y 分别对应原数组的下标, 第二个表示只获取最小 x 的坐标
    xmin1_idx = ordered_idxes[0, 0]
    # 有倒数第二小的 x 的元素在原数组的序号
    xmin2_idx = ordered_idxes[1, 0]
    # 如果最小的两个 x 相等, 比较 y
    if xy_list[xmin1_idx, 0] == xy_list[xmin2_idx, 0]:
        if xy_list[xmin1_idx, 1] <= xy_list[xmin2_idx, 1]:
            reorder_xy_list[0] = xy_list[xmin1_idx]
            first_vertex_idx = xmin1_idx
        else:
            reorder_xy_list[0] = xy_list[xmin2_idx]
            first_vertex_idx = xmin2_idx
    else:
        # 把有最小 x 的元素放在  reorder_xy_list 的下标 0 的位置
        reorder_xy_list[0] = xy_list[xmin1_idx]
        first_vertex_idx = xmin1_idx
    #######################################################################################
    # 找到第一个顶点的对角线的点
    #######################################################################################
    # connect the first point to others, the third point on the other side of the line with the middle slope
    other_vertex_idxes = list(range(4))
    # 除去第一个顶点后，其他三个顶点分别对应原数组的下标
    other_vertex_idxes.remove(first_vertex_idx)
    # k 存放第一个点和其他点的斜率
    k = np.zeros((len(other_vertex_idxes),))
    for i, other_vertex_idx in enumerate(other_vertex_idxes):
        k[i] = (xy_list[other_vertex_idx, 1] - xy_list[first_vertex_idx, 1]) \
               / (xy_list[other_vertex_idx, 0] - xy_list[first_vertex_idx, 0] + 1e-7)
    # k_mid_idx 是三个斜率的中间值在 k 中的下标
    k_mid_idx = np.argsort(k)[1]
    # 三个斜率的中间值
    k_mid = k[k_mid_idx]
    third_vertex_idx = other_vertex_idxes[k_mid_idx]
    reorder_xy_list[2] = xy_list[third_vertex_idx]
    #######################################################################################
    # 找到其他两个点
    #######################################################################################
    # determine the second point which on the bigger side of the middle line
    other_vertex_idxes.remove(third_vertex_idx)
    # 对角线的　y = k * x + b 的　b
    b_mid = xy_list[first_vertex_idx, 1] - k_mid * xy_list[first_vertex_idx, 0]
    for i, other_vertex_idx in enumerate(other_vertex_idxes):
        # delta_y = y - (k * x + b)
        # 根据　delta_y 判断该点是在对角线上方还是下方
        # 下方的点作为第 1 个点, 上方的点作为第 3 个点, 这样得到的四个点是就按逆时针排列了
        delta_y = xy_list[other_vertex_idx, 1] - (k_mid * xy_list[other_vertex_idx, 0] + b_mid)
        # delta_y 大于 0 表示在直线的下方, 注意用的是 image 的坐标系, 左上方的点作为 (0, 0)
        if delta_y > 0:
            second_vertex_idx = other_vertex_idx
        else:
            fourth_vertex_idx = other_vertex_idx
    reorder_xy_list[1] = xy_list[second_vertex_idx]
    reorder_xy_list[3] = xy_list[fourth_vertex_idx]
    #######################################################################################
    # 把左上方的点作为第一个点,按逆时针得到其他点
    #######################################################################################
    # compare slope of 13 and 24, determine the final order
    k13 = k_mid
    k24 = (xy_list[second_vertex_idx, 1] - xy_list[fourth_vertex_idx, 1]) / (
            xy_list[second_vertex_idx, 0] - xy_list[fourth_vertex_idx, 0] + 1e-7)
    if k13 < k24:
        tmp_x, tmp_y = reorder_xy_list[3, 0], reorder_xy_list[3, 1]
        for i in range(2, -1, -1):
            reorder_xy_list[i + 1] = reorder_xy_list[i]
        reorder_xy_list[0, 0], reorder_xy_list[0, 1] = tmp_x, tmp_y
    return reorder_xy_list, reorder_xy_list[[0, 3, 2, 1]]


def polygon_area(poly):
    """
    compute area of a polygon. 如果按逆时针排序, 返回的面积是正的, 否则返回的面积是负的.
    Args:
        poly (np.array or list):
    Returns:
    """
    edge = [
        (poly[1][0] - poly[0][0]) * (poly[1][1] + poly[0][1]),
        (poly[2][0] - poly[1][0]) * (poly[2][1] + poly[1][1]),
        (poly[3][0] - poly[2][0]) * (poly[3][1] + poly[2][1]),
        (poly[0][0] - poly[3][0]) * (poly[0][1] + poly[3][1])
    ]
    return np.sum(edge) / 2.


def check_and_validate_polys(FLAGS, polys, tags, size):
    """
    check so that the text poly is in the same direction and also filter some invalid polygons

    Args:
        FLAGS (argparse.Namespace):
        polys (np.array):
        tags (np.array):
        size (tuple): (h, w)

    Returns:
        validated_polys (np.array): 过滤后的 text_polys
        validated_tag (np.array): 过滤后的 text_tags

    """
    (h, w) = size
    if polys.shape[0] == 0:
        return polys
    polys[:, :, 0] = np.clip(polys[:, :, 0], 0, w - 1)
    polys[:, :, 1] = np.clip(polys[:, :, 1], 0, h - 1)

    validated_polys = []
    validated_tags = []
    for poly, tag in zip(polys, tags):
        p_area = polygon_area(poly)
        if abs(p_area) < 1:
            print('invalid poly')
            continue
        if p_area > 0:
            print('poly in wrong direction')
            poly = poly[(0, 3, 2, 1), :]
        validated_polys.append(poly)
        validated_tags.append(tag)
    return np.array(validated_polys), np.array(validated_tags)


def crop_area(FLAGS, im, polys, tags, crop_background=False, max_tries=50):
    """
    make random crop from the input image

    首先创建两个 0 数组 w_array 和 h_array, size 分别等于宽和高, 有 text 的位置设为 1
    从 w_array 中取两个值为 0 的点作为 min_x, max_x, 从 h_array 中取两个值为 0 的点作为 min_y, max_y
    如果 crop 即 [min_y:max_y, min_x:max_x] 太小, 重试
    然后把 polys 中存在于 [min_y:max_y, min_x:max_x] 的 id 保存到 selected_polys
    如果 selected_polys 为空, 表示没找到符合条件的 crop
        如果设置了 crop_background 那么就返回这个 crop,
        如果没有设置, 重试, 最大重试次数为 max_tries
    如果达到最大重试次数还是没有找到, 返回原来的 img, polys, tags
    Args:
        FLAGS:
        im:
        polys:
        tags:
        crop_background:
        max_tries:

    Returns:

    """
    h, w, _ = im.shape
    pad_h = h // 10
    pad_w = w // 10
    # 相当于一根线, 有 text 的地方涂上颜色, crop 时从没有颜色的地方取两个点, 作为 min_y 和 max_y
    h_array = np.zeros((h + pad_h * 2), dtype=np.int32)
    w_array = np.zeros((w + pad_w * 2), dtype=np.int32)
    for poly in polys:
        poly = np.round(poly, decimals=0).astype(np.int32)
        minx = np.min(poly[:, 0])
        maxx = np.max(poly[:, 0])
        w_array[minx + pad_w:maxx + pad_w] = 1
        miny = np.min(poly[:, 1])
        maxy = np.max(poly[:, 1])
        h_array[miny + pad_h:maxy + pad_h] = 1
    # ensure the cropped area not across a text
    h_axis = np.where(h_array == 0)[0]
    w_axis = np.where(w_array == 0)[0]
    if len(h_axis) == 0 or len(w_axis) == 0:
        # FIXME: 这是不可能的, 因为有 pad 的存在
        return im, polys, tags
    for i in range(max_tries):
        xx = np.random.choice(w_axis, size=2)
        xmin = np.min(xx) - pad_w
        xmax = np.max(xx) - pad_w
        xmin = np.clip(xmin, 0, w - 1)
        xmax = np.clip(xmax, 0, w - 1)
        yy = np.random.choice(h_axis, size=2)
        ymin = np.min(yy) - pad_h
        ymax = np.max(yy) - pad_h
        ymin = np.clip(ymin, 0, h - 1)
        ymax = np.clip(ymax, 0, h - 1)
        if xmax - xmin < FLAGS.min_crop_side_ratio * w or ymax - ymin < FLAGS.min_crop_side_ratio * h:
            # area too small
            continue
        if polys.shape[0] != 0:
            # (n, 4)
            poly_axis_in_area = (polys[:, :, 0] >= xmin) & (polys[:, :, 0] <= xmax) \
                                & (polys[:, :, 1] >= ymin) & (polys[:, :, 1] <= ymax)
            # sum == 4 表示 4 个顶点都符合要求
            selected_polys = np.where(np.sum(poly_axis_in_area, axis=1) == 4)[0]
        else:
            selected_polys = []
        if len(selected_polys) == 0:
            # no text in this area
            if crop_background:
                return im[ymin:ymax + 1, xmin:xmax + 1, :], polys[selected_polys], tags[selected_polys]
            else:
                continue
        im = im[ymin:ymax + 1, xmin:xmax + 1, :]
        polys = polys[selected_polys]
        tags = tags[selected_polys]
        polys[:, :, 0] -= xmin
        polys[:, :, 1] -= ymin
        return im, polys, tags

    return im, polys, tags


def shrink_poly(poly, r):
    """
    fit a poly inside the origin poly, maybe bugs here...
    used for generating the score map
    Args:
        poly (np.array): the text poly
        r (list): min length around vertex

    Returns:
        poly (np.array): shrinked poly
    """
    # shrink ratio
    R = 0.3
    # find the longer pair
    if np.linalg.norm(poly[0] - poly[1]) + np.linalg.norm(poly[2] - poly[3]) > \
            np.linalg.norm(poly[0] - poly[3]) + np.linalg.norm(poly[1] - poly[2]):
        # first move (p0, p1), (p2, p3), then (p0, p3), (p1, p2)
        # p0, p1
        theta = np.arctan2((poly[1][1] - poly[0][1]), (poly[1][0] - poly[0][0]))
        poly[0][0] += R * r[0] * np.cos(theta)
        poly[0][1] += R * r[0] * np.sin(theta)
        poly[1][0] -= R * r[1] * np.cos(theta)
        poly[1][1] -= R * r[1] * np.sin(theta)
        # p2, p3
        theta = np.arctan2((poly[2][1] - poly[3][1]), (poly[2][0] - poly[3][0]))
        poly[3][0] += R * r[3] * np.cos(theta)
        poly[3][1] += R * r[3] * np.sin(theta)
        poly[2][0] -= R * r[2] * np.cos(theta)
        poly[2][1] -= R * r[2] * np.sin(theta)
        # p0, p3
        theta = np.arctan2((poly[3][0] - poly[0][0]), (poly[3][1] - poly[0][1]))
        poly[0][0] += R * r[0] * np.sin(theta)
        poly[0][1] += R * r[0] * np.cos(theta)
        poly[3][0] -= R * r[3] * np.sin(theta)
        poly[3][1] -= R * r[3] * np.cos(theta)
        # p1, p2
        theta = np.arctan2((poly[2][0] - poly[1][0]), (poly[2][1] - poly[1][1]))
        poly[1][0] += R * r[1] * np.sin(theta)
        poly[1][1] += R * r[1] * np.cos(theta)
        poly[2][0] -= R * r[2] * np.sin(theta)
        poly[2][1] -= R * r[2] * np.cos(theta)
    else:
        # p0, p3
        theta = np.arctan2((poly[3][0] - poly[0][0]), (poly[3][1] - poly[0][1]))
        poly[0][0] += R * r[0] * np.sin(theta)
        poly[0][1] += R * r[0] * np.cos(theta)
        poly[3][0] -= R * r[3] * np.sin(theta)
        poly[3][1] -= R * r[3] * np.cos(theta)
        # p1, p2
        theta = np.arctan2((poly[2][0] - poly[1][0]), (poly[2][1] - poly[1][1]))
        poly[1][0] += R * r[1] * np.sin(theta)
        poly[1][1] += R * r[1] * np.cos(theta)
        poly[2][0] -= R * r[2] * np.sin(theta)
        poly[2][1] -= R * r[2] * np.cos(theta)
        # p0, p1
        theta = np.arctan2((poly[1][1] - poly[0][1]), (poly[1][0] - poly[0][0]))
        poly[0][0] += R * r[0] * np.cos(theta)
        poly[0][1] += R * r[0] * np.sin(theta)
        poly[1][0] -= R * r[1] * np.cos(theta)
        poly[1][1] -= R * r[1] * np.sin(theta)
        # p2, p3
        theta = np.arctan2((poly[2][1] - poly[3][1]), (poly[2][0] - poly[3][0]))
        poly[3][0] += R * r[3] * np.cos(theta)
        poly[3][1] += R * r[3] * np.sin(theta)
        poly[2][0] -= R * r[2] * np.cos(theta)
        poly[2][1] -= R * r[2] * np.sin(theta)
    return poly


def point_dist_to_line(p1, p2, p3):
    # compute the distance from p3 to p1-p2
    # np.cross 算出 (p2 - p1) 和 (p1 - p3) 两个向量组成的平行四边形的面积
    # 面积 / 底边长 = 高, 底边长是 p1-p2, 高就是 p3 到 p1-p2 边的距离
    return np.linalg.norm(np.cross(p2 - p1, p1 - p3)) / np.linalg.norm(p2 - p1)


def fit_line(p1, p2):
    """
    fit a line ax+by+c = 0
    Args:
        p1:
        p2:

    Returns:
        [a, b, c]
    """
    if p1[0] == p1[1]:
        # x 相等, 竖直线, 无法求斜率, 得到的先就是 x = p1[0]
        return [1., 0., -p1[0]]
    else:
        [k, b] = np.polyfit(p1, p2, deg=1)
        return [k, -1., b]


def line_cross_point(line1, line2):
    # line1 0= ax+by+c, compute the cross point of line1 and line2
    if line1[0] != 0 and line1[0] == line2[0]:
        print('Cross point does not exist')
        return None
    if line1[0] == 0 and line2[0] == 0:
        print('Cross point does not exist')
        return None
    if line1[1] == 0:
        x = -line1[2]
        y = line2[0] * x + line2[2]
    elif line2[1] == 0:
        x = -line2[2]
        y = line1[0] * x + line1[2]
    else:
        k1, _, b1 = line1
        k2, _, b2 = line2
        x = -(b1 - b2) / (k1 - k2)
        y = k1 * x + b1
    return np.array([x, y], dtype=np.float32)


def line_verticle(line, point):
    # get the verticle line from line across point
    if line[1] == 0:
        verticle = [0, -1, point[1]]
    else:
        if line[0] == 0:
            verticle = [1, 0, -point[0]]
        else:
            # 根据垂直的两条线的斜率的积为 -1 可以推导出来
            # Note: 数组的第一个元素是 k, 即当 b = -1 时 ax + by + c = 0 中的 a
            verticle = [-1. / line[0], -1, point[1] - (-1 / line[0] * point[0])]
    return verticle


def rectangle_from_parallelogram(poly):
    """
    fit a rectangle from a parallelogram

    Args:
        poly:

    Returns:

    """
    p0, p1, p2, p3 = poly
    # 参考 https://baike.baidu.com/item/%E7%82%B9%E7%A7%AF/9648528?fr=aladdin 的代数定义推到几何定义
    # 余弦定理求两个向量的夹角
    angle_p0 = np.arccos(np.dot(p1 - p0, p3 - p0) / (np.linalg.norm(p0 - p1) * np.linalg.norm(p3 - p0)))
    # 如果 p0 是锐角
    if angle_p0 < 0.5 * np.pi:
        # 判断边的长度是为了获取面积最小的外接矩形, 而不仅仅是外接矩形
        if np.linalg.norm(p0 - p1) > np.linalg.norm(p0 - p3):
            # new p3
            p2p3 = fit_line([p2[0], p3[0]], [p2[1], p3[1]])
            p2p3_verticle = line_verticle(p2p3, p0)
            new_p3 = line_cross_point(p2p3, p2p3_verticle)
            # new p1
            p0p1 = fit_line([p0[0], p1[0]], [p0[1], p1[1]])
            p0p1_verticle = line_verticle(p0p1, p2)
            new_p1 = line_cross_point(p0p1, p0p1_verticle)
            return np.array([p0, new_p1, p2, new_p3], dtype=np.float32)
        else:
            # new p1
            p1p2 = fit_line([p1[0], p2[0]], [p1[1], p2[1]])
            p1p2_verticle = line_verticle(p1p2, p0)
            new_p1 = line_cross_point(p1p2, p1p2_verticle)
            # new p3
            p0p3 = fit_line([p0[0], p3[0]], [p0[1], p3[1]])
            p0p3_verticle = line_verticle(p0p3, p2)
            new_p3 = line_cross_point(p0p3, p0p3_verticle)
            return np.array([p0, new_p1, p2, new_p3], dtype=np.float32)
    else:
        if np.linalg.norm(p0 - p1) > np.linalg.norm(p0 - p3):
            # new p2
            p2p3 = fit_line([p2[0], p3[0]], [p2[1], p3[1]])
            p2p3_verticle = line_verticle(p2p3, p1)
            new_p2 = line_cross_point(p2p3, p2p3_verticle)
            # new p0
            p0p1 = fit_line([p0[0], p1[0]], [p0[1], p1[1]])
            p0p1_verticle = line_verticle(p0p1, p3)
            new_p0 = line_cross_point(p0p1, p0p1_verticle)
            return np.array([new_p0, p1, new_p2, p3], dtype=np.float32)
        else:
            # new p0
            p0p3 = fit_line([p0[0], p3[0]], [p0[1], p3[1]])
            p0p3_verticle = line_verticle(p0p3, p1)
            new_p0 = line_cross_point(p0p3, p0p3_verticle)
            # new p2
            p1p2 = fit_line([p1[0], p2[0]], [p1[1], p2[1]])
            p1p2_verticle = line_verticle(p1p2, p3)
            new_p2 = line_cross_point(p1p2, p1p2_verticle)
            return np.array([new_p0, p1, new_p2, p3], dtype=np.float32)


def sort_rectangle(poly):
    # sort the four coordinates of the polygon, points in poly should be sorted clockwise
    # First find the lowest point 就是 y 值最大的点
    p_lowest = np.argmax(poly[:, 1])
    if np.count_nonzero(poly[:, 1] == poly[p_lowest, 1]) == 2:
        # if the bottom line is parallel to x-axis, then p0 must be the upper-left corner
        p0_index = np.argmin(np.sum(poly, axis=1))
        p1_index = (p0_index + 1) % 4
        p2_index = (p0_index + 2) % 4
        p3_index = (p0_index + 3) % 4
        return poly[[p0_index, p1_index, p2_index, p3_index]], 0.
    else:
        # find the point that sits right to the lowest point
        p_lowest_right = (p_lowest - 1) % 4
        angle = np.arctan(
            -(poly[p_lowest][1] - poly[p_lowest_right][1]) / (poly[p_lowest][0] - poly[p_lowest_right][0]))
        # assert angle > 0
        if angle <= 0:
            print(angle, poly[p_lowest], poly[p_lowest_right])
        if angle / np.pi * 180 > 45:
            # 认为这个点为 p2 - this point is p2
            p2_index = p_lowest
            p1_index = (p2_index - 1) % 4
            p0_index = (p2_index - 2) % 4
            p3_index = (p2_index + 1) % 4
            # 角度为 (-45, 0)
            # 角度为 (0, 45), 逆时针旋转这个角度变成水平的, 即 p0-p1 是水平的, 在上方, p2-p3 也是水平的, 在下方
            return poly[[p0_index, p1_index, p2_index, p3_index]], -(np.pi / 2 - angle)
        else:
            # 认为这个点为 p3 - this point is p3
            p3_index = p_lowest
            p0_index = (p3_index + 1) % 4
            p1_index = (p3_index + 2) % 4
            p2_index = (p3_index + 3) % 4
            # 角度为 (0, 45), 顺时针旋转这个角度变成水平的, 即 p0-p1 是水平的, 在上方, p2-p3 也是水平的, 在下方
            return poly[[p0_index, p1_index, p2_index, p3_index]], angle


def restore_rectangle_rbox(origin, geometry):
    d = geometry[:, :4]
    angle = geometry[:, 4]
    # for angle > 0
    origin_0 = origin[angle >= 0]
    d_0 = d[angle >= 0]
    angle_0 = angle[angle >= 0]
    if origin_0.shape[0] > 0:
        # (10, n)
        p = np.array([np.zeros(d_0.shape[0]), -d_0[:, 0] - d_0[:, 2],
                      d_0[:, 1] + d_0[:, 3], -d_0[:, 0] - d_0[:, 2],
                      d_0[:, 1] + d_0[:, 3], np.zeros(d_0.shape[0]),
                      np.zeros(d_0.shape[0]), np.zeros(d_0.shape[0]),
                      d_0[:, 3], -d_0[:, 2]])
        # (10, n) transpose 后变成 (n, 10) 再 reshape 变成 (n, 5, 2)
        # 假设 d0 + d2 = h 为矩形的高, d1 + d3 = w 为矩形的宽
        # p = [[0, -h], [w, -h], [w, 0], [0, 0], [距离左边的距离, -距离底边的距离]]
        p = p.transpose((1, 0)).reshape((-1, 5, 2))
        # (2, n), transpose 后变成 (n, 2)
        rotate_matrix_x = np.array([np.cos(angle_0), np.sin(angle_0)]).transpose((1, 0))
        # (n, 5, 2)
        rotate_matrix_x = np.repeat(rotate_matrix_x, 5, axis=1).reshape(-1, 2, 5).transpose((0, 2, 1))
        # (n, 2)
        rotate_matrix_y = np.array([-np.sin(angle_0), np.cos(angle_0)]).transpose((1, 0))
        # (n, 5, 2)
        rotate_matrix_y = np.repeat(rotate_matrix_y, 5, axis=1).reshape(-1, 2, 5).transpose((0, 2, 1))
        # [[-h * sin(theta)], [w * cos(theta) - h * sin(theta)], [w * cos(theta)], [0]]
        p_rotate_x = np.sum(rotate_matrix_x * p, axis=2)[:, :, np.newaxis]  # N*5*1
        # [[-h * cos(theta)], [-w * sin(theta) - h * cos(theta)], [-w * sin(theta)], [0]]
        p_rotate_y = np.sum(rotate_matrix_y * p, axis=2)[:, :, np.newaxis]  # N*5*1
        p_rotate = np.concatenate([p_rotate_x, p_rotate_y], axis=2)  # N*5*2

        # p3 的坐标
        p3_in_origin = origin_0 - p_rotate[:, 4, :]
        # (n, 2), 最后一维是 (p3_x - h * sin(theta), p3_y - h * cos(theta))
        new_p0 = p_rotate[:, 0, :] + p3_in_origin
        # (n, 2), 最后一维是 (p3_x + w * cos(theta) - h * sin(theta, p3_y - w * sin(theta) - h * cos(theta))
        new_p1 = p_rotate[:, 1, :] + p3_in_origin
        # (n, 2), 最后一维是 (p3_x + w * cos(theta), p3_y - w * sin(theta))
        new_p2 = p_rotate[:, 2, :] + p3_in_origin
        # (n, 2), p_rotate[:, 2, :] 是 [[0, 0]...[0, 0]]
        new_p3 = p_rotate[:, 3, :] + p3_in_origin

        new_p_0 = np.concatenate([new_p0[:, np.newaxis, :], new_p1[:, np.newaxis, :],
                                  new_p2[:, np.newaxis, :], new_p3[:, np.newaxis, :]], axis=1)  # N*4*2
    else:
        new_p_0 = np.zeros((0, 4, 2))
    # for angle < 0
    origin_1 = origin[angle < 0]
    d_1 = d[angle < 0]
    angle_1 = angle[angle < 0]
    if origin_1.shape[0] > 0:
        p = np.array([-d_1[:, 1] - d_1[:, 3], -d_1[:, 0] - d_1[:, 2],
                      np.zeros(d_1.shape[0]), -d_1[:, 0] - d_1[:, 2],
                      np.zeros(d_1.shape[0]), np.zeros(d_1.shape[0]),
                      -d_1[:, 1] - d_1[:, 3], np.zeros(d_1.shape[0]),
                      -d_1[:, 1], -d_1[:, 2]])
        p = p.transpose((1, 0)).reshape((-1, 5, 2))  # N*5*2

        rotate_matrix_x = np.array([np.cos(-angle_1), -np.sin(-angle_1)]).transpose((1, 0))
        rotate_matrix_x = np.repeat(rotate_matrix_x, 5, axis=1).reshape(-1, 2, 5).transpose((0, 2, 1))  # N*5*2

        rotate_matrix_y = np.array([np.sin(-angle_1), np.cos(-angle_1)]).transpose((1, 0))
        rotate_matrix_y = np.repeat(rotate_matrix_y, 5, axis=1).reshape(-1, 2, 5).transpose((0, 2, 1))

        p_rotate_x = np.sum(rotate_matrix_x * p, axis=2)[:, :, np.newaxis]  # N*5*1
        p_rotate_y = np.sum(rotate_matrix_y * p, axis=2)[:, :, np.newaxis]  # N*5*1

        p_rotate = np.concatenate([p_rotate_x, p_rotate_y], axis=2)  # N*5*2

        p3_in_origin = origin_1 - p_rotate[:, 4, :]
        new_p0 = p_rotate[:, 0, :] + p3_in_origin  # N*2
        new_p1 = p_rotate[:, 1, :] + p3_in_origin
        new_p2 = p_rotate[:, 2, :] + p3_in_origin
        new_p3 = p_rotate[:, 3, :] + p3_in_origin

        new_p_1 = np.concatenate([new_p0[:, np.newaxis, :], new_p1[:, np.newaxis, :],
                                  new_p2[:, np.newaxis, :], new_p3[:, np.newaxis, :]], axis=1)  # N*4*2
    else:
        new_p_1 = np.zeros((0, 4, 2))
    return np.concatenate([new_p_0, new_p_1])


def restore_rectangle(origin, geometry):
    return restore_rectangle_rbox(origin, geometry)


def generate_rbox(input_size, polys):
    # 不同的 poly 用不同的值来填充
    shrinked_poly_mask = np.zeros((input_size, input_size), dtype=np.uint8)
    orig_poly_mask = np.zeros((input_size, input_size), dtype=np.uint8)
    score_map = np.zeros((input_size, input_size), dtype=np.uint8)
    geo_map = np.zeros((input_size, input_size, 5), dtype=np.float32)
    for poly_idx, poly in enumerate(polys):
        r = [None, None, None, None]
        for i in range(4):
            r[i] = min(np.linalg.norm(poly[i] - poly[(i + 1) % 4]),
                       np.linalg.norm(poly[i] - poly[(i - 1) % 4]))
        # score map
        shrinked_poly = shrink_poly(poly.copy(), r).astype(np.int32)[np.newaxis, :, :]
        cv2.fillPoly(score_map, shrinked_poly, 1)
        cv2.fillPoly(shrinked_poly_mask, shrinked_poly, poly_idx + 1)
        cv2.fillPoly(orig_poly_mask, poly.astype(np.int32)[np.newaxis, :, :], 1)
        yx_in_poly = np.argwhere(shrinked_poly_mask == (poly_idx + 1))
        # if geometry == 'RBOX':
        # generate a parallelogram for any combination of two vertices
        fitted_parallelograms = []
        for i in range(4):
            p0 = poly[i]
            p1 = poly[(i + 1) % 4]
            p2 = poly[(i + 2) % 4]
            p3 = poly[(i + 3) % 4]
            edge = fit_line([p0[0], p1[0]], [p0[1], p1[1]])
            backward_edge = fit_line([p0[0], p3[0]], [p0[1], p3[1]])
            forward_edge = fit_line([p1[0], p2[0]], [p1[1], p2[1]])
            if point_dist_to_line(p0, p1, p2) > point_dist_to_line(p0, p1, p3):
                # parallel line through p2
                if edge[1] == 0:
                    edge_opposite = [1, 0, -p2[0]]
                else:
                    edge_opposite = [edge[0], -1, p2[1] - edge[0] * p2[0]]
            else:
                # parallel line through p3
                if edge[1] == 0:
                    edge_opposite = [1, 0, -p3[0]]
                else:
                    edge_opposite = [edge[0], -1, p3[1] - edge[0] * p3[0]]
            # move forward edge
            new_p1 = p1
            new_p2 = line_cross_point(forward_edge, edge_opposite)
            if point_dist_to_line(p1, new_p2, p0) > point_dist_to_line(p1, new_p2, p3):
                # across p0
                if forward_edge[1] == 0:
                    forward_opposite = [1, 0, -p0[0]]
                else:
                    forward_opposite = [forward_edge[0], -1, p0[1] - forward_edge[0] * p0[0]]
            else:
                # across p3
                if forward_edge[1] == 0:
                    forward_opposite = [1, 0, -p3[0]]
                else:
                    forward_opposite = [forward_edge[0], -1, p3[1] - forward_edge[0] * p3[0]]
            new_p0 = line_cross_point(forward_opposite, edge)
            new_p3 = line_cross_point(forward_opposite, edge_opposite)
            fitted_parallelograms.append([new_p0, new_p1, new_p2, new_p3, new_p0])
            # or move backward edge
            new_p0 = p0
            new_p3 = line_cross_point(backward_edge, edge_opposite)
            if point_dist_to_line(p0, p3, p1) > point_dist_to_line(p0, p3, p2):
                # across p1
                if backward_edge[1] == 0:
                    backward_opposite = [1, 0, -p1[0]]
                else:
                    backward_opposite = [backward_edge[0], -1, p1[1] - backward_edge[0] * p1[0]]
            else:
                # across p2
                if backward_edge[1] == 0:
                    backward_opposite = [1, 0, -p2[0]]
                else:
                    backward_opposite = [backward_edge[0], -1, p2[1] - backward_edge[0] * p2[0]]
            new_p1 = line_cross_point(backward_opposite, edge)
            new_p2 = line_cross_point(backward_opposite, edge_opposite)
            fitted_parallelograms.append([new_p0, new_p1, new_p2, new_p3, new_p0])
        # 8 个平行四边形的面积
        areas = [Polygon(t).area for t in fitted_parallelograms]
        # 面积最小的平行四边形的顶点坐标, -1 表示把最后一个元素 new_p0 去掉
        fitted_parallelograms = np.array(fitted_parallelograms)
        parallelogram = np.array(fitted_parallelograms[np.argmin(areas)][:-1], dtype=np.float32)
        # sort this polygon
        parallelogram_coord_sum = np.sum(parallelogram, axis=1)
        min_coord_idx = np.argmin(parallelogram_coord_sum)
        parallelogram = parallelogram[
            [min_coord_idx, (min_coord_idx + 1) % 4, (min_coord_idx + 2) % 4, (min_coord_idx + 3) % 4]]
        # 获得面积最小的外接矩形
        rectange = rectangle_from_parallelogram(parallelogram)
        rectange, rotate_angle = sort_rectangle(rectange)

        p0_rect, p1_rect, p2_rect, p3_rect = rectange
        for y, x in yx_in_poly:
            point = np.array([x, y], dtype=np.float32)
            # top
            geo_map[y, x, 0] = point_dist_to_line(p0_rect, p1_rect, point)
            # right
            geo_map[y, x, 1] = point_dist_to_line(p1_rect, p2_rect, point)
            # down
            geo_map[y, x, 2] = point_dist_to_line(p2_rect, p3_rect, point)
            # left
            geo_map[y, x, 3] = point_dist_to_line(p3_rect, p0_rect, point)
            # angle
            geo_map[y, x, 4] = rotate_angle

    shrinked_poly_mask = (shrinked_poly_mask > 0).astype('uint8')
    # 原 poly 和 shrinked poly 之间的像素设为 0, 计算 loss 的时候忽略这部分
    text_region_boundary_training_mask = 1 - (orig_poly_mask - shrinked_poly_mask)

    return score_map, geo_map, text_region_boundary_training_mask


def all(iterable):
    for element in iterable:
        if not element:
            return False
    return True


def get_text_file(image_file):
    txt_file = image_file.replace(os.path.basename(image_file).split('.')[1], 'txt')
    txt_file_name = txt_file.split('/')[-1]
    txt_file = txt_file.replace(txt_file_name, 'gt_' + txt_file_name)
    return txt_file


def pad_image(img, input_size, is_train):
    new_h, new_w, _ = img.shape
    max_h_w_i = np.max([new_h, new_w, input_size])
    img_padded = np.zeros((max_h_w_i, max_h_w_i, 3), dtype=np.uint8)
    if is_train:
        # adam
        if max_h_w_i == new_h:
            shift_h = 0
        else:
            shift_h = np.random.randint(max_h_w_i - new_h)
        if max_h_w_i == new_w:
            shift_w = 0
        else:
            shift_w = np.random.randint(max_h_w_i - new_w)
        # shift_h = np.random.randint(max_h_w_i - new_h + 1)
        # shift_w = np.random.randint(max_h_w_i - new_w + 1)
    else:
        shift_h = (max_h_w_i - new_h) // 2
        shift_w = (max_h_w_i - new_w) // 2
    img_padded[shift_h:new_h + shift_h, shift_w:new_w + shift_w, :] = img.copy()
    img = img_padded
    return img, shift_h, shift_w


def resize_image(img, text_polys, input_size, shift_h, shift_w):
    new_h, new_w, _ = img.shape
    img = cv2.resize(img, dsize=(input_size, input_size))
    # pad and resize text polygons
    resize_ratio_3_x = input_size / float(new_w)
    resize_ratio_3_y = input_size / float(new_h)
    text_polys[:, :, 0] += shift_w
    text_polys[:, :, 1] += shift_h
    text_polys[:, :, 0] *= resize_ratio_3_x
    text_polys[:, :, 1] *= resize_ratio_3_y
    return img, text_polys


def generator(h5_path, dataset_size, batch_size):
    h5_dataset = h5py.File(h5_path, 'r')
    images = h5_dataset['images']
    score_maps = h5_dataset['score_maps']
    geo_maps = h5_dataset['geo_maps']
    text_region_boundary_masks = h5_dataset['text_region_boundary_masks']
    indices = np.arange(0, dataset_size)
    np.random.shuffle(indices)
    current_idx = 0
    while True:
        if current_idx >= dataset_size:
            current_idx = 0
            np.random.shuffle(indices)
        batch_indices = indices[current_idx: current_idx + batch_size]
        batch_images = np.array([images[i] for i in batch_indices])
        batch_score_maps = np.array([score_maps[i, ::4, ::4, :] for i in batch_indices])
        batch_geo_maps = np.array([geo_maps[i, ::4, ::4, :] for i in batch_indices])
        batch_text_region_boundary_masks = np.array([text_region_boundary_masks[i, ::4, ::4, :] for i in batch_indices])
        current_idx += batch_size
        yield [batch_images, batch_text_region_boundary_masks, batch_score_maps], [batch_score_maps, batch_geo_maps]


def val_generator(FLAGS, idx=None, is_train=False):
    image_list = np.array(get_images(FLAGS.validation_data_path))
    if not idx is None:
        image_list = image_list[idx]
    print('{} validation images in {}'.format(
        image_list.shape[0], FLAGS.training_data_path))
    index = np.arange(0, image_list.shape[0])
    epoch = 1
    while True:
        np.random.shuffle(index)
        images = []
        image_fns = []
        score_maps = []
        geo_maps = []
        overly_small_text_region_training_masks = []
        text_region_boundary_training_masks = []
        for i in index:
            try:
                im_fn = image_list[i]
                im = cv2.imread(im_fn)
                h, w, _ = im.shape
                txt_fn = get_text_file(im_fn)
                if not os.path.exists(txt_fn):
                    print('text file {} does not exists'.format(txt_fn))
                    continue

                text_polys, text_tags = load_annotation(txt_fn)

                text_polys, text_tags = check_and_validate_polys(FLAGS, text_polys, text_tags, (h, w))

                im, shift_h, shift_w = pad_image(im, FLAGS.input_size, is_train)
                im, text_polys = resize_image(im, text_polys, FLAGS.input_size, shift_h, shift_w)
                new_h, new_w, _ = im.shape
                score_map, geo_map, overly_small_text_region_training_mask, text_region_boundary_training_mask = generate_rbox(
                    FLAGS, (new_h, new_w), text_polys, text_tags)

                im = (im / 127.5) - 1.
                images.append(im[:, :, ::-1].astype(np.float32))
                image_fns.append(im_fn)
                score_maps.append(score_map[::4, ::4, np.newaxis].astype(np.float32))
                geo_maps.append(geo_map[::4, ::4, :].astype(np.float32))
                overly_small_text_region_training_masks.append(
                    overly_small_text_region_training_mask[::4, ::4, np.newaxis].astype(np.float32))
                text_region_boundary_training_masks.append(
                    text_region_boundary_training_mask[::4, ::4, np.newaxis].astype(np.float32))

                if len(images) == FLAGS.batch_size:
                    yield [np.array(images), np.array(overly_small_text_region_training_masks),
                           np.array(text_region_boundary_training_masks), np.array(score_maps)], [np.array(score_maps),
                                                                                                  np.array(geo_maps)]
                    images = []
                    image_fns = []
                    score_maps = []
                    geo_maps = []
                    overly_small_text_region_training_masks = []
                    text_region_boundary_training_masks = []
            except Exception as e:
                import traceback
                traceback.print_exc()
                continue
        epoch += 1


def count_samples(FLAGS):
    if sys.version_info >= (3, 0):
        return len([f for f in next(os.walk(FLAGS.training_data_path))[2] if f[-4:] == ".jpg"])
    else:
        return len([f for f in os.walk(FLAGS.training_data_path).next()[2] if f[-4:] == ".jpg"])


def load_data_process(FLAGS, image_file, is_train):
    try:
        img = cv2.imread(image_file)
        h, w, _ = img.shape
        txt_file = get_text_file(image_file)
        if not os.path.exists(txt_file):
            print('text file {} does not exists'.format(txt_file))

        text_polys, text_tags = load_annotation(txt_file)
        text_polys, text_tags = check_and_validate_polys(FLAGS, text_polys, text_tags, (h, w))
        img, shift_h, shift_w = pad_image(img, FLAGS.input_size, is_train=is_train)
        img, text_polys = resize_image(img, text_polys, FLAGS.input_size, shift_h, shift_w)
        new_h, new_w, _ = img.shape
        score_map, geo_map, overly_small_text_region_training_mask, text_region_boundary_training_mask = generate_rbox(
            FLAGS, (new_h, new_w), text_polys, text_tags)
        img = (img / 127.5) - 1.
        return img[:, :, ::-1].astype(np.float32), image_file, score_map[::4, ::4, np.newaxis].astype(
            np.float32), geo_map[::4, ::4, :].astype(np.float32), overly_small_text_region_training_mask[::4, ::4,
                                                                  np.newaxis].astype(
            np.float32), text_region_boundary_training_mask[::4, ::4, np.newaxis].astype(np.float32)
    except Exception as e:
        import traceback
        traceback.print_exc()


def load_data(FLAGS, is_train=False):
    image_files = np.array(get_images(FLAGS.validation_data_path))

    loaded_data = []
    for image_file in image_files:
        data = load_data_process(FLAGS, image_file, is_train)
        loaded_data.append(data)
    images = [item[0] for item in loaded_data if item is not None]
    image_fns = [item[1] for item in loaded_data if item is not None]
    score_maps = [item[2] for item in loaded_data if item is not None]
    geo_maps = [item[3] for item in loaded_data if item is not None]
    overly_small_text_region_training_masks = [item[4] for item in loaded_data if item is not None]
    text_region_boundary_training_masks = [item[5] for item in loaded_data if item is not None]
    print('Number of validation images : %d' % len(images))

    return np.array(images), np.array(overly_small_text_region_training_masks), np.array(
        text_region_boundary_training_masks), np.array(score_maps), np.array(geo_maps)


def resize_and_pad_image(image, input_size):
    """
    resize 并且 pad 使图像的大小为 input_size, 且保持 ratio 不变
    Args:
        image:
        input_size:

    Returns:
        padded_image:
        scale:
        pad:
        window:

    """
    h, w = image.shape[:2]
    scale = min(input_size / h, input_size / w)
    new_h = round(h * scale)
    new_w = round(w * scale)
    resized_image = cv2.resize(image, (new_w, new_h))
    pad_top = (input_size - new_h) // 2
    pad_bottom = input_size - new_h - pad_top
    pad_left = (input_size - new_w) // 2
    pad_right = input_size - new_w - pad_left
    pad = [(pad_top, pad_bottom), (pad_left, pad_right), (0, 0)]
    padded_image = np.pad(resized_image, pad, mode='constant', constant_values=0)
    return padded_image, scale, pad, (pad_top, pad_left, new_h + pad_top, new_w + pad_left)


def mold_image(image):
    image = image.astype(np.float32) - np.array([103.9, 116.8, 123.7])
    return image


def create_hdf5(data_dir, input_size, h5_path, dataset_size=878):
    hdf5_dataset = h5py.File(h5_path, 'w')
    hdf5_images = hdf5_dataset.create_dataset(name='images',
                                              shape=(dataset_size, input_size, input_size, 3),
                                              dtype=np.float32)
    hdf5_score_maps = hdf5_dataset.create_dataset(name='score_maps',
                                                  shape=(dataset_size, input_size, input_size, 1),
                                                  dtype=np.float32)
    hdf5_geo_maps = hdf5_dataset.create_dataset(name='geo_maps',
                                                shape=(dataset_size, input_size, input_size, 5),
                                                dtype=np.float32)
    hdf5_text_region_boundary_masks = hdf5_dataset.create_dataset(name='text_region_boundary_masks',
                                                                  shape=(
                                                                      dataset_size, input_size, input_size, 1),
                                                                  dtype=np.float32)
    hdf5_image_fnames = hdf5_dataset.create_dataset(name='image_fnames',
                                                    shape=(dataset_size,),
                                                    dtype=h5py.special_dtype(vlen=str))
    hdf5_polys = hdf5_dataset.create_dataset(name='polys',
                                             shape=(dataset_size,),
                                             dtype=h5py.special_dtype(vlen=np.float32))
    # Create the dataset that will hold the dimensions of the polys for each image so that we can
    # restore the polys from the flattened arrays later.
    hdf5_poly_shapes = hdf5_dataset.create_dataset(name='poly_shapes',
                                                   shape=(dataset_size, 3),
                                                   dtype=np.int32)
    i = 0
    for image_path in glob.glob(osp.join(data_dir, '*.jpg')):
        image = cv2.imread(image_path)
        image, scale, pad, window = resize_and_pad_image(image, input_size)
        src_image = image.copy()
        image_fname = osp.split(image_path)[-1]
        image_fname_noext = osp.splitext(image_fname)[0]
        label_fname = 'gt_' + image_fname_noext + '.txt'
        label_path = osp.join(data_dir, label_fname)
        polys, notext_polys, small_area_polys, small_height_polys, small_width_polys = load_annotation(label_path,
                                                                                                       scale,
                                                                                                       pad,
                                                                                                       window
                                                                                                       )
        if polys.shape[0] != 0:
            image = image.astype(np.float32) - np.array([103.9, 116.8, 123.7])
            hdf5_images[i] = image
            score_map, geo_map, text_region_boundary_mask = generate_rbox(input_size, polys)
            hdf5_score_maps[i] = score_map[:, :, np.newaxis].astype(np.float32)
            hdf5_geo_maps[i] = geo_map[:, :, :].astype(np.float32)
            hdf5_text_region_boundary_masks[i] = text_region_boundary_mask[:, :, np.newaxis].astype(np.float32)
            hdf5_image_fnames[i] = image_fname_noext
            hdf5_polys[i] = polys.reshape(-1)
            hdf5_poly_shapes[i] = polys.shape
            i += 1
    print(i)
    hdf5_dataset.close()


def serialize_polys(data_dir, input_size=512):
    for image_path in glob.glob(osp.join(data_dir, '*.jpg')):
        image = cv2.imread(image_path)
        image, scale, pad, window = resize_and_pad_image(image, input_size)
        src_image = image.copy()
        image_fname = osp.split(image_path)[-1]
        image_fname_noext = osp.splitext(image_fname)[0]
        label_fname = 'gt_' + image_fname_noext + '.txt'
        label_path = osp.join(data_dir, label_fname)
        polys, notext_polys, small_area_polys, small_height_polys, small_width_polys = load_annotation(label_path,
                                                                                                       scale,
                                                                                                       pad,
                                                                                                       window
                                                                                                       )
        np.save(osp.join(data_dir, image_fname_noext + '_polys'), polys)
        np.save(osp.join(data_dir, image_fname_noext + '_notext_polys'), notext_polys)
        np.save(osp.join(data_dir, image_fname_noext + '_small_area_polys'), small_area_polys)
        np.save(osp.join(data_dir, image_fname_noext + '_small_height_polys'), small_height_polys)
        np.save(osp.join(data_dir, image_fname_noext + '_small_width_polys'), small_width_polys)


def validate_hdf5(h5_path):
    hdf5_dataset = h5py.File(h5_path, 'r')
    for i, image in enumerate(hdf5_dataset['images']):
        image = image + np.array([103.9, 116.8, 123.7])
        image = np.round(image).astype(np.uint8)
        score_map = hdf5_dataset['score_maps'][i]
        geo_map = hdf5_dataset['geo_maps'][i]
        text_region_boundary_mask = hdf5_dataset['text_region_boundary_masks'][i]
        image_fname = hdf5_dataset['image_fnames'][i]
        print(image_fname)
        polys = hdf5_dataset['polys'][i]
        poly_shapes = hdf5_dataset['poly_shapes'][i]
        polys = polys.reshape(poly_shapes)
        fig, axs = plt.subplots(4, 2, figsize=(20, 30))
        axs[0, 0].imshow(image[:, :, ::-1])
        axs[0, 0].set_xticks([])
        axs[0, 0].set_yticks([])
        for poly in polys:
            poly_h = min(abs(poly[3, 1] - poly[0, 1]), abs(poly[2, 1] - poly[1, 1]))
            poly_w = min(abs(poly[1, 0] - poly[0, 0]), abs(poly[2, 0] - poly[3, 0]))
            axs[0, 0].add_artist(Patches.Polygon(
                poly, facecolor='none', edgecolor='green', linewidth=2, linestyle='-', fill=True))
            axs[0, 0].text(poly[0, 0], poly[0, 1], '{:.0f}-{:.0f}'.format(poly_h, poly_w), color='purple')
        axs[0, 1].imshow(score_map[:, :, 0])
        axs[0, 1].set_xticks([])
        axs[0, 1].set_yticks([])
        axs[1, 0].imshow(geo_map[:, :, 0])
        axs[1, 0].set_xticks([])
        axs[1, 0].set_yticks([])
        axs[1, 1].imshow(geo_map[:, :, 1])
        axs[1, 1].set_xticks([])
        axs[1, 1].set_yticks([])
        axs[2, 0].imshow(geo_map[:, :, 2])
        axs[2, 0].set_xticks([])
        axs[2, 0].set_yticks([])
        axs[2, 1].imshow(geo_map[:, :, 3])
        axs[2, 1].set_xticks([])
        axs[2, 1].set_yticks([])
        axs[3, 0].imshow(text_region_boundary_mask[:, :, 0])
        axs[3, 0].set_xticks([])
        axs[3, 0].set_yticks([])
        plt.tight_layout()
        plt.show()
        plt.close()
    hdf5_dataset.close()
    # cv2.fillPoly(image, polys.astype(np.int32), (0, 255, 0))
    # cv2.fillPoly(image, notext_polys.astype(np.int32), (0, 0, 255))
    # cv2.fillPoly(image, small_area_polys.astype(np.int32), (255, 255, 0))
    # cv2.fillPoly(image, small_height_polys.astype(np.int32), (255, 0, 0))
    # cv2.fillPoly(image, small_width_polys.astype(np.int32), (0, 255, 255))
    # show_image(image, 'text_polys')
    # show_image(src_image, 'source')
    # cv2.waitKey(0)


def get_valid_dataset_size(data_dir, input_size=512):
    i = 0
    for image_path in glob.glob(osp.join(data_dir, '*.jpg')):
        image = cv2.imread(image_path)
        image, scale, pad, window = resize_and_pad_image(image, input_size)
        image_fname = osp.split(image_path)[-1]
        image_fname_noext = osp.splitext(image_fname)[0]
        label_fname = 'gt_' + image_fname_noext + '.txt'
        label_path = osp.join(data_dir, label_fname)
        polys, notext_polys, small_area_polys, small_height_polys, small_width_polys = load_annotation(label_path,
                                                                                                       scale,
                                                                                                       pad,
                                                                                                       window
                                                                                                       )
        if polys.shape[0] != 0:
            i += 1
    logger.debug('dataset_size of {} is {}'.format(data_dir, i))
    return i


if __name__ == '__main__':
    logger = logging.getLogger('data_process2')
    logger.setLevel(logging.DEBUG)  # default log level
    formatter = logging.Formatter("%(asctime)s %(name)-8s %(levelname)-8s %(lineno)-4d %(message)s")  # output format
    sh = logging.StreamHandler(stream=sys.stdout)  # output to standard output
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    validate_hdf5('val_471_15_test_8_8_64_no###.h5')
    # check_annotations('data/val_data')
    # train_dataset_size = get_valid_dataset_size('data/train_data')
    # val_dataset_size = get_valid_dataset_size('data/val_data')
    # create_hdf5('data/train_data', input_size=512, h5_path='train_1125_13_train_15_train_8_8_64_no###.h5',
    #             dataset_size=train_dataset_size)
    # create_hdf5('data/val_data', input_size=512, h5_path='val_471_15_test_8_8_64_no###.h5',
    #             dataset_size=val_dataset_size)
    # serialize_polys('data/val_data')
    # generator('train.h5', train_dataset_size, 16)
    # corners2 = np.array([[3771, 1850], [3887, 1850], [3887, 2070], [3771, 2070]])
    # reordered_corner2 = reorder_vertexes(corners2)
    # print(polygon_area(reordered_corner2[0]))
    # print(polygon_area(reordered_corner2[1]))
    # parser = argparse.ArgumentParser()
    # FLAGS = parser.parse_args()
    # FLAGS.suppress_warnings_and_error_messages = False
    # FLAGS.min_crop_side_ratio = 0.1
    # images_dir = 'data/train_data'
    # for image_path in glob.glob(osp.join(images_dir, '*.jpg')):
    #     image = cv2.imread(image_path)
    #     h, w = image.shape[:2]
    #     image_fname = osp.split(image_path)[-1]
    #     image_fname_noext = osp.splitext(image_fname)[0]
    #     label_fname = 'gt_' + image_fname_noext + '.txt'
    #     label_path = osp.join(images_dir, label_fname)
    #     text_polys, text_tags = load_annotation(label_path)
    #     text_polys, text_tags = check_and_validate_polys(FLAGS, text_polys, text_tags, (h, w))
    #     crop_image, crop_text_polys, crop_text_tags = crop_area(FLAGS, image, text_polys, text_tags,
    #                                                             crop_background=True)
    #     cv2.drawContours(crop_image, crop_text_polys.astype(np.int32), -1, (0, 255, 0), 2)
    #     show_image(crop_image, 'crop_image')
    #     cv2.drawContours(image, text_polys.astype(np.int32), -1, (0, 255, 0), 2)
    #     show_image(image, 'image')
    #     cv2.waitKey(0)
