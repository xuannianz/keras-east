import cv2
import numpy as np
import os.path as osp
from data_processor import load_annotation, reorder_vertexes, show_image, fit_line, point_dist_to_line, line_cross_point
from data_processor import rectangle_from_parallelogram, sort_rectangle
from shapely.geometry import Polygon


def get_parallelograms_of_poly(poly):
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
    return fitted_parallelograms


def get_enclosing_rectangle(parallelograms):
    # 8 个平行四边形的面积
    areas = [Polygon(t).area for t in parallelograms]
    # 面积最小的平行四边形的顶点坐标, -1 表示把最后一个元素 new_p0 去掉
    parallelogram = np.array(parallelograms[np.argmin(areas)][:-1], dtype=np.float32)
    # sort this polygon
    parallelogram_coord_sum = np.sum(parallelogram, axis=1)
    min_coord_idx = np.argmin(parallelogram_coord_sum)
    parallelogram = parallelogram[
        [min_coord_idx, (min_coord_idx + 1) % 4, (min_coord_idx + 2) % 4, (min_coord_idx + 3) % 4]]
    # 获得面积最小的外接矩形
    rectange = rectangle_from_parallelogram(parallelogram)
    rectange, rotate_angle = sort_rectangle(rectange)
    return rectange


image_path = 'data/train_data/img_159.jpg'
image = cv2.imread(image_path)
h, w = image.shape[:2]
image_fname = osp.split(image_path)[-1]
image_fname_noext = osp.splitext(image_fname)[0]
label_fname = 'gt_' + image_fname_noext + '.txt'
label_path = osp.join('data/train_data', label_fname)
text_polys, _ = load_annotation(label_path)
text_poly = text_polys[36]
counter_clockwise, clockwise = reorder_vertexes(text_poly)
cv2.drawContours(image, [counter_clockwise.astype(np.int32)], -1, (0, 0, 0), 1)
parallelograms = get_parallelograms_of_poly(counter_clockwise)
rectange = get_enclosing_rectangle(parallelograms)
for parallelogram in parallelograms:
    color = np.random.randint(0, 255, (3,)).tolist()
    cv2.drawContours(image, [np.array(parallelogram).astype(np.int32)], -1, color, 1)

cv2.drawContours(image, [rectange.astype(np.int32)], -1, (0, 255, 0), 2)
show_image(image, 'small')
cv2.waitKey(0)
