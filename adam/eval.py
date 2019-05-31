import numpy as np
import cv2
import lanms
from adam.data_processor import resize_and_pad_image, mold_image, restore_rectangle_rbox, show_image
from model import EAST_model
import os.path as osp
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
east = EAST_model()
model = east.model
model.load_weights('icdar_2015_2013_55_0.0038_0.0090.h5', by_name=True)


def sort_poly(p):
    min_axis = np.argmin(np.sum(p, axis=1))
    p = p[[min_axis, (min_axis + 1) % 4, (min_axis + 2) % 4, (min_axis + 3) % 4]]
    if abs(p[0, 0] - p[1, 0]) > abs(p[0, 1] - p[1, 1]):
        return p
    else:
        return p[[0, 3, 2, 1]]


def inference(image_path):
    image = cv2.imread(image_path)
    image_fname = osp.split(image_path)[-1]
    image_fname_noext = osp.splitext(image_fname)[0]
    label_fname = 'res_' + image_fname_noext + '.txt'
    src_image = image.copy()
    image, scale, pad, window = resize_and_pad_image(image, 512)
    input = mold_image(image)
    input = np.expand_dims(input, axis=0)
    score_map, geo_map = model.predict([input, np.zeros((1, 128, 128, 1)), np.zeros((1, 128, 128, 1))])
    # filter the score map
    score_map = score_map[0, :, :, 0]
    geo_map = geo_map[0]
    # argwhere 返回一个二维数组, 每一个元素表示满足条件的值的下标
    yx_text = np.argwhere(score_map > 0.8)
    # sort the text boxes via the y axis
    yx_text = yx_text[np.argsort(yx_text[:, 0])]
    # restore
    # *4 是为了恢复到原来图像的大小, ::-1 用于交换 yx 的位置, 把 x 放在前面, y 放在后面
    text_box_restored = restore_rectangle_rbox(yx_text[:, ::-1] * 4, geo_map[yx_text[:, 0], yx_text[:, 1], :])
    # cv2.drawContours(image, text_box_restored.round().astype(np.int32), -1, (0, 255, 0))
    # show_image(image, 'restored')
    # cv2.waitKey(0)
    print('{} text boxes before nms'.format(text_box_restored.shape[0]))
    boxes = np.zeros((text_box_restored.shape[0], 9), dtype=np.float32)
    boxes[:, :8] = text_box_restored.reshape((-1, 8))
    boxes[:, 8] = score_map[yx_text[:, 0], yx_text[:, 1]]
    # nms
    boxes = lanms.merge_quadrangle_n9(boxes.astype('float32'), 0.2)
    # here we filter some low score boxes by the average score map, this is different from the original paper
    if boxes.shape[0] != 0:
        for i, box in enumerate(boxes):
            mask = np.zeros_like(score_map, dtype=np.uint8)
            cv2.fillPoly(mask, box[:8].reshape((-1, 4, 2)).astype(np.int32) // 4, 1)
            # score_map 上 mask 对应的部分的平均值作为这个 box 的 score
            boxes[i, 8] = cv2.mean(score_map, mask)[0]
        boxes = boxes[boxes[:, 8] > 0.1]
        boxes = np.reshape(boxes[:, :8], (-1, 4, 2))
        boxes[:, :, 0] = boxes[:, :, 0] - pad[1][0]
        boxes[:, :, 1] = boxes[:, :, 1] - pad[0][0]
        boxes /= scale
        boxes = boxes.round().astype(np.int32)
        label_path = osp.join('data/pred', label_fname)
        with open(label_path, 'w') as f:
            for box in boxes:
                box = sort_poly(box.astype(np.int32))
                if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(box[3] - box[0]) < 5:
                    continue
                f.write('{},{},{},{},{},{},{},{}\n'.format(box[0, 0], box[0, 1], box[1, 0], box[1, 1],
                                                           box[2, 0], box[2, 1], box[3, 0], box[3, 1]))
        # cv2.drawContours(src_image, boxes, -1, (0, 255, 0), 1)
        # show_image(src_image, 'src_image')
        # cv2.waitKey(0)


if __name__ == '__main__':
    import glob

    for image_path in glob.glob('data/val_data/*.jpg'):
        inference(image_path)
