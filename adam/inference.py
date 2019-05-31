from keras import models
from adam.data_processor import resize_and_pad_image, mold_image, restore_rectangle_rbox, show_image
import cv2
import numpy as np
from adam.model import EAST_model
import matplotlib.pyplot as plt
import lanms
import glob
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
east = EAST_model()
model = east.model
model.load_weights('icdar_2015_2013_37_0.0049_0.0092.h5', by_name=True)
for image_path in glob.glob('/home/adam/.keras/datasets/icdar2015/incidental_scene_text/task1_test_images/*.jpg'):
    # for image_path in glob.glob('/home/adam/.keras/datasets/icdar2013/focused_scene_text/task12_images/*.jpg'):
    # for image_path in glob.glob('data/val_data/*.jpg'):
    # for image_path in glob.glob('/home/adam/.keras/datasets/text/VOCdevkit/VOC2007/JPEGImages/*.jpg'):
    image = cv2.imread(image_path)
    image, *_ = resize_and_pad_image(image, 512)
    input = mold_image(image)
    input = np.expand_dims(input, axis=0)
    score_map, geo_map = model.predict([input, np.zeros((1, 128, 128, 1)), np.zeros((1, 128, 128, 1))])
    # fig, axs = plt.subplots(3, 2, figsize=(20, 30))
    # image = np.round(image).astype(np.uint8)
    # axs[0, 0].imshow(image[:, :, ::-1])
    # axs[0, 0].set_xticks([])
    # axs[0, 0].set_yticks([])
    # axs[0, 1].imshow(score_map[0, :, :, 0])
    # axs[0, 1].set_xticks([])
    # axs[0, 1].set_yticks([])
    # axs[1, 0].imshow(geo_map[0, :, :, 0])
    # axs[1, 0].set_xticks([])
    # axs[1, 0].set_yticks([])
    # axs[1, 1].imshow(geo_map[0, :, :, 1])
    # axs[1, 1].set_xticks([])
    # axs[1, 1].set_yticks([])
    # axs[2, 0].imshow(geo_map[0, :, :, 2])
    # axs[2, 0].set_xticks([])
    # axs[2, 0].set_yticks([])
    # axs[2, 1].imshow(geo_map[0, :, :, 3])
    # axs[2, 1].set_xticks([])
    # axs[2, 1].set_yticks([])
    # plt.tight_layout()
    # plt.show()
    # plt.close()
    # filter the score map
    score_map = score_map[0, :, :, 0]
    geo_map = geo_map[0]
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
    # nms part
    boxes = lanms.merge_quadrangle_n9(boxes.astype('float32'), 0.2)
    # here we filter some low score boxes by the average score map, this is different from the orginal paper
    if boxes.shape[0] != 0:
        for i, box in enumerate(boxes):
            mask = np.zeros_like(score_map, dtype=np.uint8)
            cv2.fillPoly(mask, box[:8].reshape((-1, 4, 2)).astype(np.int32) // 4, 1)
            # score_map 上 mask 对应的部分的平均值作为这个 box 的 score
            boxes[i, 8] = cv2.mean(score_map, mask)[0]
        boxes = boxes[boxes[:, 8] > 0.1]
        boxes = np.reshape(boxes[:, :8], (-1, 4, 2))
        cv2.drawContours(image, boxes.round().astype(np.int32), -1, (0, 255, 0))
    show_image(image, 'restored')
    cv2.waitKey(0)
