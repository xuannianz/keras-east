import cv2
import numpy as np

src_image = cv2.imread('/home/adam/.keras/datasets/text_detection/ICDAR2013_2015/val_data/img_140_2013.jpg')
image = np.transpose(src_image, [1, 0, 2])
cv2.namedWindow('src_image', cv2.WINDOW_NORMAL)
cv2.imshow('src_image', src_image)
cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.imshow('image', image)
flipped_image = cv2.flip(image, 1)
cv2.namedWindow('flipped_image', cv2.WINDOW_NORMAL)
cv2.imshow('flipped_image', flipped_image)
cv2.imwrite('/home/adam/.keras/datasets/text_detection/ICDAR2013_2015/val_data/img_140_2013.jpg', flipped_image)
cv2.waitKey(0)
