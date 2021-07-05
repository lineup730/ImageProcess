import cv2
from ImgFunction import *

img = cv2.imread('img/fly.jpg')
resize_img = resize_img(img, 30)
rotate_img = rotate_img(resize_img, 90);
crop_img = crop_img(img, 500, 900, 100, 900)

cv2.imshow('fly', resize_img)
cv2.imshow('fly2', rotate_img)
cv2.imshow('fly3', crop_img)


cv2.waitKey(0)
cv2.destroyAllWindows()
