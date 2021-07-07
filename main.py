import cv2
import numpy as np
from ImgFunction import *

img = cv2.imread('img/fly.jpg')
resize_img = resize_img(img, 10)
#rotate_img = rotate_img(resize_img, 90);
#crop_img = crop_img(img, 500, 900, 100, 900)
#mi_img = modify_intensity(resize_img, 2)
#light_img = modify_lightness(resize_img, 100)
#saturation_img = modify_saturation(resize_img, 100)
#B_color_img = modify_color_temperature(resize_img, 0,90,90)
#G_color_img = modify_color_temperature(resize_img, 20,0,20)
#R_color_img = modify_color_temperature(resize_img, 20,20,0)
#noise_img = gaussian_noise(resize_img,0 , 0.1)
#contrast_img = modify_contrast(resize_img, 100)
#rh_img = reduce_highlights(resize_img, 0.2, 0.2)
#rh2_img = reduce_highlights(resize_img, 2, 2)

cv2.imshow('fly', resize_img)


cv2.waitKey(0)
cv2.destroyAllWindows()
