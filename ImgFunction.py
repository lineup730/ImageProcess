import cv2

'''
img:    input image
coordinate: (y,x)
            (0,0)---------(0,x)
              l             l
              l             l             
              l             l
            (y,0)---------(y,x)
'''
def crop_img(img, x_left, x_right, y_height, y_bottom):

    # crop image
    crop_img = img[y_height:y_bottom, x_left:x_right]  # notice: first y, then x

    return crop_img


'''
img:    input image
angle:  rotate angle, + for anti-clockwise, - for clockwise
'''
def rotate_img(img, angle):
    (h, w, d) = img.shape # 讀取圖片大小
    center = (w // 2, h // 2) # 找到圖片中心

    # 第一個參數旋轉中心，第二個參數旋轉角度(-順時針/+逆時針)，第三個參數縮放比例
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    # 第三個參數變化後的圖片大小
    rotate_img = cv2.warpAffine(img, M, (w, h))

    return rotate_img

'''
img:    input image
scale:  scale factor 1~100
'''
def resize_img(img, scale):

    if scale <= 0 or scale > 100:
        return image

    width = int(img.shape[1] * scale / 100) # 縮放後圖片寬度
    height = int(img.shape[0] * scale / 100) # 縮放後圖片高度
    dim = (width, height) # 圖片形狀 
    resize_img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)  

    return resize_img