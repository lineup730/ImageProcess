import cv2
import numpy as np

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

'''
img:    input image
scale:  scale factor 0.01~100
'''
def modify_intensity(img, scale):

    if scale < 0.01 or scale > 100:
        return img

    maxIntensity = 255.0 # depends on dtype of image data

    # Parameters for manipulating image data
    phi = 1
    theta = 1

    # Modify intensity 
    process_img = (maxIntensity/phi)*(img/(maxIntensity/theta))**scale
    process_img = np.array(process_img, dtype=np.uint8)

    return process_img


'''
scale:  -scale percent ~ +scale persent
'''
def modify_lightness(img, scale):
    # 圖像歸一化，且轉換為浮點型
    fImg = img.astype(np.float32)
    fImg = fImg / 255.0

    # 顏色空間轉換 BGR -> HLS
    hlsImg = cv2.cvtColor(fImg, cv2.COLOR_BGR2HLS)
    hlsCopy = np.copy(hlsImg)

    # 亮度調整
    hlsCopy[:, :, 1] = (1 + scale / 100.0) * hlsCopy[:, :, 1]
    hlsCopy[:, :, 1][hlsCopy[:, :, 1] > 1] = 1  # 應該要介於 0~1，計算出來超過1 = 1

    # 顏色空間反轉換 HLS -> BGR 
    result_img = cv2.cvtColor(hlsCopy, cv2.COLOR_HLS2BGR)
    result_img = ((result_img * 255).astype(np.uint8))

    return result_img


'''
scale:  -scale percent ~ +scale persent
'''
def modify_saturation(img, scale):

    # 圖像歸一化，且轉換為浮點型
    fImg = img.astype(np.float32)
    fImg = fImg / 255.0

    # 顏色空間轉換 BGR -> HLS
    hlsImg = cv2.cvtColor(fImg, cv2.COLOR_BGR2HLS)
    hlsCopy = np.copy(hlsImg)

    # 飽和度調整
    hlsCopy[:, :, 2] = (1 + scale / 100.0) * hlsCopy[:, :, 2]
    hlsCopy[:, :, 2][hlsCopy[:, :, 2] > 1] = 1  # 應該要介於 0~1，計算出來超過1 = 1

    # 顏色空間反轉換 HLS -> BGR 
    result_img = cv2.cvtColor(hlsCopy, cv2.COLOR_HLS2BGR)
    result_img = ((result_img * 255).astype(np.uint8))

    return result_img

'''
B, G ,R:    -255 ~ +255
'''
def modify_color_temperature(img, B, G ,R):

    # ---------------- 冷色調 ---------------- #  

#     height = img.shape[0]
#     width = img.shape[1]
#     dst = np.zeros(img.shape, img.dtype)

    # 1.計算三個通道的平均值，並依照平均值調整色調
    imgB = img[:, :, 0] 
    imgG = img[:, :, 1]
    imgR = img[:, :, 2] 

    # 調整色調請調整這邊~~ 
    # 白平衡 -> 三個值變化相同
    # 冷色調(增加b分量) -> 除了b之外都增加
    # 暖色調(增加r分量) -> 除了r之外都增加
    bAve = cv2.mean(imgB)[0] + B
    gAve = cv2.mean(imgG)[0] + G
    rAve = cv2.mean(imgR)[0] + R
    aveGray = (int)(bAve + gAve + rAve) / 3

    # 2. 計算各通道增益係數，並使用此係數計算結果
    bCoef = aveGray / bAve
    gCoef = aveGray / gAve
    rCoef = aveGray / rAve
    imgB = np.floor((imgB * bCoef))  # 向下取整
    imgG = np.floor((imgG * gCoef))
    imgR = np.floor((imgR * rCoef))

    # 3. 變換後處理
#     for i in range(0, height):
#         for j in range(0, width):
#             imgb = imgB[i, j]
#             imgg = imgG[i, j]
#             imgr = imgR[i, j]
#             if imgb > 255:
#                 imgb = 255
#             if imgg > 255:
#                 imgg = 255
#             if imgr > 255:
#                 imgr = 255
#             dst[i, j] = (imgb, imgg, imgr)

    # 將原文第3部分的演算法做修改版，加快速度
    imgb = imgB
    imgb[imgb > 255] = 255
    imgb[imgb < 0] = 0

    imgg = imgG
    imgg[imgg > 255] = 255
    imgg[imgg < 0] = 0

    imgr = imgR
    imgr[imgr > 255] = 255
    imgr[imgr < 0] = 0

    cold_rgb = np.dstack((imgb, imgg, imgr)).astype(np.uint8) 
    return cold_rgb