import cv2
import numpy as np
import math

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

    # 0 <= sigma <= 1
    scale = np.clip(scale, 1 , 100)

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

def gaussian_noise(img, mean=0, sigma=0.1):

    # 0 <= sigma <= 1
    sigma = np.clip(sigma, 0 , 1)

    # int -> float (標準化)
    img = img / 255
    # 隨機生成高斯 noise (float + float)
    noise = np.random.normal(mean, sigma, img.shape)
    # noise + 原圖
    gaussian_out = img + noise
    # 所有值必須介於 0~1 之間，超過1 = 1，小於0 = 0
    gaussian_out = np.clip(gaussian_out, 0, 1)

    # 原圖: float -> int (0~1 -> 0~255)
    gaussian_out = np.uint8(gaussian_out*255)
    # noise: float -> int (0~1 -> 0~255)
    noise = np.uint8(noise*255)

    return gaussian_out


'''
contrast    : -255~255
'''
def modify_contrast(img, contrast):
    
    #增加對比度: 白的更白，黑的更黑
    #減少對比度: 白黑都接近灰
    
    c = contrast / 255.0 

    #tan > 45度: y>x，為分數 (0~1)，表示更接近0
    #tan < 45度: x>y，為假分數 (>1)，表示更遠離0
    
    #將 c 代入，得到約 tan((1~89)/180*pi)，
    k = math.tan((45 + 44 * c) / 180 * math.pi)

    #我們從255的一半 127.5開始看，
    #後面127.5是正的
    #前面127.5是負的，因為乘上k，
    #k可以決定要更大的負(整個式子結果更負)或更小的負(整個式子結果更正)
    img = (img - 127.5) * k + 127.5

    # 所有值必須介於 0~255 之間，超過255 = 255，小於 0 = 0
    img = np.clip(img, 0, 255).astype(np.uint8)

    return img



def reduce_highlights(img, alpha = 0.2, beta = 0.4):

    alpha = float(np.clip(alpha, 0 , 2))
    beta = float(np.clip(beta, 0 , 2))

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 先轉成灰階處理
    ret, thresh = cv2.threshold(img_gray, 200, 255, 0)  # 利用 threshold 過濾出高光的部分，目前設定高於 200 即為高光
    contours, hierarchy  = cv2.findContours(thresh.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_zero = np.zeros(img.shape, dtype=np.uint8) 

#     print(len(contours))

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour) 
        img_zero[y:y+h, x:x+w] = 255 
        mask = img_zero 

    # alpha，beta 共同決定高光消除後的模糊程度
    # alpha: 亮度的缩放因子，默認是 0.2， 範圍[0, 2], 值越大，亮度越低
    # beta:  亮度缩放後加上的参数，默認是 0.4， 範圍[0, 2]，值越大，亮度越低
    result = cv2.illuminationChange(img, mask, alpha, beta) 

    return result