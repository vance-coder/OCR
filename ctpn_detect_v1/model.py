# coding:utf-8
##添加文本方向 检测模型，自动检测文字方向，0、90、180、270
import sys
from math import *
from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2
import pytesseract
import numpy as np
from PIL import Image

from angle.predict import predict as angle_detect  ##文字方向检测
from ctpn.text_detect import text_detect


def img_to_string(image):
    return pytesseract.image_to_string(image, config='-l eng+chi_sim --oem 3 --psm 3')


def crnnRec(im, text_recs, ocrMode='keras', adjust=False):
    """
    crnn模型，ocr识别
    @@model,
    @@converter,
    @@im:Array
    @@text_recs:text box

    """
    images = []
    results = {}
    xDim, yDim = im.shape[1], im.shape[0]

    for index, rec in enumerate(text_recs):
        results[index] = [
            rec,
        ]
        xlength = int((rec[6] - rec[0]) * 0.1)
        ylength = int((rec[7] - rec[1]) * 0.2)
        if adjust:
            pt1 = (max(1, rec[0] - xlength), max(1, rec[1] - ylength))
            pt2 = (rec[2], rec[3])
            pt3 = (min(rec[6] + xlength, xDim - 2),
                   min(yDim - 2, rec[7] + ylength))
            pt4 = (rec[4], rec[5])
        else:
            pt1 = (max(1, rec[0]), max(1, rec[1]))
            pt2 = (rec[2], rec[3])
            pt3 = (min(rec[6], xDim - 2), min(yDim - 2, rec[7]))
            pt4 = (rec[4], rec[5])

        degree = degrees(atan2(pt2[1] - pt1[1], pt2[0] - pt1[0]))  # 图像倾斜角度

        partImg = dumpRotateImage(im, degree, pt1, pt2, pt3, pt4)
        # 根据ctpn进行识别出的文字区域，进行不同文字区域的crnn识别
        image = Image.fromarray(partImg).convert('L')

        # 图片的长宽如果小于30px，则按比例放大
        w, h = image.size
        factor = 30 / min(image.size)
        if factor > 1:
            print('turn size...')
            image = image.resize((int(w * factor), int(h * factor)))

        images.append(image)
        # image.save(f'./temp/{index}.png')

        # 进行识别出的文字识别
        # sim_pred = pytesseract.image_to_string(image, config='-l eng+chi_sim --oem 3 --psm 3')
        # results[index].append(sim_pred)

    with ProcessPoolExecutor() as executor:
        res = [executor.submit(img_to_string, img) for img in images]
    for idx, r in enumerate(res):
        results[idx].append(r.result())

    return results


def dumpRotateImage(img, degree, pt1, pt2, pt3, pt4):
    height, width = img.shape[:2]
    heightNew = int(width * fabs(sin(radians(degree))) +
                    height * fabs(cos(radians(degree))))
    widthNew = int(height * fabs(sin(radians(degree))) +
                   width * fabs(cos(radians(degree))))
    matRotation = cv2.getRotationMatrix2D((width / 2, height / 2), degree, 1)
    matRotation[0, 2] += (widthNew - width) / 2
    matRotation[1, 2] += (heightNew - height) / 2
    imgRotation = cv2.warpAffine(
        img, matRotation, (widthNew, heightNew), borderValue=(255, 255, 255))
    pt1 = list(pt1)
    pt3 = list(pt3)

    [[pt1[0]], [pt1[1]]] = np.dot(matRotation,
                                  np.array([[pt1[0]], [pt1[1]], [1]]))
    [[pt3[0]], [pt3[1]]] = np.dot(matRotation,
                                  np.array([[pt3[0]], [pt3[1]], [1]]))
    ydim, xdim = imgRotation.shape[:2]
    imgOut = imgRotation[max(1, int(pt1[1])):min(ydim - 1, int(pt3[1])),
             max(1, int(pt1[0])):min(xdim - 1, int(pt3[0]))]
    # height,width=imgOut.shape[:2]
    return imgOut


def model(img, model='keras', adjust=False, detectAngle=False):
    """
    @@param:img,
    @@param:model,选择的ocr模型，支持keras\pytorch版本
    @@param:adjust 调整文字识别结果
    @@param:detectAngle,是否检测文字朝向
    
    """
    angle = 0
    if detectAngle:
        # 进行文字旋转方向检测，分为[0, 90, 180, 270]四种情况
        angle = angle_detect(img=np.copy(img))  ##文字朝向检测
        print('The angel of this character is:', angle)
        im = Image.fromarray(img)
        print('Rotate the array of this img!')
        if angle == 90:
            im = im.transpose(Image.ROTATE_90)
        elif angle == 180:
            im = im.transpose(Image.ROTATE_180)
        elif angle == 270:
            im = im.transpose(Image.ROTATE_270)
        img = np.array(im)
    # 进行图像中的文字区域的识别
    text_recs, tmp, img = text_detect(img)
    # 识别区域排列
    text_recs = sort_box(text_recs)

    result = crnnRec(img, text_recs, model, adjust=adjust)
    return result, tmp, angle


def sort_box(box):
    """
    对box排序,及页面进行排版
    text_recs[index, 0] = x1
        text_recs[index, 1] = y1
        text_recs[index, 2] = x2
        text_recs[index, 3] = y2
        text_recs[index, 4] = x3
        text_recs[index, 5] = y3
        text_recs[index, 6] = x4
        text_recs[index, 7] = y4
    """

    box = sorted(box, key=lambda x: sum([x[1], x[3], x[5], x[7]]))
    return box
