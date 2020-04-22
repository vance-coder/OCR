# coding:utf-8
import time
from glob import glob

import numpy as np
from PIL import Image

import model


# TODO
# 运行前需要的配置
# 1. 修改 angle/predict.py 32行 模型地址
# 2. 修改 /ctpn/ctpn/model.py 36 行 checkpoint 地址

def demo():
    im = Image.open("/Users/liuliangjun/Downloads/test2.jpg")
    img = np.array(im.convert('RGB'))
    t = time.time()
    '''
    result,img,angel分别对应-识别结果，图像的数组，文字旋转角度
    '''
    result, img, angle = model.model(
        img, model='crnn', adjust=True, detectAngle=True)
    print(result, img, angle)
    print("It takes time:{}s".format(time.time() - t))
    print("---------------------------------------")
    for key in result:
        print(result[key][1])

    Image.fromarray(img).show()


if __name__ == '__main__':
    demo()
