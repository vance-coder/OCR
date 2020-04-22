import numpy as np
# import tensorflow as tf
from .ctpn.detectors import TextDetector
from .ctpn.model import ctpn
from .ctpn.other import draw_boxes

'''
进行文本区域识别-网络结构为cnn+rnn
'''


def text_detect(img):
    # ctpn网络测到
    scores, boxes, img = ctpn(img)
    text_detector = TextDetector()
    boxes = text_detector.detect(boxes, scores[:, np.newaxis], img.shape[:2])
    text_recs, tmp = draw_boxes(
        img, boxes, caption='im_name', wait=True, is_display=True)
    return text_recs, tmp, img
