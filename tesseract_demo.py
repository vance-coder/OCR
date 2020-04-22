import pytesseract
from PIL import Image

"""
tesseract delete alpha channel 
the best dpi for tesseract is at lest 300dpi
image.convert("L")  Done 
"""


def demo():
    image = Image.open('/Users/liuliangjun/Downloads/pic_3.png')
    w, h = image.size
    factor = 30 / min(image.size)
    if factor > 1:
        image = image.resize((int(w * factor), int(h * factor)))

    # default config --oem 3 --psm 3
    text = pytesseract.image_to_string(image, config='-l eng+chi_sim --oem 3 --psm 3')
    print('Result:')
    print(text)


# text detection of opencv （The effect is not good）
def text_detection():
    import cv2
    import matplotlib.pyplot as plt

    img = cv2.imread('/Users/liuliangjun/Downloads/test4.png')
    mser = cv2.MSER_create(_min_area=300)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    regions, boxes = mser.detectRegions(gray)

    for box in boxes:
        x, y, w, h = box
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

    plt.imshow(img, 'brg')
    plt.show()


if __name__ == '__main__':
    # text_detection()
    demo()
