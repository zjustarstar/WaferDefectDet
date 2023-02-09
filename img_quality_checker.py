import cv2
import glob
import numpy as np
import os
import config as CFG


# 该值越大越好
def get_img_quality(img_path):
    img = cv2.imread(img_path)
    if img is None:
        msg = "fail to load image"
        return CFG.RESULT_FAIL

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgH, imgW = img_gray.shape[:2]

    margin = 300  # 不考虑边上的
    roiImg = img_gray[margin:imgH-margin, margin:imgW-margin]
    img_array = np.array(roiImg)
    img_array_line = img_array.reshape([1,img_array.shape[0]*img_array.shape[1]])
    img_std = np.std(img_array_line)

    return img_std


def compare_img_quality(img1_path, img2_path):
    std1 = get_img_quality(img1_path)
    std2 = get_img_quality(img2_path)
    if std1 == CFG.RESULT_FAIL or std2 == CFG.RESULT_FAIL:
        return CFG.RESULT_FAIL
    else:
        return std1 > std2


def test_posCorrection():
    # image_path = "testimg/quality/x20"
    # images = glob.glob(os.path.join(image_path, '*'))
    # for img in images:
    #     std = get_img_quality(img)
    #     print("file={}, std={}".format(img, std))

    res = compare_img_quality("testimg/quality/x20/2.jpg", "testimg/quality/x20/1.jpg")
    print(res, int(res))


if __name__ == '__main__':
    test_posCorrection()

