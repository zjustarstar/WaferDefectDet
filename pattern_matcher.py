import numpy as np
import cv2
import config as CFG


def pattern_matcher(img_path, temp_path):
    match_thresh = 0.2
    msg = "OK"

    template = cv2.imread(temp_path)
    if template is None:
        msg = "fail to load template image"
        return CFG.RESULT_FAIL, msg, []
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    template = cv2.Canny(template_gray, 100, 50)
    (tH, tW) = template.shape[:2]

    # load the image, convert it to grayscale, and initialize the
    # bookkeeping variable to keep track of the matched region
    image = cv2.imread(img_path)
    if image is None:
        msg = "fail to load image"
        return CFG.RESULT_FAIL, msg, []
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # detect edges in the resized, grayscale image and apply template
    # matching to find the template in the image
    edged_img = cv2.Canny(gray, 100, 50)
    result = cv2.matchTemplate(edged_img, template, cv2.TM_CCOEFF_NORMED)

    # 找到最佳和最差匹配点
    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(result)
    (startX, startY) = (int(maxLoc[0]), int(maxLoc[1]))
    res_rects = [[startX, startY, startX+tW, startY+tH]]
    #print(minVal, maxVal, minLoc, maxLoc)

    # 查找所有匹配值中大于阈值的点;
    loc = np.where(result >= 0.2)
    for i in zip(*loc[::-1]):
        cv2.rectangle(image, i, (i[0] + tW, i[1] + tH), 255, 1)

    return CFG.RESULT_OK, msg, res_rects


def test_matcher():
    image_path = "testimg/temp_matcher/img1.jpg"
    temp_path = "testimg/temp_matcher/temp1.jpg"
    image = cv2.imread(image_path)
    _, _, res = pattern_matcher(image_path, temp_path)

    for r in res:
        cv2.rectangle(image, (r[0], r[1]), (r[2], r[3]), 255, 1)

    # draw a bounding box around the detected result and display the image
    # cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)
    cv2.imshow("Image", image)
    cv2.waitKey(0)


if __name__ == '__main__':
    test_matcher()

