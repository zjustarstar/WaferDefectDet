import cv2
import config as CFG


def cell_abnormal_det(img_path):
    msg = "OK"

    image = cv2.imread(img_path)
    # 最后两个返回的参数是：是否是缺陷，以及缺陷类型
    if image is None:
        msg = "fail to load image"
        return CFG.RESULT_FAIL, msg, CFG.AD_NG, 0

    return CFG.RESULT_OK, msg, CFG.AD_GOOD, 0


def test_detection():
    image_path = "testimg/temp_matcher/img1.jpg"
    image = cv2.imread(image_path)

    cv2.imshow("Image", image)
    cv2.waitKey(0)


if __name__ == '__main__':
    test_detection()

