import cv2
import config as CFG


def get_reference_width(img_path):
    msg = "OK"

    image = cv2.imread(img_path)
    if image is None:
        msg = "fail to load image"
        return CFG.RESULT_FAIL, msg, 0

    return CFG.RESULT_OK, msg, 0


def test_checker():
    image_path = "testimg/temp_matcher/img1.jpg"
    image = cv2.imread(image_path)

    cv2.imshow("Image", image)
    cv2.waitKey(0)


if __name__ == '__main__':
    test_checker()

