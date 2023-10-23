import cv2
import math
import glob
import os
import config as CFG
import line_det as lineDet

SHOW_LINE = True


# angle为正，逆时针从中心开始旋转;为负时顺时针
def rotate_img(image, angle):
    rows, cols, _ = image.shape
    center = (cols/2, rows/2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    # 旋转后增加的宽和高
    inc_rows = int(cols * math.sin(abs(angle) * math.pi / 180))
    inc_cols = int(rows * math.sin(abs(angle) * math.pi / 180))
    print("inc_rows=%d,inc_cols=%d" % (inc_rows, inc_cols))
    rotated_img = cv2.warpAffine(image, M, (cols+inc_cols, rows+inc_rows), borderValue=(255, 255, 255))

    #cv2.imwrite("rotated.jpg", rotated_img)
    return rotated_img


def pos_correction(img_path, method=1, debug=False):
    '''
    :param img_path: cell 图像
    :param method: 1表示采用霍夫变换，2表示采用LSD
    :return: 返回图像的倾斜角度，以及角度旋转矫正后的图像
    '''
    ori_src = cv2.imread(img_path)
    if ori_src is None:
        msg = "fail to load image"
        return CFG.RESULT_FAIL, msg, 0, None

    # hough
    if method == 1:
        rs = 6
        lines = lineDet.det_line_by_hough(ori_src, rs, debug)
    # lsd
    else:
        rs = 4
        lines = lineDet.det_line_by_lsd(ori_src, rs, debug)

    if lines is None:
        msg = "fail to find lines in image"
        return CFG.RESULT_FAIL, msg, 0, None

    final_angle, finale_lines = lineDet.get_angle_by_lines(lines, debug)
    if final_angle == CFG.RESULT_FAIL:
        msg = "no suitable angles"
        return CFG.RESULT_FAIL, msg, 0, None
    # print(finale_lines)

    # 保存原始大小的旋转后的图
    if final_angle != 0:
        rotated_img = rotate_img(ori_src, final_angle)
    else:
        rotated_img = ori_src
    if debug:
        # 最终图像上的线条显示
        rimg = cv2.resize(ori_src, (int(ori_src.shape[1] / rs), int(ori_src.shape[0] / rs)))
        for line in finale_lines:
            cv2.line(rimg, (line[0], line[1]), (line[2], line[3]), (0, 0, 255), 1)
        cv2.imshow("image_final_lines", rimg)

        # 显示旋转后的图像
        resize_scale = 6
        rr = cv2.resize(rotated_img, (int(ori_src.shape[1] / resize_scale), int(ori_src.shape[0] / resize_scale)))
        cv2.imshow("rotated", rr)
        cv2.waitKey(0)

    return CFG.RESULT_OK, "OK", final_angle, rotated_img


def pos_correction_withsave(img_path, method=1, debug=False):
    '''
    :param img_path: cell 图像
    :return: 返回图像的倾斜角度，以及角度旋转矫正后的图像
    '''
    rslt, msg, final_angle, rotated_img = pos_correction(img_path, method, debug)
    if rslt != CFG.RESULT_OK:
        print(msg)
        rotated_img_path = ''
    else:
        rotated_img_path = img_path.split('.')[0] + "_rotate." + img_path.split('.')[1]

        # 最终保存在共享目录
        res_path, res_file = os.path.split(rotated_img_path)
        rotated_img_path = CFG.SHARE_DIR + "temp\\" + res_file
        cv2.imwrite(rotated_img_path, rotated_img)

    return rslt, msg, final_angle, rotated_img_path


def test_posCorrection():
    image_path = "testimg/pos_corr/03162_rotate.jpg"
    pos_correction_withsave(image_path, method=2, debug=True)


def test_posCorrection_dir():
    dir = "testimg/temp_matcher/dd"
    images = glob.glob(os.path.join(dir, '*'))
    for img in images:
        print("processing file:{}".format(img))
        pos_correction_withsave(img, debug=False)


def test_linedet():
    image_path = "testimg/pos_corr/c3.jpg"

    img = cv2.imread(image_path)
    rs = 4
    lines = lineDet.det_line_by_lsd(img, rs)

    rimg = cv2.resize(img, (int(img.shape[1] / rs), int(img.shape[0] / rs)))
    for l in lines:
        x1, y1, x2, y2 = l
        cv2.line(rimg, (x1, y1), (x2, y2), (0, 0, 255), 1)

    # 显示检测到的图像
    cv2.imshow("lsd", rimg)
    cv2.waitKey(0)


if __name__ == '__main__':
    test_posCorrection()
    # test_linedet()

