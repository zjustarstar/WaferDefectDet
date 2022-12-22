import numpy as np
import cv2
import math
import config as CFG
import numpy as np


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


def pos_correction(img_path):
    MAX_ROATATE_ANGLE = 30  # 倾斜程度不能超过这个角度
    src = cv2.imread(img_path)
    if src is None:
        msg = "fail to load image"
        return CFG.RESULT_FAIL, msg, []
    src_gray = 255 - cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    src_edge = cv2.Canny(src_gray, 120, 50)
    cv2.imshow("init", src_edge)
    # 填充一些小细缝
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))  # 定义结构元素的形状和大小
    # edge_d = cv2.dilate(src_edge, kernel)  # 膨胀
    # src_edge = cv2.erode(edge_d, kernel) # 腐蚀
    # cv2.imshow("edges", src_edge)
    #
    # # 删除小的连通区域
    # contours, _ = cv2.findContours(src_edge, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    # # 对于每一个连通区域
    # for i in range(len(contours)):
    #     for num, contour in enumerate(contours):
    #         if cv2.contourArea(contour) < 50:
    #             x, y, w, h = cv2.boundingRect(contour)
    #             src_edge[y:y + h, x:x + w] = 0
    # cv2.imshow("edges2", src_edge)

    nscale = 10  # 方便计算;
    # 一条直线所需最少的的曲线交点。超过设定阈值才被检测出线段，值越大，基本上意味着检出的线段越长，检出的线段个数越少
    thre = 100   # 最短的线
    # minLinLength: 能组成一条直线的最少点的数量. 点数量不足的直线将被抛弃.
    # maxLineGap: 能被认为在一条直线上的两点的最大距离
    lines = cv2.HoughLinesP(src_edge, 1, np.pi / 180, thre,
                            minLineLength=100, maxLineGap=30)

    all_angles = []
    for i in range(len(lines)):
        for x1, y1, x2, y2 in lines[i]:
            cv2.line(src, (x1, y1), (x2, y2), (0, 0, 255), 2)
            # 水平线
            if abs(x1-x2) > abs (y1-y2):
                theta = math.atan(abs(y1-y2) / abs(x1-x2))
                angle = int(180 * theta / math.pi * nscale)
                if y2 < y1:
                    angle = -angle  # rotate函数逆时针旋转为正。y2<y1时，需要顺时针旋转;
            # 竖直线
            else:
                theta = math.atan(abs(x1-x2) / abs(y1-y2))
                angle = int(180 * theta / math.pi * nscale)
                if y2 > y1:
                    angle = -angle

            # 避开不合理的直线
            if abs(angle) < MAX_ROATATE_ANGLE * nscale:
                all_angles.append(angle)

    # 计算所有角度中，最多的相同角度是哪个角度。该角度即为旋转角度
    distict_angles = list(set(all_angles))
    print(distict_angles)
    angle_cnt = [all_angles.count(distict_angles[i]) for i in range(len(distict_angles))]
    ind = angle_cnt.index(max(angle_cnt))
    final_angle = distict_angles[ind] / nscale
    print(final_angle)
    #
    rotated_img = rotate_img(src, final_angle)
    cv2.imshow("image-lines", src)
    cv2.imshow("rotated", rotated_img)
    cv2.waitKey(0)


def test_posCorrection():
    image_path = "testimg/defect/t2.png"
    image = cv2.imread(image_path)
    pos_correction(image_path)


if __name__ == '__main__':
    test_posCorrection()

