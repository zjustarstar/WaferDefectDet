import cv2
import math
import numpy as np
import copy

def det_line_by_hough(ori_src, resize_scale=6, debug=False):
    '''
    根据霍夫变换检测图中直线
    :param ori_src: 输入的原图
    :param resize_scale: # 为了加快速度进行的缩放
    :param debug: 调试模式，会显示一些中间结果
    :return: 返回直线序列[x1,y1,x2,y2]，注意的是：这些线是在resize_scale尺寸下的坐标
    反正是为了计算旋转角度，所以也无所谓
    '''
    src = cv2.resize(ori_src, (int(ori_src.shape[1] / resize_scale), int(ori_src.shape[0] / resize_scale)))
    src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

    x = cv2.Sobel(src_gray, cv2.CV_16S, 1, 0)
    y = cv2.Sobel(src_gray, cv2.CV_16S, 0, 1)

    # 转换数据并合成
    Scale_absX = cv2.convertScaleAbs(x)  # 格式转换函数
    Scale_absY = cv2.convertScaleAbs(y)
    result = cv2.addWeighted(Scale_absX, 0.5, Scale_absY, 0.5, 0)  # 图像混合

    thre = 100  # 最短的线
    # minLinLength: 能组成一条直线的最少点的数量. 点数量不足的直线将被抛弃.
    # maxLineGap: 能被认为在一条直线上的两点的最大距离

    res_thre, src_edge = cv2.threshold(result, 0, 255, cv2.THRESH_OTSU)
    print(res_thre)
    lines = cv2.HoughLinesP(src_edge, 1, np.pi / 180, thre,
                            minLineLength=100, maxLineGap=10)

    if lines is None:
        return []
    # 如果线过多..
    elif len(lines) > 200:
        plines = copy.deepcopy(lines)
        ratio = 1.2
        for i in range(5):
            print("too much lines, round {}".format(i))
            res_thre, src_edge = cv2.threshold(result, int(res_thre * ratio), 255, cv2.THRESH_BINARY)
            lines = cv2.HoughLinesP(src_edge, 1, np.pi / 180, thre,
                                    minLineLength=100, maxLineGap=30)
            # 如果没线了，返回至最近的一次;
            if len(lines) == 0:
                lines = copy.deepcopy(plines)
                break
            elif len(lines) < 200:
                break
            plines = copy.deepcopy(lines)

    if debug:
        cv2.imshow("edge map", src_edge)

    # 将最外层的维度去除,然后转为list
    return lines[:,0].tolist()


def det_line_by_lsd(ori_src, resize_scale=4, debug=False):
    LINE_LENGTH_THRE = 70

    img_resize = cv2.resize(ori_src, (int(ori_src.shape[1] / resize_scale), int(ori_src.shape[0] / resize_scale)))
    img_gray = cv2.cvtColor(img_resize, cv2.COLOR_RGB2GRAY)

    lsd = cv2.createLineSegmentDetector()
    lines = lsd.detect(img_gray)[0]
    # Remove singleton dimension
    lines = lines[:, 0]

    # Filter out the lines whose length is lower than the threshold
    dx = lines[:, 2] - lines[:, 0]
    dy = lines[:, 3] - lines[:, 1]
    lengths = np.sqrt(dx * dx + dy * dy)
    mask = lengths >= LINE_LENGTH_THRE
    lines = lines[mask]

    # 转换为整数坐标,且对线的坐标进行排序,x小的在前
    final_lines = []
    for l in lines:
        line = int(l[0]),int(l[1]),int(l[2]),int(l[3])
        if l[0] > l[2]:
            line = int(l[2]),int(l[3]),int(l[0]),int(l[1])
        final_lines.append(line)

    if debug:
        print("LSD lines:{}".format(len(final_lines)))

    return final_lines


def get_angle_by_lines(lines, debug=False):
    MAX_ROATATE_ANGLE = 15  # 倾斜程度不能超过这个角度

    nscale = 10  # 方便计算;
    all_angles = []
    final_lines = []
    for i in range(len(lines)):
        x1, y1, x2, y2 = lines[i]

        # 水平线
        if abs(x1 - x2) > abs(y1 - y2):
            theta = math.atan(abs(y1 - y2) / abs(x1 - x2))
            angle = int(180 * theta / math.pi * nscale)
            if y2 < y1:
                angle = -angle  # rotate函数逆时针旋转为正。y2<y1时，需要顺时针旋转;
        # 竖直线
        else:
            theta = math.atan(abs(x1 - x2) / abs(y1 - y2))
            angle = int(180 * theta / math.pi * nscale)
            if y2 > y1:
                angle = -angle

        # 避开不合理的直线
        if abs(angle) < MAX_ROATATE_ANGLE * nscale:
            all_angles.append(angle)
            final_lines.append([x1, y1, x2, y2])

    if all_angles is None:
        return -1, []

    # 计算所有角度中，最多的相同角度是哪个角度。该角度即为旋转角度
    distict_angles = list(set(all_angles))
    angle_cnt = [all_angles.count(distict_angles[i]) for i in range(len(distict_angles))]
    ind = angle_cnt.index(max(angle_cnt))
    final_angle = distict_angles[ind] / nscale

    if debug:
        print("distinct angles:")
        print(distict_angles)
        print("final angle:{}".format(final_angle))
        print("angle_cnt:")
        print(angle_cnt)

    return final_angle, final_lines


