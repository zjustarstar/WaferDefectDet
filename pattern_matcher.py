import datetime

import cv2
import config as CFG
import logging
import numpy as np
import pattern_pos_correction as ppc


def init_log():
    logger = logging.getLogger(CFG.LOG_NAME)
    logger.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler("server.log")
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


def nms(dets, thresh, max_result):
    '''
    :param dets:
    :param thresh: 某个pattern区域和已有区域的重叠度超过该值,则去掉
    :param max_result: 最多返回多少个匹配的pattern
    :return:
    '''
    """Pure Python NMS baseline."""
    dets = np.array(dets)
    x1 = dets[:, 0]#第0列
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    #从大到小排列，取index
    order = scores.argsort()[::-1]
    #keep为最后保留的边框的编号
    keep = []
    while order.size > 0:
    #order[0]是当前分数最大的窗口，之前没有被过滤掉，肯定是要保留的
        i = order[0]
        keep.append(i)
        #计算窗口i与其他所以窗口的交叠部分的面积
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        #和anchor矩形的面积比
        ovr = inter / (areas[i])
        #ind为所有与窗口i的面积比小于threshold值的窗口的index，其他窗口此次都被窗口i吸收
        inds = np.where(ovr <= thresh)[0]
        #下一次计算前要把窗口i去除，所有i对应的在order里的位置是0，所以剩下的加1
        order = order[inds + 1]

    # 返回最终的rect
    ret_rect = []
    keep = keep[0:min(len(keep), max_result)]
    for i in keep:
        ret_rect.append([int(x1[i]), int(y1[i]), int(x2[i]), int(y2[i]), (scores[i])])

    return ret_rect


def margin_matcher(edge_src_img, edge_temp_img):
    (tH, tW) = edge_temp_img.shape[:2]
    blockH, blockW = int(tH/2), int(tW/2)

    found = False

    # 左上角,右上角,左下角,右下角
    max_match = 0
    distx, disty = 0, 0
    final_loc = [0, 0]

    marginx = [0, blockW, 0, blockW]
    marginy = [0, 0, blockH, blockH]
    for i in range(4):
        # sub-block
        temp = edge_temp_img[marginy[i]:marginy[i]+blockH, marginx[i]:marginx[i]+blockW]
        result = cv2.matchTemplate(edge_src_img, temp, cv2.TM_CCORR_NORMED)
        (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(result)
        if maxVal > max_match:
            max_match = maxVal
            distx, disty = marginx[i], marginy[i]
            final_loc = maxLoc

    if max_match > CFG.ALG_MATCH_THRESHOLD:
        found = True
        (startX, startY) = (int(final_loc[0]), int(final_loc[1]))
        startX -= distx
        startY -= disty
    else:
        startX, startY = 0, 0

    return found, startX, startY, max_match


def pattern_matcher(img_path, temp_path, CurPos, TotalPos, CellW, CellH, pos_corr=True, isDectProcess=True):
    '''
    :param img_path: 当前抓拍的图像的路径
    :param temp_path: 模板的路径
    :param CurPos: 当前点位
    :param TotalPos: 总的点位：可选的点位方案是1,4,5,9等
    :param CellW: 要返回的cell的大小
    :param CellH:
    :param pos_corr: 是否进行角度矫正.默认要进行角度矫正
    :param isDectProcess: 是检测流程还是训练流程。流程不同，矫正后的图像的保存路径不同
    :return: 要返回当前匹配到的cell pattern的左上角坐标，以及倾斜角度，矫正后的cell 图像路径
    '''

    logger = logging.getLogger(CFG.LOG_NAME)
    overlap_thresh = 0.3   # 用于nms
    max_patterns = 6       # 最多有多少个pattern
    resize_scale = 6       # 用于匹配时的缩放比例

    logger.info("match threshold={0}".format(CFG.ALG_MATCH_THRESHOLD))

    if CurPos < 0 or CellH < 10 or CellW < 10:
        msg = "错误的cell匹配参数"
        return CFG.RESULT_FAIL_WRONGPARAM, msg, 0, 0, 0, 0, ""

    msg = "OK"
    template = cv2.imread(temp_path)
    if template is None:
        msg = "fail to load template image"
        return CFG.RESULT_FAIL, msg, 0, 0, 0, 0, ""
    template = cv2.resize(template, (int(template.shape[1] / resize_scale), int(template.shape[0] / resize_scale)))
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    # template_edge = cv2.Canny(template_gray, 120, 200)
    (tH, tW) = template.shape[:2]

    ori_frame = cv2.imread(img_path)
    if ori_frame is None:
        msg = "fail to load image"
        return CFG.RESULT_FAIL, msg, 0, 0, 0, 0, ""

    # 基于旋转后的图像进行后续操作
    if pos_corr:
        # 可能的倾斜矫正
        rslt, msg, final_angle, rotated_frame = ppc.pos_correction(img_path)
        logger.info("rotated angle={}".format(final_angle))
        if rotated_frame is None:
            return rslt, msg, 0, 0, 0, 0, ""
    else:
        rotated_frame = ori_frame
        final_angle = 0

    frame = rotated_frame
    frame = cv2.resize(frame, (int(frame.shape[1] / resize_scale), int(frame.shape[0] / resize_scale)))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow("rotated", frame)

    # detect edges in the resized, grayscale image and apply template
    # edged_img = cv2.Canny(gray, 120, 200)
    cv2.equalizeHist(template_gray, template_gray)
    cv2.equalizeHist(gray, gray)
    result = cv2.matchTemplate(gray, template_gray, cv2.TM_CCORR_NORMED)

    # 选择最佳匹配位置
    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(result)
    logger.info("full pattern match minVal={0}, maxVal={1}".format(minVal, maxVal))

    if maxVal < CFG.ALG_MATCH_THRESHOLD:
        found, startX, startY, maxVal2 = margin_matcher(gray, template_gray)
        logger.info("part pattern match, maxVal={}".format(maxVal2))
        if not found:
            msg = "fail to find matched block"
            return CFG.RESULT_FAIL_NO_MATCHBLOCK, msg, 0, 0, 0, final_angle, ''
        else:
            print("find match in margin regions, maxVal={}".format(maxVal2))
    else:
        (startX, startY) = (int(maxLoc[0]), int(maxLoc[1]))

    startX, startY = startX * resize_scale, startY * resize_scale

    # 保存cell区域
    X = max(startX, 0)
    Y = max(startY, 0)
    roi_img = rotated_frame[Y:Y+CellH, X:X+CellW]
    if isDectProcess:
        newpath = temp_path.replace('/Template/', '/Grab/')
        roi_path = newpath
    else:
        newpath = temp_path.replace('/Template/', '/Grab/')
        roi_path = newpath

    # 调试用
    if newpath == temp_path:
        roi_path = "cell.jpg"

    cv2.imwrite(roi_path, roi_img)
    return CFG.RESULT_OK, msg, startX, startY, maxVal, final_angle, roi_path

    #print(minVal, maxVal, minLoc, maxLoc)
    # # 总点位只有1，可能有多个pattern
    # elif TotalPos == 1:
    #     # 查找所有匹配值中大于阈值的点;
    #     all_rects = []
    #     loc = np.where(result >= match_thresh)
    #     for i in zip(*loc[::-1]):
    #         conf = result[i[1]][i[0]]
    #         all_rects.append([i[0], i[1], i[0]+tW, i[1]+tH, conf])
    #
    #     nms_rect = []
    #     if len(all_rects)>0:
    #         nms_rect = nms(all_rects, overlap_thresh, max_patterns)
    #     else:
    #         print("no match patterns found")

    # return CFG.RESULT_OK, msg, nms_rect


def test_matcher():
    image_path = "testimg/temp_matcher/dd/cc.jpg"
    temp_path = "testimg/temp_matcher/rotated_temp1.jpg"

    # 测试不进行角度矫正
    image = cv2.imread(image_path)
    CellH, CellW = 1500, 1500
    _, msg, startX, startY, maxVal, angle, roi_path = pattern_matcher(image_path, temp_path, 0, 4,
                                                              CellW, CellH, True)
    print("msg={}, startX={}, starY={}, Angle={}".format(msg, startX, startY, angle))

    res = [[startX, startY, startX + CellW, startY + CellH]]
    if res is None:
        print("no pattern")
        return

    for r in res:
        cv2.rectangle(image, (r[0], r[1]), (r[2], r[3]), (0, 0, 255), 3)

    scale = 6
    image = cv2.resize(image, (int(image.shape[1] / scale), int(image.shape[0] / scale)))
    tempimg = cv2.imread(temp_path)
    tempimg = cv2.resize(tempimg, (int(tempimg.shape[1] / scale), int(tempimg.shape[0] / scale)))
    cv2.imshow("Image", image)
    cv2.imshow("temp", tempimg)
    cv2.imwrite("match_result.jpg", image)
    cv2.waitKey(0)


if __name__ == '__main__':
    logger = init_log()
    test_matcher()

