import datetime
import os.path
import random
import time

import cv2
import config as CFG
import logging
import numpy as np
import pattern_pos_correction as ppc
import cell_abnormal_detection as cad


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


# 核心模板匹配程序
def pattern_match_main(img_path, temp_path, pos_corr=True, onlyMostSim=False, anchor=[0, 0]):
    '''
    模板匹配核心程序
    :param img_path: 原图
    :param temp_path: 模板图
    :param pos_corr: 是否对原图进行角度矫正
    :param onlyMostSim: 是否只返回最高匹配度。如果False，则根据anchor，返回离anchor最近的点
    :param anchor:
    :return: CFG.RESULT_OK, msg, startX, startY, maxVal, final_angle, rotated_frame
    '''
    overlap_thresh = 0.3  # 用于nms
    max_patterns = 6  # 最多有多少个pattern
    resize_scale = 6  # 用于匹配时的缩放比例

    msg = "OK"
    template = cv2.imread(temp_path)
    if template is None:
        msg = "fail to load template image"
        return CFG.RESULT_FAIL, msg, 0, 0, 0, 0, None
    template = cv2.resize(template, (int(template.shape[1] / resize_scale), int(template.shape[0] / resize_scale)))
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    # template_edge = cv2.Canny(template_gray, 120, 200)
    (tH, tW) = template.shape[:2]

    ori_frame = cv2.imread(img_path)
    if ori_frame is None:
        msg = "fail to load image"
        return CFG.RESULT_FAIL, msg, 0, 0, 0, 0, None

    # 基于旋转后的图像进行后续操作
    if pos_corr:
        # 可能的倾斜矫正
        rslt, msg, final_angle, rotated_frame = ppc.pos_correction(img_path)
        logger.info("rotated angle={}".format(final_angle))
        if rotated_frame is None:
            return rslt, msg, 0, 0, 0, 0, None
    else:
        rotated_frame = ori_frame
        final_angle = 0

    frame = rotated_frame
    frame = cv2.resize(frame, (int(frame.shape[1] / resize_scale), int(frame.shape[0] / resize_scale)))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect edges in the resized, grayscale image and apply template
    # edged_img = cv2.Canny(gray, 120, 200)
    cv2.equalizeHist(template_gray, template_gray)
    cv2.equalizeHist(gray, gray)
    result = cv2.matchTemplate(gray, template_gray, cv2.TM_CCORR_NORMED)

    # 最佳匹配位置
    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(result)
    logger.info("full pattern match minVal={0}, maxVal={1}".format(minVal, maxVal))

    if maxVal < CFG.ALG_MATCH_THRESHOLD:
        msg = "fail to find matched block"
        return CFG.RESULT_FAIL_NO_MATCHBLOCK, msg, 0, 0, 0, final_angle, None
    # 如果有符合阈值的
    else:
        # 最高相似度的
        (startX, startY) = (int(maxLoc[0]), int(maxLoc[1]))
        # 如果有多个pattern，则返回离anchor最近的
        if not onlyMostSim:
            all_rects = []
            loc = np.where(result >= CFG.ALG_MATCH_THRESHOLD)
            for i in zip(*loc[::-1]):
                conf = result[i[1]][i[0]]
                all_rects.append([i[0], i[1], i[0] + tW, i[1] + tH, conf])

            nms_rect = []
            if len(all_rects) > 0:
                nms_rect = nms(all_rects, overlap_thresh, max_patterns)

            # 找离anchor最近的
            dist = ori_frame.shape[0] + ori_frame.shape[1]
            for i in range(len(nms_rect)):
                if abs(nms_rect[i][0]-anchor[0]) + abs(nms_rect[i][1]-anchor[1]) < dist:
                    dist = abs(nms_rect[i][0]-anchor[0]) + abs(nms_rect[i][1]-anchor[1])
                    startX, startY = nms_rect[i][0], nms_rect[i][1]
                    maxVal = nms_rect[i][4]

    # 在原图中的坐标
    startX, startY = startX * resize_scale, startY * resize_scale
    return CFG.RESULT_OK, msg, startX, startY, maxVal, final_angle, rotated_frame


def pattern_matcher(img_path, temp_path, CurPos, TotalPos, requireCut, CellW, CellH,
                    pos_corr=True, isDectProcess=True):
    '''
    :param img_path: 当前抓拍的图像的路径
    :param temp_path: 模板的路径
    :param CurPos: 当前点位
    :param TotalPos: 总的点位：可选的点位方案是1,4,5,9等
    :param requireCut: 是否要求截图
    :param CellW: 要返回的cell的大小
    :param CellH:
    :param pos_corr: 是否进行角度矫正.默认要进行角度矫正
    :param isDectProcess: 是检测流程还是训练流程。流程不同，矫正后的图像的保存路径不同
    :return: CFG.RESULT_OK, msg, startX, startY, maxVal, final_angle, roi_path, isDefect
    '''

    logger = logging.getLogger(CFG.LOG_NAME)
    ori_frame = cv2.imread(img_path)
    if ori_frame is None:
        msg = "fail to load image"
        return CFG.RESULT_FAIL, msg, 0, 0, 0, 0, None,0
    (tH, tW) = ori_frame.shape[:2]

    # 如果需要切割cell，则左上角作为锚点.
    # 否则，如果只是为了查找位置, 不保存cell,则以中心点作为锚点
    if requireCut:
        anchor = [0, 0]
    else:
        anchor = [int(tW/2), int(tH/2)]
    rlt, msg, startX, startY, maxVal, final_angle, rotated_frame = \
        pattern_match_main(img_path, temp_path, pos_corr, False, anchor)
    if rlt != CFG.RESULT_OK:
        return rlt, msg, startX, startY, 0, final_angle, '', 0

    # 保存cell区域
    ImgH, ImgW = rotated_frame.shape[:2]
    X, Y = max(startX, 0), max(startY, 0)
    # 如果要截图，则需要满足截图尺寸大小
    if requireCut:
        if (X+CellW) > ImgW or (Y+CellH) > ImgH:
            msg = "find match, wrong position"
            return CFG.RESULT_FAIL_WRONG_POSITION, msg, startX, startY, 0, final_angle, '', 0
    # 不截图，只返回匹配坐标
    else:
        msg = "OK, require no cut"
        return CFG.RESULT_OK, msg, startX, startY, maxVal, final_angle, '', 0

    roi_img = rotated_frame[Y:Y+CellH, X:X+CellW]

    isDefect = 0
    # 检测主流程
    if isDectProcess:
        date = datetime.datetime.now()
        tempDir = CFG.SHARE_HISTORY_DIR + date.strftime("%Y%m%d")
        if not os.path.exists(tempDir):
            os.mkdir(tempDir)
        strFile = date.strftime("%H%M%S%f")[:-3] + ".jpg"
        newpath = tempDir + "\\" + strFile

        isDefect, maxVal_defect = cad.isDefect_byPatchCompare(temp_path, roi_img, CFG.ALG_MATCH_THRESHOLD)
        roi_path = newpath
    # 训练流程
    else:
        newpath = temp_path.replace('/Template/', '/Grab/')
        roi_path = newpath

    # 调试用
    if newpath == temp_path:
        roi_path = "cell.jpg"
    cv2.imwrite(roi_path, roi_img)
    return CFG.RESULT_OK, msg, startX, startY, maxVal, final_angle, roi_path, isDefect


def test_matcher():
    image_path = "testimg/temp_matcher/x55.jpg"
    temp_path = "testimg/temp_matcher/x5_temp.jpg"

    # 测试不进行角度矫正
    image = cv2.imread(image_path)
    CellH, CellW = 1000, 1000
    rst, msg, startX, startY, maxVal, angle, roi_path, isDefect = pattern_matcher(image_path, temp_path, 0, 4,
                                                              False, CellW, CellH, True)
    if rst != CFG.RESULT_OK:
        print(msg)
        return

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

