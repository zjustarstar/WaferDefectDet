import numpy as np
import cv2 as cv
import config as CFG
import os
import copy

# 初始化hog检测
x = [0.011314,0.041145,0.057675,0.130656,0.088781,-0.079858,-0.072506,-0.041080,-0.046738,0.025202,0.000149,-0.025301,-0.019453,-0.085137,-0.106970,-0.057582,-0.030235,0.007541,0.007103,0.011266,0.006384,0.098947,0.082444,-0.058195,-0.059087,-0.019316,0.018208,0.082575,0.011456,-0.011543,0.038405,-0.085470,-0.085269,-0.050091,-0.030271,0.078290,-0.042340,-0.035960,0.017432,0.079057,-0.046214,-0.028014,-0.030239,-0.016573,-0.017372,-0.012556,0.054907,0.102632,0.188807,0.031923,-0.040985,-0.048416,-0.030870,-0.049767,0.019905,0.017895,0.002217,0.082681,-0.012965,-0.022146,-0.039347,-0.019410,0.008513,-0.031284,0.007253,0.035504,0.196752,0.095542,-0.057630,-0.017343,-0.032251,-0.070552,0.105658,0.098690,0.067543,0.109918,-0.098040,-0.048853,-0.021105,0.011177,0.043139,0.051382,0.060616,0.008331,-0.023571,-0.061553,0.164108,0.056854,0.044951,0.093435,-0.055186,0.015133,0.000760,0.147447,0.088393,-0.053860,-0.023555,-0.039840,-0.062287,-0.052095,0.003094,-0.023593,0.063499,0.041933,0.099484,-0.002882,-0.020020,-0.036303,-0.051628,-0.031506,-0.067733,-0.052857,0.062812,0.235697,0.078595,0.014363,-0.007800,-0.049636,-0.035965,-0.024505,0.018484,0.008096,0.106110,0.000176,0.007806,-0.029437,-0.117702,-0.060465,-0.110068,-0.030547,0.053531,0.173072,-0.011683,-0.011793,-0.022245,-0.013313,-0.038171,-0.063675,0.016391,-0.009211,0.055344,-0.038688,0.009852,0.012664,0.052642,-0.019582,-0.030243,-0.026103,-0.108094,-0.018586,-0.045125,0.008997,0.042899,-0.055782,-0.081878,-0.113922,0.017204,-0.025420,0.015135,-0.041481,0.019088,-0.047218,0.043802,0.012356,-0.056025,-0.028689,-0.096228,-0.030209,-0.057538,0.039653,0.087856,-0.061706,-0.030133,-0.068654,0.013588,-0.046902,0.071533,0.018619,0.096368,0.009966,0.044450,0.017646,0.012070,0.030004,0.026043,-0.055250,-0.050131,0.002908,0.034617,0.084468,-0.048613,0.015509,0.046547,-0.049915,-0.035531,-0.050400,-0.008366,0.077339,0.031029,0.046490,0.022449,0.101890,0.111128,-0.007779,-0.030595,-0.004941,-0.052328,0.055245,0.041432,0.046178,0.017803,-0.057158,-0.025395,-0.039695,0.001057,0.002620,0.103242,0.048883,0.016505,0.059221,-0.015432,-0.040746,-0.028975,0.000840,0.025973,0.031434,0.062761,0.015670,0.101946,0.054283,-0.053605,0.003101,-0.017391,-0.054647,0.072458,0.069463,0.019503,0.022086,-0.099341,-0.040739,-0.017738,0.022894,0.054627,0.038055,0.082265,0.032017,0.056166,-0.064366,-0.102193,-0.042648,-0.031961,0.008659,-0.070274,-0.001644,0.016088,0.135588,0.138237,0.040028,0.029980,-0.026429,-0.079364,-0.083545,-0.052794,-0.059138,0.078047,0.119422,0.134011,0.065743,0.009513,-0.017852,-0.033840,0.056532,0.049387,0.095374,0.022352,-0.025772,-0.019773,-0.041537,-0.100915,-0.181572,-0.103535,-0.099788,-0.039331,-0.051711,0.098598,0.077382,0.034799,-0.091145,-0.055693,-0.078202,-0.101765,-0.018436,0.082163,0.119329,0.004338,-0.010219,0.003332,0.054555,-0.022774,-0.060412,-0.006000,0.050059,0.063075,-0.006442,0.049230,0.105385,-0.060479,-0.101123,-0.136691,-0.086414,-0.055443,0.090796,0.028453,0.008728,-0.009052,0.088880,-0.014997,-0.060634,-0.043003,-0.035063,0.019302,-0.017688,0.028859,0.068801,0.052413,-0.013479,-0.055289,-0.041865,-0.058720,0.039663,-0.004015,0.073729,0.130994,-0.044835,-0.049972,-0.072006,-0.027274,-0.029464,0.108335,0.060991,0.073323,0.033894,0.061464,-0.007567,0.003518,-0.005582,-0.011904,0.040435,-0.059787,0.009203,0.055247,-0.023756,-0.059027,-0.011778,-0.027926,0.010287,0.102731,0.006140,0.048762,-0.016555,-0.050610,-0.037161,-0.032174,0.059666,0.159024,0.065864,0.034964,0.015314,-0.063173,0.095805,0.033756,-0.003882,0.030472,-0.062096,-0.042174,-0.050854,-0.016142,0.018140,-0.079300,-0.052260,-0.092418,-0.032884,0.040167,0.134180,0.069076,0.085905,0.034646,0.025773,-0.003054,-0.019341,-0.015868,-0.070494,-0.025843,-0.042265,0.046173,0.057486,0.103321,0.058056,-0.021250,0.005307,-0.119438,-0.045100,-0.016288,0.018936,0.126550,0.056904,0.073561,0.017747,-0.018481,-0.139257,-0.092248,-0.029884,-0.011082,0.029468,0.141461,0.052897,-0.020906,-0.027590,-0.069780,-0.026940,-0.037226,0.023920,0.093520,0.053490,-0.040158,-0.085819,-0.104926,-0.103472,0.015459,0.038334,0.074526,0.062430,-0.047330,0.043827,0.057143,0.052226,-0.089994,-0.035397,-0.012612,0.016116,0.007238,-0.071562,-0.047616,-0.021775,-0.022598,-0.082854,0.000779,0.057219,0.043477,-0.003371,-0.063407,-0.102996,-0.028898,0.000510,-0.029477,0.032158,0.103479,0.120826,0.039458,0.016087,0.085317,0.086201,0.131181,-0.001986,-0.062167,-0.011557,-0.003294,0.007830,0.030850,-0.048823,-0.065754,-0.029885,-0.113016,0.006440,-0.006796,0.005563,0.039552,0.168955,-0.009768,-0.052855,-0.000852,-0.056551,0.010834,-0.003565,0.070441,0.164160,0.120922,0.070770,0.037562,0.078159,-0.123440,-0.140317,-0.103445,-0.092943,0.007258,0.080988,0.011395,-0.076777,0.022546,-0.035317,-0.053463,-0.039132,-0.026387,0.126318,0.068243,-0.038906,-0.028777,0.027334,0.039678,-0.014356,-0.073756,-0.019260,0.097310,-0.028889,-0.043863,-0.012305,0.094843,0.143576,0.119369,-0.014083,0.009775,0.001580,0.076439,-0.032141,-0.085754,-0.003773,-0.042928,-0.067247,-0.067403,-0.035572,0.087313,0.059396,0.070081,0.041330,0.185174,0.075576,-0.055546,-0.094650,-0.050182,-0.022831,0.033576,-0.014416,-0.049663,-0.058342,-0.022962,0.074651,0.049014,0.111198,0.141773,0.108237,-0.012218,-0.045043,-0.062054,-0.053872,0.000870,0.001621,0.073967,0.119276,-0.040191,-0.048971,-0.059311,-0.048155,-0.047775,-0.013645,-0.010767,0.018396,-0.002081,-0.075930,-0.076808,-0.106607,-0.080176,-0.082534,-0.018192,-0.009615,-0.023730,-0.000458,0.059786,0.003190,-0.036214,-0.037003,0.032630,0.009696,-0.015911,0.017935,0.064133,-0.036004,-0.050594,-0.096230,-0.084881,0.047249,0.056943,0.061120,0.046850,0.027098,-0.085720,-0.060415,-0.067734,0.002715,0.050310,0.070854,0.024693,0.018515,-0.019691,-0.079369,-0.048973,-0.034463,0.040693,0.132943,0.121773,0.048109,0.001961,-0.028908,-0.071378,-0.090946,-0.071086,-0.000537,0.047173,0.047281,0.037003,0.037911,0.017214,-0.016783,-0.013305,0.032561,0.145203,0.085545,-0.065811,-0.089528,-0.079666,-0.028957,-0.033680,-0.043741,-0.030335,0.104342,0.145611,0.096378,0.023457,-0.012553,-0.008646,0.009611,-0.005878,-0.026409,0.065878,0.093207,0.113093,0.003727,-0.021256,0.023918,0.053303,-0.002744,0.023011,0.131815,0.050346,-0.064833,-0.079512,-0.079638,-0.036684,0.062233,0.011493,-0.000057,0.053080,0.005609,0.004425,-0.018753,-0.020624,0.079997,-0.013700,-0.029706,-0.009604,0.090969,0.031114,0.076745,-0.003245,-0.038770,-0.024424,-0.037444,0.031432,0.039634,0.098753,0.008367,0.091997,0.003694,-0.046439,0.004013,0.100107,-0.001315,-0.031796,0.016487,-0.078866,-0.018696,-0.016033,0.011498,0.085405,0.131068,0.092526,0.130904,0.117100,-0.061847,-0.053783,-0.048746,-0.020535,0.023317,-0.008174,0.012856,0.007219,0.067545,-0.094512,-0.010841,-0.030869,-0.049089,-0.017596,-0.090146,-0.020915,-0.030602,0.003473,-0.075948,-0.026321,-0.041912,-0.045370,-0.070439,-0.060400,-0.047181,-0.046365,-0.027407,0.039582,0.008280,0.021274,-0.005860,-0.027773,-0.059154,-0.032047,-0.067912,-0.065017,-0.033072,0.041278,0.029864,0.032107,0.004642,-0.057505,-0.069860,-0.048215,-0.056548,0.022161,-0.000420,0.000348,-0.001994,-0.008491,-0.042508,-0.021704,-0.017374,-0.094084,-0.073724,-0.000184,-0.027260,-0.037317,-0.004721,-0.127653,-0.058439,-0.032330,0.016534,0.070679,0.099368,0.068370,0.016858,-0.058790,-0.011510,-0.055876,0.007167,0.078766,0.163907,0.149232,0.064546,-0.000592,0.046019,-0.038163,-0.032316,-0.031174,-0.072286,-0.032153,0.032444,-0.037128,-0.058424,-0.041128,0.022210,-0.042770,-0.024302,-0.060940,0.051812,0.103058,0.066674,0.033556,0.039460,0.036262,-0.019778,-0.002395,0.045228,0.064831,0.085310,0.005558,-0.031724,0.020221,0.054600,0.063735,0.023632,0.077738,0.065457,0.082648,0.016696,-0.036066,-0.032122,0.136515,-0.013116,-0.040740,-0.059184,-0.044057,0.042540,0.028787,0.052706,0.128368,0.136611,0.096668,0.038638,0.071638,-0.074222,-0.037324,0.003004,0.029130,0.074784,0.062653,0.084864,0.054239,0.136682,0.074407,0.075828,0.026817,-0.049344,-0.098002,-0.050031,0.043709,0.016652,0.085217,0.013572,0.063780,-0.000996,-0.064357,-0.069863,0.037107,0.057436,0.046988,0.156617,-0.003964,-0.001629,-0.000784,0.008975,-0.053597,0.002663,0.003339,-0.027596,0.005979,-0.140064,-0.036107,-0.046271,-0.068453,-0.042326,0.015923,0.128287,0.099303,0.107498,-0.083149,-0.037610,-0.048452,-0.065666,-0.073769,-0.052328,-0.009728,-0.014688,0.006849,-0.034879,-0.053733,-0.060372,-0.035553,-0.109988,0.068545,0.029440,-0.013455,0.013498,-0.178925,-0.099996,-0.110373,-0.109605,0.009885,0.037769,-0.002000,-0.022974,0.036628,-0.095970,-0.110833,-0.108128,-0.071921,-0.021004,-2.031360]
winSize = (48, 48)
blockSize = (16, 16)  # 105
blockStride = (8, 8)  # 4 cell
cellSize = (8, 8)
nBin = 9  # 9 bin 3780
# hog create hog 1 win 2 block 3 blockStride 4 cell 5 bin
hog = cv.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nBin)
hog.setSVMDetector(np.array(x))

# 图片缩放尺寸
scale = 0.5


def cross_pos_checker(img_path):
    return CFG.RESULT_OK, "OK", True


def detect_solidcross(img, scale=1):
    newsize = (int(img.shape[1] / scale), int(img.shape[0] / scale))
    img_resize = cv.resize(img, newsize)
    # , hitThreshold=0.5
    rects, scores = hog.detectMultiScale(img_resize, hitThreshold=0.2)
    for i in range(len(rects)):
       rects[i] = rects[i] * scale

    return rects, scores


def show_inner_outer_rect(img, outer_rect, inner_rect):
    r = copy.deepcopy(inner_rect)
    r[2] = r[0] + r[2]
    r[3] = r[1] + r[3]
    o = outer_rect
    cv.rectangle(img, (r[0], r[1]), (r[2], r[3]), (255, 0, 0), 3)
    cv.rectangle(img, (o[0], o[1]), (o[2], o[3]), (0, 0, 255), 3)

    cv.imwrite("finalres.jpg", img)

    # 结果显示
    newsize = (int(img.shape[1] / 6), int(img.shape[0] / 6))
    img_for_show = cv.resize(img, newsize)
    cv.imshow("a", img_for_show)
    cv.waitKey(0)


def show_cross_det_result(img_path, rects, clr=(255, 0, 0), scale=6, save_rect=False):
    draw_rects = copy.deepcopy(rects)
    img = cv.imread(img_path)
    if img is None:
        print("no cross found")
        return

    for i in range(len(draw_rects)):
        r = draw_rects[i]
        draw_rects[i][2] = r[0] + r[2]
        draw_rects[i][3] = r[1] + r[3]

    for (x, y, xx, yy) in draw_rects:
        if save_rect:
            roiImg = img[y:yy, x:xx]
            name = "testimg/cross/{}_{}_{}_{}.jpg".format(x, xx, y, yy)
            cv.imwrite(name, roiImg)
        cv.rectangle(img, (x, y), (xx, yy), clr, 3)
    cv.imwrite("res.jpg", img)

    # 结果显示
    newsize = (int(img.shape[1] / scale), int(img.shape[0] / scale))
    img_for_show = cv.resize(img, newsize)
    cv.imshow("a", img_for_show)
    cv.waitKey(0)


def find_temp(temp_path, src_path, method=2):
    resize_scale = 1
    overlap_thresh = 0.1  # 用于nms
    max_patterns = 6  # 最多有多少个pattern

    tempimg = cv.imread(temp_path)
    if tempimg is None:
        return
    template = cv.resize(tempimg, (int(tempimg.shape[1] / resize_scale), int(tempimg.shape[0] / resize_scale)))
    tempimg_gray = cv.cvtColor(template, cv.COLOR_BGR2GRAY)
    (tH, tW) = template.shape[:2]
    _, temp_binary_img = cv.threshold(tempimg_gray, 0, 255, cv.THRESH_OTSU)
    temp_binary_img = 255 - temp_binary_img
    cv.imshow("temp_edge", temp_binary_img)

    srcimg = cv.imread(src_path)
    srcimg_resize = cv.resize(srcimg, (int(srcimg.shape[1] / resize_scale), int(srcimg.shape[0] / resize_scale)))
    srcimg_gray = cv.cvtColor(srcimg_resize, cv.COLOR_BGR2GRAY)
    thre1, src_binary_img = cv.threshold(srcimg_gray,0, 255, cv.THRESH_OTSU)
    thre2, src_binary_img = cv.threshold(srcimg_gray,thre1-20, 255, cv.THRESH_BINARY)
    src_binary_img = 255 - src_binary_img
    cv.imshow("src_edge", src_binary_img)

    # detect edges in the resized, grayscale image and apply template
    # edged_img = cv2.Canny(gray, 120, 200)
    # cv.equalizeHist(tempimg_gray, tempimg_gray)
    # cv.equalizeHist(srcimg_gray, srcimg_gray)
    if method == 2:
        result = cv.matchTemplate(src_binary_img, temp_binary_img, cv.TM_CCORR_NORMED)
    else:
        # cv.equalizeHist(tempimg_gray, tempimg_gray)
        # cv.equalizeHist(srcimg_gray, srcimg_gray)
        result = cv.matchTemplate(srcimg_gray, tempimg_gray, cv.TM_CCORR_NORMED)

    # 最佳匹配位置
    (minVal, maxVal, minLoc, maxLoc) = cv.minMaxLoc(result)
    cv.rectangle(srcimg_resize, (maxLoc[0], maxLoc[1]), (maxLoc[0]+tW, maxLoc[1]+tH), (0, 0, 255), 2)

    # all_rects = []
    # loc = np.where(result >= 0.75)
    # for i in zip(*loc[::-1]):
    #     conf = result[i[1]][i[0]]
    #     all_rects.append([i[0], i[1], i[0] + tW, i[1] + tH, conf])
    #
    # nms_rect = []
    # if len(all_rects) > 0:
    #     nms_rect = nms(all_rects, overlap_thresh, max_patterns)
    # for r in nms_rect:
    #     cv.rectangle(srcimg_resize, (r[0], r[1]), (r[2], r[3]), (0, 0, 255), 2)

    cv.imwrite("match.jpg", srcimg_resize)
    cv.imshow("match", srcimg_resize)
    cv.waitKey(0)


def get_accurate_outer_rect(img, rectsize=140):
    rows, cols, _ = img.shape
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    rectW, rectH = rectsize, rectsize
    center_size = int(1.0/3.0*rectsize)
    global_minL = 255
    outer_rect, inner_rect = [], []

    offxy = 5
    min_centerL, minL = 255, 255
    minRow, minCol = 0, 0
    # 计算最小亮度的框
    for row in range(offxy, rows-offxy-rectH):
        for col in range(offxy, cols-offxy-rectW):
            row_up, row_dn, col_left, col_right = 0, 0, 0, 0
            sumL = 0

            # 中间区域
            center_row_up, center_row_dn, center_col_left, center_col_right = 0,0,0,0
            center_sumL=0
            for w in range(-2, 3):
                row_up = sum(img_gray[row+w, col:col + rectW])
                row_dn = sum(img_gray[row+w + rectH, col:col + rectW])
                col_left = sum(img_gray[row:row + rectH, col+w])
                col_right = sum(img_gray[row:row + rectH, col+w + rectW])
                sumL += (row_dn + row_up + col_left + col_right)

                # 中间区域
                center_row_up = sum(img_gray[row+w, col+center_size:col + 2*center_size])
                center_row_dn = sum(img_gray[row+w + rectH, col+center_size:col + 2*center_size])
                center_col_left = sum(img_gray[row+center_size:row + 2*center_size, col+w])
                center_col_right = sum(img_gray[row+center_size:row + 2*center_size, col+w + rectW])
                center_sumL += (center_row_dn + center_row_up + center_col_left + center_col_right)

            avgL = sumL / (rectW * 2 * 5 + rectH * 2 * 5)
            avgCenterL = center_sumL / (center_size*4*5)
            if avgL < minL:
                minRow, minCol = row, col
                minL = avgL
                min_centerL = avgCenterL
    final_rect = [minCol, minRow, minCol+rectW, minRow+rectH]    # 最终的rect,x1,y1,x2,y2

    return final_rect, min_centerL


# 从中找到带有黑框的rect
def find_black_outer_rect(img, rects):
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    rectW, rectH = 200, 200
    expand = 70
    global_minL = 255
    outer_rect, inner_rect = [], []
    for r in rects:
        x, y = r[0], r[1]
        minL = 255
        minRow, minCol = 0, 0
        # 计算最小亮度的框
        for row in range(expand, 0, -1):
            for col in range(expand, 0, -1):
                row_up = sum(img_gray[y-row, x-col:x-col+rectW])
                row_dn = sum(img_gray[y-row+rectH, x-col:x-col+rectW])
                col_left = sum(img_gray[y-row:y-row+rectH, x-col])
                col_right = sum(img_gray[y-row:y-row+rectH, x-col+rectW])
                avgL = (row_dn + row_up + col_left + col_right) / (rectW*2+rectH*2)
                if avgL<minL:
                    minRow, minCol = row, col
                    minL = avgL

        # 所有rect中最可能的框
        if minL < global_minL:
            outer_rect = [x-minCol, y-minRow, x-minCol+rectW, y-minRow+rectH]
            for i in range(4):
                outer_rect[i] = max(outer_rect[i], 0)
            inner_rect = r
            global_minL = minL

    # cv.rectangle(img, (outer_rect[0], outer_rect[1]), (outer_rect[2], outer_rect[3]), (0, 0, 255), 2)
    # cv.imwrite("final.jpg", img)

    return outer_rect, inner_rect


def save_rect_img(img, rects, path):
    expand = 20   # 往外扩展一点
    paths = []    # 保存的路径
    for r in rects:
        x, y, w, h = r
        roiImg = img[y-expand:y+h+2*expand, x-expand:x+2*expand+w]
        filename = str(x)+"_"+str(y)+"_"+".jpg"
        f = os.path.join(path, filename)
        paths.append(f)
        cv.imwrite(f, roiImg)
    return paths


def test_only_cross_img(img_path):
    img = cv.imread(img_path)
    if img is None:
        print("fail to load image")

    # avgCenter值小的，就是对应外框的大小
    outer_rect, avgCenterL4 = get_accurate_outer_rect(img, 140)
    outer_rect3, avgCenterL3 = get_accurate_outer_rect(img, 130)
    if avgCenterL3 > avgCenterL4:
        x, y, xx, yy = outer_rect
    else:
        x, y, xx, yy = outer_rect3
    print("avgCenterL4={}, avgCenterL3={}".format(avgCenterL4, avgCenterL3))

    # 内层rect
    inner_rects, scores = detect_solidcross(img, scale=0.5)
    outer_rect_center_x = x + ((xx-x)/2)
    outer_rect_center_y = y + ((yy - y) / 2)

    # 最外层的rect
    cv.rectangle(img, (x, y), (xx, yy), (255, 0, 0), 2)
    # 内层的rect
    for i in range(len(inner_rects)):
        r = inner_rects[i]
        inner_rects[i][2] = r[0] + r[2]
        inner_rects[i][3] = r[1] + r[3]

    for (x, y, xx, yy) in inner_rects:
        cv.rectangle(img, (x, y), (xx, yy), (0,0,255), 2)

    r = inner_rects[0]
    inner_rect_center_x = r[0] + (r[2]-r[0])/2
    inner_rect_center_y = r[1] + (r[3] - r[1]) / 2
    print(outer_rect, inner_rects)
    deltax = inner_rect_center_x - outer_rect_center_x
    deltay = inner_rect_center_y - outer_rect_center_y
    print("delta_x={},delta_y={}".format(deltax, deltay))

    cv.imshow("a", img)
    cv.waitKey(0)


def test_full_image(img_path, scale=3, save_rect=False):
    img = cv.imread(img_path)
    if img is None:
        print("fail to load image")

    rects, scores = detect_solidcross(img, scale=scale)
    print(rects, scores)
    if len(rects) == 0:
        print("fail to find cross position")
        exit(0)
    else:
        print("total {} rects found".format(len(rects)))

    # 保存检测到的区域
    # savepath = "E:\camera_data\cross_marker"
    # fpath = save_rect_img(img, rects, savepath)
    # print(fpath)

    # 检测标记框
    # outer_rect, inner_rect = find_black_outer_rect(img, rects)
    # print(outer_rect,inner_rect)
    show_cross_det_result(img_path, rects, scale=6, save_rect=save_rect)

    # show_inner_outer_rect(img, outer_rect, inner_rect)


if __name__ == '__main__':
    temp_path = "testimg/cross_temp.jpg"
    src_img_path = "testimg/cross_locate/AGT/20230524103622.jpg"
    # find_temp(temp_path, src_img_path, method=2)
    test_full_image(src_img_path, scale=3, save_rect=True)

    img_path = "testimg/cross_double/0627/2676_2850_126_300.jpg"
    # test_only_cross_img(img_path)





