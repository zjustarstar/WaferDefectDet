import cv2
import numpy as np
import config as CFG
import line_det as lineDet


def det_dicing_channel_by_grad(img_path):
    ori_src = cv2.imread(img_path)
    if ori_src is None:
        msg = "fail to load image"
        return CFG.RESULT_FAIL, msg

    resize_scale = 6
    src = cv2.resize(ori_src, (int(ori_src.shape[1] / resize_scale), int(ori_src.shape[0] / resize_scale)))
    src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

    x = cv2.Sobel(src_gray, cv2.CV_16S, 1, 0)
    y = cv2.Sobel(src_gray, cv2.CV_16S, 0, 1)

    # 转换数据并合成
    Scale_absX = cv2.convertScaleAbs(x)  # 格式转换函数
    Scale_absY = cv2.convertScaleAbs(y)
    result = cv2.addWeighted(Scale_absX, 0.5, Scale_absY, 0.5, 0)  # 图像混合

    res_thre, src_binary = cv2.threshold(result, 0, 255, cv2.THRESH_OTSU)
    src_binary = 255 - src_binary
    print(res_thre)

    cv2.imshow("binary", src_binary)
    cv2.waitKey(0)


def det_dicing_channel_by_lsd(img_path):
    ori_src = cv2.imread(img_path)
    rows, cols, _ = ori_src.shape
    rs = 6
    lines = lineDet.det_line_by_lsd(ori_src, rs, False)

    # 统计直线分布
    dist_thre = 4
    hlines = [0] * int(rows/rs)
    vlines = [0] * int(cols/rs)
    for l in lines:
        x1, y1, x2, y2 = l
        # 竖线分布
        if abs(x1-x2) < dist_thre:
            vlines[x1] = vlines[x1] + abs(y2 - y1)
        if abs(y1-y2) < dist_thre:
            hlines[y1] = hlines[y1] + abs(x2-x1)

    h_line = hlines.index(max(hlines))
    v_line = vlines.index(max(vlines))
    # hlines[h_line] = 0
    # vlines[v_line] = 0
    # h_line2 = hlines.index(max(hlines))
    # v_line2 = vlines.index(max(vlines))

    # hlines = list(filter(lambda x: x>0, hlines))
    # vlines = list(filter(lambda x: x>0, vlines))
    #
    # print("hlines={}".format(hlines))
    # print("vlines={}".format(vlines))

    rimg = cv2.resize(ori_src, (int(ori_src.shape[1] / rs), int(ori_src.shape[0] / rs)))
    # for line in lines:
    #     cv2.line(rimg, (line[0], line[1]), (line[2], line[3]), (255, 0, 0), 2)

    # hline & vline
    cv2.line(rimg, (0, h_line), (cols-1, h_line), (0, 0, 255), 2)
    cv2.line(rimg, (v_line, 0), (v_line, rows-1), (0, 0, 255), 2)
    # cv2.line(rimg, (0, h_line2), (cols-1, h_line2), (255, 0, 0), 2)
    # cv2.line(rimg, (v_line2, 0), (v_line2, rows-1), (255, 0, 0), 2)

    cv2.imshow("final_lines", rimg)
    cv2.waitKey(0)


if __name__ == '__main__':
    img_path = './testimg/pos_corr/1.jpg'
    det_dicing_channel_by_lsd(img_path)

