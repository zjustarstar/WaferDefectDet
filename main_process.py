import cv2
import logging
from flask import json

# 各个算法模块
import config as CFG
import ocr
import pattern_matcher as pm
import ref_width_checker as rwc
import cell_abnormal_detection as abd
import cross_position_checker as cpc


def do_by_commandID(id):
    logger = logging.getLogger(CFG.LOG_NAME)
    logger.info("do_by_commandID")

    # PP_GetRectileID, 版号读取
    if id == 3:
        img_path = "testimg/ocr/ng.jpg"
        rslt, msg, boxes, txts, scores = ocr.get_rectileID(img_path)
        json_data = {"rslt":rslt, "ErrMsg":msg, "ImagePath":img_path, "rectID":txts, "rectLoc":boxes}
    # GetReferanceWidth, 获取参考宽度
    elif id == 4:
        img_path = "testimg/temp_matcher/img1.jpg"
        rslt, msg, ref_width = rwc.get_reference_width(img_path)
        json_data = {"rslt": rslt, "ErrMsg": msg, "ImagePath": img_path, "Width": ref_width}
    # 缺陷检测
    elif id == 5:
        img_path = "testimg/defect/a2.png"
        rslt, msg, is_defect = abd.cell_abnormal_det(img_path)
        json_data = {"rslt": rslt, "ErrMsg": msg, "ImagePath": img_path, "isDefect": is_defect}
    # PP_GetCellPattern, 获取pattern
    elif id == 6:
        img_path = "testimg/temp_matcher/img1.jpg"
        temp_path = "testimg/temp_matcher/temp1.jpg"
        rslt, msg, pattern_rects = pm.pattern_matcher(img_path, temp_path)
        json_data = {"rslt": rslt, "ErrMsg": msg, "ImagePath": img_path, "PatternRects": pattern_rects}
    # 位偏检测
    elif id == 7:
        img_path = "testimg/cross_locate/b1.png"
        rslt, msg, isPosOK = cpc.cross_pos_checker(img_path)
        json_data = {"rslt": rslt, "ErrMsg": msg, "ImagePath": img_path, "isPosOK": isPosOK}

    return json_data


if __name__ == '__main__':
    print("")
