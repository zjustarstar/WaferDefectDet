import logging
import os
# 各个算法模块
import config as CFG
import ocr
import pattern_matcher as pm
import ref_width_checker as rwc
import cell_abnormal_detection as abd
import cross_position_checker as cpc
import pattern_pos_correction as ppc

import wcf_client as wcfClient


# 只需要一个commandID作为参数;
def do_by_commandID(id, img_filepath, request):
    '''
    根据输入的命令进行相应的动作
    :param id: 命令id
    :param img_filepath: 抓拍到的图像的全路径
    :param request:对方发过来的请求，以json格式封装，带有各种参数
    :return:
    '''
    logger = logging.getLogger(CFG.LOG_NAME)
    logger.info("do_by_commandID")

    # PP_GetRectileID, 版号读取
    if id == 3:
        img_path = "testimg/ocr/ng.jpg"
        rslt, msg, boxes, txts, scores = ocr.get_rectileID(img_filepath)
        json_data = {"rslt":rslt, "ErrMsg":msg, "ImagePath":img_filepath, "rectID":txts, "rectLoc":boxes}
    # GetReferanceWidth, 获取参考宽度
    elif id == 4:
        img_path = "testimg/temp_matcher/img1.jpg"
        rslt, msg, ref_width = rwc.get_reference_width(img_path)
        ref_width = 300
        json_data = {"rslt": rslt, "ErrMsg": msg, "ImagePath": img_filepath, "Width": ref_width}
    # 缺陷检测
    elif id == 5:
        img_path = "testimg/defect/a2.png"
        rslt, msg, is_defect, defect_type = abd.cell_abnormal_det(img_path)
        json_data = {"rslt": rslt, "ErrMsg": msg, "ImagePath": img_filepath,
                     "isDefect": is_defect, "DefectType": defect_type}
    # PP_GetCellPattern, 获取pattern
    elif id == 6:
        CurPos = request.json.get("PartCellNo")
        TotalPos = request.json.get("PartCellCount")
        temp_path = request.json.get("TemplatePath")
        CellW = request.json.get("PartCellWidth")
        CellH = request.json.get("PartCellHeight")
        isDetectProcess = request.json.get("isDetectionProcess")


        # print(CurPos,TotalPos,temp_path,CellW,CellH)

        # img_path = "testimg/temp_matcher/img1.jpg"
        # temp_path = "testimg/temp_matcher/temp1.jpg"
        # CurPos = 0    # 当前点位
        # TotalPos = 4  # 总的点位.如果大于1，只返回一个cell pattern
        # CellW, CellH = 100, 100 # 要裁剪返回的图像的大小
        if temp_path is None:
            msg = "template path is None"
            json_data = {"rslt": CFG.RESULT_FAIL, "ErrMsg": msg, "CellStartX": 0,
                         "CellStartY": 0, "Angle": 0, "CellImgPath": ''}
        else:
            temp_path = CFG.SHARE_DIR + temp_path
            rslt, msg, startX, startY, maxVal, angle, cell_img_path = pm.pattern_matcher(img_filepath, temp_path,
                                                                                CurPos, TotalPos,
                                                                                CellW, CellH, True, isDetectProcess)
            json_data = {"rslt": rslt, "ErrMsg": msg, "CellStartX": startX,
                         "CellStartY": startY, "MaxMatchVal": maxVal,
                         "Angle": angle, "CellImgPath": cell_img_path}
    # 位偏检测
    elif id == 7:
        img_path = "testimg/cross_locate/b1.png"
        rslt, msg, isPosOK = cpc.cross_pos_checker(img_path)
        json_data = {"rslt": rslt, "ErrMsg": msg, "ImagePath": img_filepath, "isPosOK": isPosOK}
    # 缺陷检测的模型训练
    elif id == 8:
        rslt = 0
        msg = "OK"
        json_data = {"rslt": rslt, "ErrMsg": msg}
    # 确认cell图像的角度，并将矫正后的图像的地址返回
    elif id == 9:
        img_path = "testimg/defect/cc.png"
        rslt, msg, angle, rotateImgPath = ppc.pos_correction_withsave(img_filepath)

        json_data = {"rslt": rslt, "ErrMsg": msg, "Angle": angle, "RotatedImagePath": rotateImgPath}
    # 调节相机的增益和曝光
    elif id == 10:
        rslt = CFG.RESULT_OK
        ExposurePow = request.json.get("Exposure")
        GainPow = request.json.get("Gain")
        CFG.ALG_MATCH_THRESHOLD = request.json.get("Threshold")
        res_path = wcfClient.SetCameraParams(ExposurePow, GainPow)

        json_data = {"rslt": rslt, "ErrMsg": "OK", "Path": res_path}

    return json_data


if __name__ == '__main__':
    print("")
