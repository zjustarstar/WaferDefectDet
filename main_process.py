import cv2
import logging

# 各个算法模块
import config as CFG
import ocr
import pattern_matcher as pm
import ref_width_checker as rwc
import cell_abnormal_detection as abd
import cross_position_checker as cpc
import pattern_pos_correction as ppc


# 只需要一个commandID作为参数;
def do_by_commandID(id, img_filepath):
    '''
    根据输入的命令进行相应的动作
    :param id: 命令id
    :param img_filepath: 抓拍到的图像的全路径
    :return:
    '''
    logger = logging.getLogger(CFG.LOG_NAME)
    logger.info("do_by_commandID")

    # PP_GetRectileID, 版号读取
    if id == 3:
        img_path = "testimg/ocr/ng.jpg"
        rslt, msg, boxes, txts, scores = ocr.get_rectileID(img_path)
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
        img_path = "testimg/temp_matcher/img1.jpg"
        temp_path = "testimg/temp_matcher/temp1.jpg"
        CurPos = 0    # 当前点位
        TotalPos = 4  # 总的点位.如果大于1，只返回一个cell pattern
        CellW, CellH = 100, 100 # 要裁剪返回的图像的大小
        rslt, msg, startX, startY, angle, cell_img_path = pm.pattern_matcher(img_path, temp_path,
                                                                            CurPos, TotalPos,
                                                                            CellW, CellH)
        json_data = {"rslt": rslt, "ErrMsg": msg, "CellStartX": startX,
                     "CellStartY":startY, "Angle":angle, "CellImgPath": cell_img_path}
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
        rslt, msg, angle, rotateImgPath = ppc.pos_correction_withsave(img_path)
        json_data = {"rslt": rslt, "ErrMsg": msg, "Angle":angle, "RotatedImagePath": rotateImgPath}

    return json_data


if __name__ == '__main__':
    print("")
