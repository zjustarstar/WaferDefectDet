import glob
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
import img_quality_checker as iqc
import patchcore_main as patchcore
import wcf_client as wcfClient
import datetime

patchcore_models = [0]


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

    json_data = {}

    # PP_GetRectileID, 版号读取
    if id == 3:
        img_path = "testimg/ocr/ng.jpg"
        rslt, msg, boxes, txts, scores = ocr.get_rectileID(img_filepath)
        json_data = {"rslt": rslt, "ErrMsg": msg, "ImagePath": img_filepath, "rectID": txts, "rectLoc": boxes}
    # GetReferanceWidth, 获取参考宽度
    elif id == 4:
        img_path = "testimg/temp_matcher/img1.jpg"
        rslt, msg, ref_width = rwc.get_reference_width(img_path)
        ref_width = 300
        json_data = {"rslt": rslt, "ErrMsg": msg, "ImagePath": img_filepath, "Width": ref_width}
    # 缺陷检测
    elif id == 5:
        img_path = request.json.get("ProductProcedurePath")
        defect_type = 0
        bi = request.json.get("BlockIndex")
        score = patchcore.get_single_image_score(img_path, patchcore_models[bi])
        is_defect = 1 if score > 2.0 else 0
        json_data = {"rslt": CFG.RESULT_OK, "ErrMsg": 'OK', "ImagePath": img_filepath,
                     "score": score, "isDefect": is_defect, "DefectType": defect_type}
    # PP_GetCellPattern, 获取pattern
    elif id == 6:
        CurPos = request.json.get("PartCellNo")
        TotalPos = request.json.get("PartCellCount")
        temp_path = request.json.get("TemplatePath")
        CellW = request.json.get("PartCellWidth")
        CellH = request.json.get("PartCellHeight")
        requireCut = request.json.get("RequireCut")
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
            if isDetectProcess:
                rslt, msg, startX, startY, maxVal, angle, cell_img_path, isDefect = pm.pattern_matcher(img_filepath,
                                                                                                       temp_path,
                                                                                                       CurPos, TotalPos,
                                                                                                       requireCut,
                                                                                                       CellW, CellH,
                                                                                                       True,
                                                                                                       isDetectProcess,
                                                                                                       patchcore_models[CurPos])
            else:
                rslt, msg, startX, startY, maxVal, angle, cell_img_path, isDefect = pm.pattern_matcher(img_filepath,
                                                                                                       temp_path,
                                                                                                       CurPos, TotalPos,
                                                                                                       requireCut,
                                                                                                       CellW, CellH,
                                                                                                       True,
                                                                                                       isDetectProcess)

            json_data = {"rslt": rslt, "ErrMsg": msg, "CellStartX": startX,
                         "CellStartY": startY, "MaxMatchVal": maxVal,
                         "Angle": angle, "CellImgPath": cell_img_path, "isDefect": isDefect}
    # 位偏检测
    elif id == 7:
        img_path = "testimg/cross_locate/b1.png"
        rslt, msg, isPosOK = cpc.cross_pos_checker(img_path)
        json_data = {"rslt": rslt, "ErrMsg": msg, "ImagePath": img_filepath, "isPosOK": isPosOK}
    # 缺陷检测的模型训练
    elif id == 8:
        rslt = CFG.RESULT_OK
        msg = "OK"
        block_path = request.json.get("ProductProcedurePath")
        if not os.path.exists(block_path):
            rslt = CFG.RESULT_FAIL
            msg = f'{block_path}路径不存在！'
        else:
            ok_img_path = os.path.join(block_path, "OK")
            weight_path = os.path.join(block_path, "Weights")
            images = glob.glob(os.path.join(ok_img_path, "*.jpg"))
            if not images:
                rslt = CFG.RESULT_FAIL
                msg = f'{block_path}为空文件夹（无JPG文件）！'
            else:
                patchcore.train_data(ok_img_path, weight_path)
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
        CFG.ALG_MATCH_THRESHOLD = request.json.get("MatchThreshold")
        CFG.ALG_Defect_THRESHOLD = request.json.get("DefectThreshold")
        print(CFG.ALG_MATCH_THRESHOLD, CFG.ALG_Defect_THRESHOLD)
        res_path = wcfClient.SetCameraParams(ExposurePow, GainPow)

        json_data = {"rslt": rslt, "ErrMsg": "OK", "Path": res_path}
    # 直接返回当前抓拍的未经任何处理的图像
    elif id == 11:
        rslt = CFG.RESULT_OK
        json_data = {"rslt": rslt, "ErrMsg": '', "PicturePath": img_filepath}
    elif id == 12:
        # print(time.now())
        preImgPath = request.json.get("PreImagePath")
        # preImgPath = CFG.SHARE_DIR + preImgPath
        imgQua = iqc.get_img_quality(img_filepath)
        # 如果比较失败
        if imgQua == CFG.RESULT_FAIL:
            json_data = {"rslt": CFG.RESULT_FAIL, "ErrMsg": 'fail to load image',
                         "CurImgQuality": 0, "curImgPath": img_filepath}
        else:
            json_data = {"rslt": CFG.RESULT_OK, "ErrMsg": '',
                         "CurImgQuality": imgQua, "curImgPath": img_filepath}
        # print(time.now())
    # 设置当前的产品/流程,同时加载模型
    elif id == 13:
        procedureDir = request.json.get("ProductProcedurePath")
        detCount = request.json.get("DetPointCount")
        # 设置一些参数
        CFG.PRODUCT_PROCEDURE_DIR = CFG.SHARE_DIR + procedureDir
        CFG.DET_POINT_COUNT = detCount
        # 加载模型
        for i in range(detCount):
            path = procedureDir + "\\Block" + str(i + 1) + "\\Weights"
            m = patchcore.load_model(path)
            # print(path)
            if m is None:
                ErrMsg = f'Block {i} 模型为空'
                logger.info(ErrMsg)
                json_data = {"rslt": CFG.RESULT_FAIL, "ErrMsg": ErrMsg}
                return json_data
            else:
                patchcore_models.append(m)
        logger.info("load model ok")
        json_data = {"rslt": CFG.RESULT_OK, "ErrMsg": ''}
    elif id == 14:
        procedureDir = request.json.get("ProductProcedurePath")
        temp_path = request.json.get("TemplatePath")
        CellW = request.json.get("PartCellWidth")
        CellH = request.json.get("PartCellHeight")
        SubTempW = request.json.get("SubTempWidth")
        SubTempH = request.json.get("SubTempHeight")
        rslt, msg, startX, startY, \
        cellImgPath, leftBlockPath, \
        rightTempPath, downTempPath = pm.get_multiple_temp_by_match(img_filepath, temp_path,
                                                                                 procedureDir,
                                                                                 CellW, CellH,
                                                                                 SubTempW, SubTempH)
        json_data = {"rslt": rslt, "ErrMsg": msg, "CellStartX": startX,
                     "CellStartY": startY, "CellImgPath": cellImgPath,
                     "LeftBlockPath": leftBlockPath,
                     "RightTempPath": rightTempPath, "DownTempPath": downTempPath}


    return json_data


if __name__ == '__main__':
    print("")
