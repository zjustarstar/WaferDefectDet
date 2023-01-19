# -*- coding: utf-8 -*-
from suds.client import Client
import config as CFG
import logging


def SetCameraParams(exposure, gainpow):
    logger = logging.getLogger(CFG.LOG_NAME)

    try:
        client = Client(CFG.WCF_SERVER_ADDRESS)
        # print(client)

        result = client.service.SetMicAndGetPic(exposure, gainpow)
        logger.info("访问WCF服务SetMicAndGetPic返回结果:%s", result)
        # print(result)
    except Exception as e:
        logger.info("访问WCF服务SetMicAndGetPic出错: %s", e)
        print("Error:", e)

    return result


def GetPicture():
    logger = logging.getLogger(CFG.LOG_NAME)
    try:
        client = Client(CFG.WCF_SERVER_ADDRESS)
        # print(client)
        result = client.service.GetPicture()
        logger.info("访问WCF服务GetPicture返回结果:%s", result)
        # print(result)
    except Exception as e:
        logger.info("访问WCF服务GetPicture出错: %s", e)
        print("Error:", e)
        return None

    return result
