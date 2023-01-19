# -*- coding: utf-8 -*-
from suds.client import Client
from config import CFG
import json



def SetCameraParams(exposure, gainpow):
    client = Client(CFG.WCF_SERVER_ADDRESS)
    # print(client)
    result = client.service.SetMicAndGetPic(exposure, gainpow)
    print(result)

    return result


def GetPicture():
    client =Client(CFG.WCF_SERVER_ADDRESS)
    # print(client)
    result = client.service.GetPicture()
    print(result)

    # jsonStr =json.load(result)
    # print(jsonStr)
    return result
