import cv2
import numpy as np

from CamOperation_class import CameraOperation
from MvImport.CameraParams_header import *
from MvImport.MvCameraControl_class import *


class CameraOperation(object):
    def __init__(self):
        self.m_device_list = None
        self.m_camera = MvCamera()
        self.m_selCam_index = 0
        self.m_obj_camoperation = CameraOperation(self.m_camera, self.m_device_list, self.m_selCam_index)
        self.m_isopen = False
        self.m_isgrabbing = False

    def open_device(self):
        # 1.枚举所有摄像机设备
        self.m_device_list = MV_CC_DEVICE_INFO_LIST()  # 当前发现的所有设备信息
        result = MvCamera.MV_CC_EnumDevices(nTLayerType=MV_GIGE_DEVICE, stDevList=self.m_device_list)
        if result != 0:
            print("Enum devices fail!")
            return False
        if self.m_device_list.nDeviceNum == 0:
            print("Find no device!")
            return False
        print("Find %d devices!" % self.m_device_list.nDeviceNum)
        camera_list = []
        for i in range(0, self.m_device_list.nDeviceNum):
            mvcc_dev_info = cast(self.m_device_list.pDeviceInfo[i], POINTER(MV_CC_DEVICE_INFO)).contents
            if mvcc_dev_info.nTLayerType == MV_GIGE_DEVICE:
                print("\ngige device: [%d]" % i)
                ch_user_defined_name = ""
                for per in mvcc_dev_info.SpecialInfo.stGigEInfo.chUserDefinedName:
                    if 0 == per:
                        break
                    ch_user_defined_name = ch_user_defined_name + chr(per)
                print("device user define name: %s" % ch_user_defined_name)

                ch_model_name = ""
                for per in mvcc_dev_info.SpecialInfo.stGigEInfo.chModelName:
                    if 0 == per:
                        break
                    ch_model_name = ch_model_name + chr(per)

                print("device model name: %s" % ch_model_name)

                nip1 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0xff000000) >> 24)
                nip2 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x00ff0000) >> 16)
                nip3 = ((mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x0000ff00) >> 8)
                nip4 = (mvcc_dev_info.SpecialInfo.stGigEInfo.nCurrentIp & 0x000000ff)
                print("current ip: %d.%d.%d.%d\n" % (nip1, nip2, nip3, nip4))
                camera_list.append(
                    "[" + str(i) + "]GigE: " + ch_user_defined_name + " " + ch_model_name + "(" + str(nip1) + "." + str(
                        nip2) + "." + str(nip3) + "." + str(nip4) + ")")

        # 2.选择要打开的相机
        if len(camera_list) > 1:
            self.m_selCam_index = int(input("please input the number of the device to connect:"))

        # 3.打开选择的相机
        self.m_obj_camoperation = CameraOperation(obj_cam=self.m_camera,
                                                  st_device_list=self.m_device_list,
                                                  n_connect_num=self.m_selCam_index)
        result = self.m_obj_camoperation.Open_device()
        if result != 0:
            print("Open device failed!")
            return False
        else:
            self.m_isopen = True
        return True

    # 开始取流
    def start_grabbing(self):
        result = self.m_camera.MV_CC_StartGrabbing()
        if result != 0:
            print("Start grabbing failed!")
            return False
        else:
            self.m_isgrabbing = True
            return True

    def get_image(self):
        st_out_frame = MV_FRAME_OUT()
        memset(byref(st_out_frame), 0, sizeof(st_out_frame))
        result = self.m_camera.MV_CC_GetImageBuffer(stFrame=st_out_frame, nMsec=1000)
        _image = None
        if result == 0:
            print("get one frame: Width[%d], Height[%d], nFrameNum[%d]" % (
                st_out_frame.stFrameInfo.nWidth, st_out_frame.stFrameInfo.nHeight,
                st_out_frame.stFrameInfo.nFrameNum))
            p_data = (c_ubyte * st_out_frame.stFrameInfo.nWidth * st_out_frame.stFrameInfo.nHeight)()
            cdll.msvcrt.memcpy(byref(p_data), st_out_frame.pBufAddr,
                               st_out_frame.stFrameInfo.nWidth * st_out_frame.stFrameInfo.nHeight)
            data = np.frombuffer(p_data,
                                 count=int(st_out_frame.stFrameInfo.nWidth * st_out_frame.stFrameInfo.nHeight),
                                 dtype=np.uint8)
            _image = data.reshape((st_out_frame.stFrameInfo.nHeight, st_out_frame.stFrameInfo.nWidth))

        self.m_camera.MV_CC_FreeImageBuffer(st_out_frame)

        return _image

    # 停止取流
    def stop_grabbing(self):
        result = self.m_camera.MV_CC_StopGrabbing()
        if result != 0:
            print("Stop grabbing failed!")
            return False
        else:
            self.m_isgrabbing = False
            return True

    # 关闭设备
    def close_device(self):
        if self.m_isopen:
            self.m_obj_camoperation.Close_device()
            self.m_isopen = False

        self.m_isgrabbing = False

    def grab_image(self):
        if not self.start_grabbing():
            self.close_device()
            return None

        # 图像保存
        image = self.get_image()
        co.stop_grabbing()

        return image


if __name__ == '__main__':
    co = CameraOperation()
    # 打开设备
    if not co.open_device():
        co.close_device()
        sys.exit(0)
    if not co.start_grabbing():
        co.close_device()
        sys.exit(0)

    # 图像保存
    image = co.get_image()
    # image = cv2.resize(image, (2048, 1024), interpolation=cv2.INTER_AREA)
    # cv2.imshow("image", image)
    # cv2.imwrite("image.jpg", image)

    co.stop_grabbing()
    co.close_device()
