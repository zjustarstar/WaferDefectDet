from flask import Flask, request, json
import logging
import config as CFG
import os
import json
import wcf_client as wcfClient


# 相机模块
import camera_op as Camera
import main_process as mp

# 传入__name__初始化一个Flask实例
app = Flask(__name__)


def init_camera():
    camera = Camera.xscCameraOperation()
    # 打开设备
    if not camera.open_device():
        camera.close_device()
    return camera


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


# app.route装饰器映射URL和执行的函数。这个设置将根URL映射到了hello_world函数上
@app.route('/')
def hello_world():
    return 'Hello World!'


@app.route('/post', methods=["post", "get"])
def post():
    if type(request) == type(None) or type(request.data) == type(None):
        info = 'ERROR:  request is None :{0} request.data is None :{1}'.format(
            type(request) == type(None), type(request.data) == type(None))
        return ""

    print(request.data)

    if request.method == "POST":
        command_id = request.json.get(CFG.COMMAND_ID)
        logger.info("request:{0}".format(request.data))
        if command_id is None:
            return ""

        # frame = camera.grab_image()

        # 获取图像.
        imgPath = wcfClient.GetPicture()
        if not os.path.exists(imgPath):
            json_result = {"rslt": CFG.RESULT_FAIL, "ErrMsg": "picture path not exist"}
            return json.dumps(json_result)

        json_result = mp.do_by_commandID(command_id, imgPath, request)
        logger.info("result:{0}\n".format(json_result))

        return json.dumps(json_result)


if __name__ == '__main__':
    # 运行本项目，host=0.0.0.0可以让其他电脑也能访问到该网站，port指定访问的端口。
    # 默认的host是127.0.0.1，port为8888

    logger = init_log()
    #camera = init_camera()
    app.run(host='0.0.0.0', port=8888)
